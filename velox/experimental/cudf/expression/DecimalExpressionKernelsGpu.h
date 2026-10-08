/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {

// Outcome of a checked decimal binary op. Division-by-zero (divide or modulo
// with a zero divisor) is kept distinct from arithmetic overflow so the host
// can raise the matching Velox/Presto CPU error (e.g. "Division by zero" /
// "Modulus by zero" vs "Decimal overflow in ...").
enum class DecimalBinaryOpStatus : int32_t {
  kOk = 0,
  kOverflow = 1,
  kDivisionByZero = 2,
};

/// A decimal literal operand decoded on the host once, so the per-batch
/// kernels take it by value instead of reading the cuDF scalar back from the
/// device on every call.
struct DecimalScalarValue {
  cudf::data_type type;
  bool valid;
  __int128_t value;
};

/// Decodes a DECIMAL64 or DECIMAL128 cuDF scalar. Reads the device-resident
/// validity and value, so it synchronizes \p stream.
DecimalScalarValue decodeDecimalScalar(
    const cudf::scalar& scalar,
    cuda::stream_ref stream);

/// Collects the status of every checked decimal kernel launched on one stream,
/// so the host reads them all with one synchronization instead of one per
/// launch.
///
/// While a ScopedDeferredDecimalStatus is active on the calling thread, a
/// launch on stream() records into its own flag and returns without
/// synchronizing. check() then raises the error of the first failing launch in
/// launch order, which is the error an immediate per-launch check would have
/// raised: later launches may compute on the placeholder values a failed launch
/// wrote, but they cannot change which error is reported.
class DeferredDecimalStatus {
 public:
  DeferredDecimalStatus(
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  /// Collector active on the calling thread, or nullptr.
  static DeferredDecimalStatus* current();

  cuda::stream_ref stream() const {
    return stream_;
  }

  /// Returns a zeroed device flag for one launch of \p op on stream().
  int32_t* reserve(cudf::binary_operator op);

  /// Reads every recorded flag with one synchronization of stream() and throws
  /// the error of the first failing launch.
  void check();

 private:
  cuda::stream_ref stream_;
  rmm::device_async_resource_ref mr_;
  rmm::device_uvector<int32_t> flags_;
  std::vector<cudf::binary_operator> ops_;
};

/// Makes \p status the calling thread's collector for its lifetime.
class ScopedDeferredDecimalStatus {
 public:
  explicit ScopedDeferredDecimalStatus(DeferredDecimalStatus& status);
  ~ScopedDeferredDecimalStatus();

  ScopedDeferredDecimalStatus(const ScopedDeferredDecimalStatus&) = delete;
  ScopedDeferredDecimalStatus& operator=(const ScopedDeferredDecimalStatus&) =
      delete;

 private:
  DeferredDecimalStatus* previous_;
};

// CUDA implementations that return {result, status}. The status is tracked with
// a single device-side flag (OR'd via a per-thread register, then one
// atomicOr per thread), matching the fail-fast semantics of Presto / Velox
// CPU decimal arithmetic; no per-row (O(n)) status column is allocated.
std::pair<std::unique_ptr<cudf::column>, DecimalBinaryOpStatus>
decimalBinaryOperationWithOverflow(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<cudf::column>, DecimalBinaryOpStatus>
decimalBinaryOperationWithOverflow(
    const cudf::column_view& lhs,
    const DecimalScalarValue& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<cudf::column>, DecimalBinaryOpStatus>
decimalBinaryOperationWithOverflow(
    const DecimalScalarValue& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

namespace detail {

/// Maps the raw status bits a checked decimal kernel ORs into its flag to a
/// DecimalBinaryOpStatus.
DecimalBinaryOpStatus decodeDecimalBinaryOpStatus(int32_t flag);

/**
 * @brief Decodes a cuDF decimal scalar into a raw __int128_t payload.
 *
 * Shared by the divide (host) and binary-op (device-launch) paths so the
 * DECIMAL64/DECIMAL128 scalar unpacking lives in exactly one place.
 *
 * @param s DECIMAL64 or DECIMAL128 fixed-point scalar.
 * @param stream CUDA stream used to read the device-resident scalar value.
 * @return The scalar's underlying integer representation as __int128_t.
 */
__int128_t getDecimalScalarValue(
    const cudf::scalar& s,
    cuda::stream_ref stream);

/**
 * @brief Dispatches a per-row device loop for fixed-point decimal division.
 *
 * Computes (lhs * rescaleFactor) / rhs with half-away-from-zero rounding on the
 * remainder, writing into out. The host must install the null stencil on out
 * (bitmask_and of the operands) before launch; the kernel skips null rows. A
 * zero divisor sets the division-by-zero status bit.
 *
 * @param inType DECIMAL64 or DECIMAL128 input storage width selector.
 * @param outType DECIMAL64 or DECIMAL128 output storage width selector.
 * @param lhs Left-hand decimal operand column.
 * @param rhs Right-hand decimal operand column.
 * @param out Mutable output column to write divided values into.
 * @param rescaleFactor Fixed-point scale factor, typically
 * DecimalUtil::kPowersOfTen[aRescale] from the caller.
 * @param stream CUDA stream for kernel execution.
 * @return DecimalBinaryOpStatus: kOk, kOverflow, or kDivisionByZero.
 */
DecimalBinaryOpStatus decimalDivideColumnColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    __int128_t rescaleFactor,
    cuda::stream_ref stream);

/**
 * @brief Fixed-point decimal division with a column lhs and scalar rhs.
 *
 * Same kernel math as decimalDivideColumnColumn, but rhs is a single
 * __int128_t decimal payload (already decoded from a cuDF scalar).
 *
 * @param inType DECIMAL64 or DECIMAL128 input storage width selector.
 * @param outType DECIMAL64 or DECIMAL128 output storage width selector.
 * @param lhs Left-hand decimal operand column.
 * @param rhsValue Right-hand decimal operand as a decoded __int128_t payload.
 * @param out Mutable output column to write divided values into.
 * @param rescaleFactor Fixed-point scale factor, typically
 * DecimalUtil::kPowersOfTen[aRescale] from the caller.
 * @param stream CUDA stream for kernel execution.
 * @return DecimalBinaryOpStatus: kOk, kOverflow, or kDivisionByZero.
 */
DecimalBinaryOpStatus decimalDivideColumnScalar(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    __int128_t rhsValue,
    cudf::mutable_column_view out,
    __int128_t rescaleFactor,
    cuda::stream_ref stream);

/**
 * @brief Fixed-point decimal division with a scalar lhs and column rhs.
 *
 * Same kernel math as decimalDivideColumnColumn, but lhs is a single
 * __int128_t decimal payload and rhs is per-row.
 *
 * @param inType DECIMAL64 or DECIMAL128 input storage width selector.
 * @param outType DECIMAL64 or DECIMAL128 output storage width selector.
 * @param lhsValue Left-hand decimal operand as a decoded __int128_t payload.
 * @param rhs Right-hand decimal operand column.
 * @param out Mutable output column to write divided values into.
 * @param rescaleFactor Fixed-point scale factor, typically
 * DecimalUtil::kPowersOfTen[aRescale] from the caller.
 * @param stream CUDA stream for kernel execution.
 * @return DecimalBinaryOpStatus: kOk, kOverflow, or kDivisionByZero.
 */
DecimalBinaryOpStatus decimalDivideScalarColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    __int128_t lhsValue,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    __int128_t rescaleFactor,
    cuda::stream_ref stream);

} // namespace detail
} // namespace facebook::velox::cudf_velox
