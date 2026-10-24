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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cuda/std/span>
#include <cuda/stream>

#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

struct DecimalSumStateColumns {
  std::unique_ptr<cudf::column> sum;
  std::unique_ptr<cudf::column> count;
};

/** Directly reduces DECIMAL64 input into one DECIMAL128 sum and INT64 count. */
DecimalSumStateColumns reduceDecimal64SumCount(
    const cudf::column_view& input,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Decodes intermediate decimal SUM aggregate state stored as a cuDF STRING
 * column (fixed-size packed bytes per row, converted from Velox VARBINARY) into
 * two device columns: a DECIMAL128 sum with scale -scale (matching Velox
 * intermediate state) and an INT64 partial row count. Handles empty input,
 * all-null state without touching payload buffers, and propagates the source
 * null mask to both outputs when present.
 *
 * @param stateCol STRING column of packed sum/count payloads.
 * @param scale decimal scale used to set the output sum column's scale to
 *        -scale.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return decoded sum and count columns.
 */
DecimalSumStateColumns deserializeDecimalSumState(
    const cudf::column_view& stateCol,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Encodes partial decimal SUM state (DECIMAL64 or DECIMAL128 sums plus INT64
 * counts) into a single STRING column (later converted to Velox VARBINARY):
 * per-row fixed-width payloads and string offsets (INT32 or INT64 depending on
 * total char size and cuDF large-strings settings). The output null mask
 * matches buildStateValidityMask: a row is invalid if the sum or count is null,
 * or the count is zero.
 *
 * @param sumCol per-row partial sums (DECIMAL64 or DECIMAL128).
 * @param countCol per-row INT64 partial row counts.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return STRING column of serialized state.
 */
std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Finalizes AVG from intermediate SUM state: divides each sum by its count on
 * device with decimal-specific rounding (see averageRoundDecimalSum),
 * producing a column of the same decimal type as the sum. Rows are null where
 * buildStateValidityMask marks them invalid (null sum/count or zero count),
 * matching serializeDecimalSumState.
 *
 * @param sumCol per-row partial sums (DECIMAL64 or DECIMAL128).
 * @param countCol per-row INT64 partial row counts.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return per-row decimal average column.
 */
std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

// Native DECIMAL128 SUM state is ROW(sum DECIMAL(38, s), overflow BIGINT) and
// AVG state is ROW(sum DECIMAL(38, s), count BIGINT, overflow BIGINT). The
// value of a state is sum + overflow * 2^127, the CPU
// LongDecimalWithOverflowState contract, so intermediate states may exceed
// DECIMAL(38) or int128 and later cancel. The parent is always valid; sum is
// null for a group without non-null input; count and overflow are never null.
inline constexpr cudf::size_type kDecimal128StateSumChild = 0;
inline constexpr cudf::size_type kDecimal128StateCountChild = 1;

inline constexpr cudf::size_type decimal128StateOverflowChild(bool average) {
  return average ? 2 : 1;
}

// Number of INT64 limb columns splitDecimal128SumLimbs produces.
inline constexpr size_t kDecimal128SumLimbs = 4;

struct Decimal128StateChildren {
  cudf::column_view sum;
  // Set for AVG state only.
  std::optional<cudf::column_view> count;
  cudf::column_view overflow;
};

/**
 * Validates a native DECIMAL128 SUM/AVG state column and returns its sliced
 * children.
 *
 * @param state STRUCT state column.
 * @param average true for the AVG layout, false for SUM.
 * @param stream CUDA stream for device work.
 */
Decimal128StateChildren decimal128StateChildren(
    const cudf::column_view& state,
    bool average,
    cuda::stream_ref stream);

/**
 * Splits DECIMAL128 values into kDecimal128SumLimbs INT64 columns of 32-bit
 * limbs, least significant first. Each limb has the null mask of 'values', so
 * per-group INT64 SUMs of the limbs are exact for fewer than 2^31 rows per
 * group and are null exactly where the decimal SUM would be.
 *
 * @param values DECIMAL128 column.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for the limb columns.
 */
std::vector<std::unique_ptr<cudf::column>> splitDecimal128SumLimbs(
    const cudf::column_view& values,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

struct Decimal128SumOverflowColumns {
  std::unique_ptr<cudf::column> sum;
  std::unique_ptr<cudf::column> overflow;
};

/**
 * Combines per-group limb SUMs and overflow SUMs into canonical (sum,
 * overflow) columns: the exact value is sum + overflow * 2^127 with
 * |sum| < 2^127, and both share its sign. sum is null where limbSums[0] is
 * null; overflow is never null.
 *
 * @param limbSums kDecimal128SumLimbs INT64 limb sums.
 * @param overflowSum INT64 overflow sums, or nullptr for raw input.
 * @param scale Velox decimal scale of the sum.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for the outputs.
 */
Decimal128SumOverflowColumns assembleDecimal128SumState(
    cuda::std::span<const cudf::column_view> limbSums,
    const cudf::column_view* overflowSum,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Finalizes DECIMAL128 SUM state like the CPU DecimalSumAggregate: throws the
 * same VeloxUserError when a group's value does not fit DECIMAL(38).
 *
 * @param sum state sums; null rows produce null.
 * @param overflow state overflow counts.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for the output.
 * @return DECIMAL(38, s) sums.
 */
std::unique_ptr<cudf::column> finalizeDecimal128Sum(
    const cudf::column_view& sum,
    const cudf::column_view& overflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Finalizes DECIMAL128 AVG state with DecimalUtil::computeAverage. Rows are
 * null where the sum is null or the count is zero.
 *
 * @param sum state sums.
 * @param count INT64 state counts.
 * @param overflow state overflow counts.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for the output.
 */
std::unique_ptr<cudf::column> finalizeDecimal128Average(
    const cudf::column_view& sum,
    const cudf::column_view& count,
    const cudf::column_view& overflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Builds a native DECIMAL128 SUM/AVG state column from its children.
 *
 * @param sum DECIMAL128 sums.
 * @param count INT64 counts for AVG, nullptr for SUM.
 * @param overflow INT64 overflow counts.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for the struct.
 */
std::unique_ptr<cudf::column> makeDecimal128StateColumn(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    std::unique_ptr<cudf::column> overflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
