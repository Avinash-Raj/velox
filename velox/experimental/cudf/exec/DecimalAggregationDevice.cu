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

#include "velox/experimental/cudf/exec/DecimalAggregationDevice.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_for.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/transform.h>

#include <concepts>
#include <cstdint>
#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

// Mirrors the CPU LongDecimalWithOverflowState layout so serialized SUM state
// is interchangeable between CPU and GPU aggregation.
struct DecimalSumState {
  int64_t count; // count of non-null input rows aggregated
  int64_t overflow; // net int128 carries: the value is sum + overflow * 2^127
  uint64_t lower; // lower 64 bits of the decimal sum
  int64_t upper; // upper 64 bits of the decimal sum (signed)
};

using UInt128 = unsigned __int128;

// DecimalUtil::kOverflowMultiplier.
constexpr UInt128 kOverflowMultiplier = UInt128{1} << 127;

constexpr __int128_t kMinInt128 = static_cast<__int128_t>(kOverflowMultiplier);

constexpr __int128_t decimal128Bound() {
  __int128_t bound = 1;
  for (int i = 0; i < 38; ++i) {
    bound *= 10;
  }
  return bound - 1;
}

// DecimalUtil::kLongDecimalMax; kLongDecimalMin is its negation.
constexpr __int128_t kLongDecimalMax = decimal128Bound();

struct DecimalSumCount {
  __int128_t sum;
  int64_t count;
};

struct Decimal64ToSumCount {
  int64_t const* values;
  cudf::bitmask_type const* nullMask;
  cudf::size_type offset;

  __device__ DecimalSumCount operator()(cudf::size_type idx) const {
    return nullMask && !cudf::bit_is_set(nullMask, offset + idx)
        ? DecimalSumCount{0, 0}
        : DecimalSumCount{static_cast<__int128_t>(values[idx]), 1};
  }
};

struct AddDecimalSumCount {
  __device__ DecimalSumCount
  operator()(DecimalSumCount lhs, DecimalSumCount rhs) const {
    return {lhs.sum + rhs.sum, lhs.count + rhs.count};
  }
};

struct StoreDecimalSumCount {
  DecimalSumCount const* result;
  __int128_t* sum;
  int64_t* count;

  __device__ void operator()(cudf::size_type) const {
    *sum = result->sum;
    *count = result->count;
  }
};

static_assert(sizeof(DecimalSumState) == detail::kDecimalSumStateSize);

__device__ __forceinline__ void
splitToWords(int64_t value, int64_t& upper, uint64_t& lower) {
  lower = static_cast<uint64_t>(value);
  upper = value < 0 ? -1 : 0;
}

__device__ __forceinline__ void
splitToWords(__int128_t value, int64_t& upper, uint64_t& lower) {
  lower = static_cast<uint64_t>(value);
  upper = static_cast<int64_t>(value >> 64);
}

template <typename OffsetT>
struct FillOffsetsFunctor {
  cuda::std::span<OffsetT> offsets;

  __device__ void operator()(cudf::size_type idx) const {
    int64_t offset = static_cast<int64_t>(idx) * detail::kDecimalSumStateSize;
    offsets[idx] = static_cast<OffsetT>(offset);
  }
};

template <typename SumT, typename OffsetT>
struct PackStateFunctor {
  cuda::std::span<const SumT> sums;
  cuda::std::span<const int64_t> counts;
  // nullptr when the sums carry no overflow.
  const int64_t* overflows;
  cuda::std::span<const OffsetT> offsets;
  uint8_t* chars;

  __device__ void operator()(cudf::size_type idx) const {
    int64_t offset = static_cast<int64_t>(offsets[idx]);
    auto* state = reinterpret_cast<DecimalSumState*>(chars + offset);
    int64_t upper;
    uint64_t lower;
    splitToWords(sums[idx], upper, lower);
    state->count = counts[idx];
    state->overflow = overflows == nullptr ? 0 : overflows[idx];
    state->lower = lower;
    state->upper = upper;
  }
};

struct UnpackStateFunctor {
  cudf::column_device_view state;
  cuda::std::span<__int128_t> sums;
  cuda::std::span<int64_t> counts;
  cuda::std::span<int64_t> overflows;
  int32_t* invalidStateFlag;

  __device__ void operator()(cudf::size_type idx) const {
    if (state.is_null(idx)) {
      return;
    }
    auto const serialized = state.element<cudf::string_view>(idx);
    auto const isAligned = reinterpret_cast<std::uintptr_t>(serialized.data()) %
            alignof(DecimalSumState) ==
        0;
    if (serialized.size_bytes() !=
            static_cast<cudf::size_type>(detail::kDecimalSumStateSize) ||
        !isAligned) {
      atomicOr(invalidStateFlag, 1);
      return;
    }
    auto const* packed =
        reinterpret_cast<const DecimalSumState*>(serialized.data());
    counts[idx] = packed->count;
    overflows[idx] = packed->overflow;
    sums[idx] = (static_cast<__int128_t>(packed->upper) << 64) | packed->lower;
  }
};

// DecimalArithmetic::divideWithRoundUp for a positive divisor: the quotient
// truncates toward zero and, unless noRoundUp, rounds half away from zero.
// Returns the remainder, which has the sign of the dividend.
__device__ __int128_t divideRoundHalfUp(
    __int128_t& quotient,
    __int128_t dividend,
    int64_t divisor,
    bool noRoundUp) {
  const bool negative = dividend < 0;
  const UInt128 magnitude = negative
      ? UInt128{0} - static_cast<UInt128>(dividend)
      : static_cast<UInt128>(dividend);
  const auto unsignedDivisor = static_cast<UInt128>(divisor);
  UInt128 unsignedQuotient = magnitude / unsignedDivisor;
  const UInt128 remainder = magnitude % unsignedDivisor;
  if (!noRoundUp && remainder * 2 >= unsignedDivisor) {
    ++unsignedQuotient;
  }
  quotient = static_cast<__int128_t>(
      negative ? UInt128{0} - unsignedQuotient : unsignedQuotient);
  return static_cast<__int128_t>(negative ? UInt128{0} - remainder : remainder);
}

// DecimalUtil::computeAverage for overflow != 0: divides 2^127 * overflow and
// sum by count separately, then rounds their combined remainder.
__device__ __int128_t
averageWithOverflow(__int128_t sum, int64_t count, int64_t overflow) {
  const auto unsignedCount = static_cast<UInt128>(count);
  const auto unsignedOverflow = static_cast<UInt128>(overflow);
  const UInt128 quotMul =
      (kOverflowMultiplier / unsignedCount) * unsignedOverflow;
  const UInt128 remMul =
      (kOverflowMultiplier % unsignedCount) * unsignedOverflow;
  __int128_t quotSum;
  const __int128_t remSum = divideRoundHalfUp(quotSum, sum, count, true);
  __int128_t remTotal;
  divideRoundHalfUp(
      remTotal,
      static_cast<__int128_t>(remMul + static_cast<UInt128>(remSum)),
      count,
      false);
  return static_cast<__int128_t>(
      quotMul + static_cast<UInt128>(quotSum) + static_cast<UInt128>(remTotal));
}

// Half-up (sum + overflow * 2^127) / count divide for AVG.
template <typename SumT>
struct AvgRoundFunctor {
  cuda::std::span<const SumT> sums;
  cuda::std::span<const int64_t> counts;
  // nullptr when the sums carry no overflow; set only for DECIMAL128 sums.
  const int64_t* overflows;
  cuda::std::span<SumT> out;

  __device__ void operator()(cudf::size_type idx) const {
    auto count = counts[idx];
    if (count == 0) {
      out[idx] = SumT{0};
      return;
    }
    if constexpr (cuda::std::is_same_v<SumT, __int128_t>) {
      if (overflows != nullptr && overflows[idx] != 0) {
        out[idx] = averageWithOverflow(sums[idx], count, overflows[idx]);
        return;
      }
    }
    auto sum = sums[idx];
    using U = cuda::std::make_unsigned_t<SumT>;
    U absSum = sum < 0 ? -static_cast<U>(sum) : static_cast<U>(sum);
    U half = static_cast<U>(count / 2);
    U rounded = (absSum + half) / static_cast<U>(count);
    // Use `U{0} - rounded` below to avoid signed overflow
    out[idx] = static_cast<SumT>(sum < 0 ? U{0} - rounded : rounded);
  }
};

// Moves one 2^127 between sum and overflow where their signs disagree, and
// clears an int128 minimum sum, so both share the sign of the exact value
// sum + overflow * 2^127 and |sum| < 2^127.
__device__ void canonicalizeSumOverflow(__int128_t& sum, int64_t& overflow) {
  if (overflow > 0 && sum < 0) {
    sum = static_cast<__int128_t>(
        static_cast<UInt128>(sum) + kOverflowMultiplier);
    --overflow;
  } else if (overflow < 0 && sum > 0) {
    sum = static_cast<__int128_t>(
        static_cast<UInt128>(sum) - kOverflowMultiplier);
    ++overflow;
  } else if (sum == kMinInt128) {
    sum = 0;
    --overflow;
  }
}

struct AssembleSumOverflowFunctor {
  // Validity of sums; nullptr when every row is valid.
  const cudf::bitmask_type* sumMask;
  cudf::size_type sumOffset;
  __int128_t* sums;
  const int64_t* wraps;
  // Merged states' overflow sums; nullptr for raw input.
  const int64_t* stateOverflows;
  const cudf::bitmask_type* stateOverflowMask;
  cudf::size_type stateOverflowOffset;
  int64_t* overflows;

  __device__ void operator()(cudf::size_type idx) const {
    if (sumMask != nullptr && !cudf::bit_is_set(sumMask, sumOffset + idx)) {
      sums[idx] = 0;
      overflows[idx] = 0;
      return;
    }
    // wraps * 2^128 == (2 * wraps) * 2^127.
    int64_t overflow = 2 * wraps[idx];
    if (stateOverflows != nullptr &&
        (stateOverflowMask == nullptr ||
         cudf::bit_is_set(stateOverflowMask, stateOverflowOffset + idx))) {
      overflow += stateOverflows[idx];
    }
    __int128_t sum = sums[idx];
    canonicalizeSumOverflow(sum, overflow);
    sums[idx] = sum;
    overflows[idx] = overflow;
  }
};

struct AdjustSumForOverflowFunctor {
  const cudf::bitmask_type* sumMask;
  cudf::size_type sumOffset;
  const __int128_t* sums;
  const int64_t* overflows;
  __int128_t* out;
  cudf::size_type* firstInvalidRow;

  __device__ void operator()(cudf::size_type idx) const {
    if (sumMask != nullptr && !cudf::bit_is_set(sumMask, sumOffset + idx)) {
      out[idx] = 0;
      return;
    }
    const auto sum = sums[idx];
    const auto overflow = overflows[idx];
    // DecimalUtil::adjustSumForOverflow: the value fits int128 only when a
    // single overflow is offset by a sum of the opposite sign.
    __int128_t value = sum;
    bool valid = overflow == 0;
    if ((overflow == 1 && sum < 0) || (overflow == -1 && sum > 0)) {
      value = static_cast<__int128_t>(
          kOverflowMultiplier * static_cast<UInt128>(overflow) +
          static_cast<UInt128>(sum));
      valid = true;
    }
    if (!valid || value > kLongDecimalMax || value < -kLongDecimalMax) {
      atomicMin(firstInvalidRow, idx);
    }
    out[idx] = value;
  }
};

template <typename BuildOp>
void launchDeviceFor(
    cudf::size_type size,
    BuildOp buildOp,
    cuda::stream_ref stream) {
  if (size == 0) {
    return;
  }
  auto op = buildOp();
  cub::DeviceFor::ForEachN(
      cuda::counting_iterator<cudf::size_type>{0}, size, op, stream.get());
  CUDF_CUDA_TRY(cudaGetLastError());
}

struct StateValidPredicate {
  cudf::column_device_view sum;
  cudf::column_device_view count;

  __device__ bool operator()(cudf::size_type idx) const {
    if (sum.is_null(idx) || count.is_null(idx)) {
      return false;
    }
    return count.element<int64_t>(idx) != 0;
  }
};

std::pair<cuda::device_buffer<std::byte>, cudf::size_type>
buildStateValidityMaskImpl(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto numRows = sumCol.size();
  if (numRows == 0) {
    return {
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
        0};
  }
  auto sumDeviceView = cudf::column_device_view::create(sumCol, stream);
  auto countDeviceView = cudf::column_device_view::create(countCol, stream);
  StateValidPredicate pred{*sumDeviceView, *countDeviceView};
  // Build a BOOL8 column of per-row validity, then convert via the public API.
  auto bools = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::BOOL8},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto iter = cuda::counting_iterator{0};
  thrust::transform(
      rmm::exec_policy(stream),
      iter,
      iter + numRows,
      bools->mutable_view().begin<bool>(),
      pred);
  auto [mask, nullCount] = cudf::bools_to_mask(bools->view(), stream, mr);
  return {std::move(*mask), nullCount};
}

} // namespace

namespace detail {

void reduceDecimal64SumCount(
    cudf::column_view input,
    cudf::mutable_column_view sum,
    cudf::mutable_column_view count,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      input.type().id() == cudf::type_id::DECIMAL64,
      "Direct decimal reduction requires DECIMAL64 input");
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128 && sum.size() == 1,
      "Direct decimal reduction requires one DECIMAL128 sum output");
  CUDF_EXPECTS(
      count.type().id() == cudf::type_id::INT64 && count.size() == 1,
      "Direct decimal reduction requires one INT64 count output");

  auto indices = cuda::counting_iterator<cudf::size_type>{0};
  auto transform = Decimal64ToSumCount{
      input.data<int64_t>(),
      input.nullable() ? input.null_mask() : nullptr,
      input.offset()};
  auto result = rmm::device_uvector<DecimalSumCount>(1, stream, mr);
  size_t tempStorageBytes = 0;
  CUDF_CUDA_TRY(
      cub::DeviceReduce::TransformReduce(
          nullptr,
          tempStorageBytes,
          indices,
          result.data(),
          input.size(),
          AddDecimalSumCount{},
          transform,
          DecimalSumCount{0, 0},
          stream.get()));
  auto tempStorage = rmm::device_buffer(tempStorageBytes, stream, mr);
  CUDF_CUDA_TRY(
      cub::DeviceReduce::TransformReduce(
          tempStorage.data(),
          tempStorageBytes,
          indices,
          result.data(),
          input.size(),
          AddDecimalSumCount{},
          transform,
          DecimalSumCount{0, 0},
          stream.get()));
  cub::DeviceFor::ForEachN(
      cuda::counting_iterator<cudf::size_type>{0},
      1,
      StoreDecimalSumCount{
          result.data(), sum.data<__int128_t>(), count.data<int64_t>()},
      stream.get());
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename T>
concept OffsetStorageType =
    std::same_as<T, int32_t> || std::same_as<T, int64_t>;

template <typename T>
concept DecimalSumStorageType =
    std::same_as<T, int64_t> || std::same_as<T, __int128_t>;

template <typename SumT, typename OffsetT>
concept ValidDecimalPackStorageTypes =
    DecimalSumStorageType<SumT> && OffsetStorageType<OffsetT>;

struct fillOffsetsForDecimalSumStateKernel {
  cudf::mutable_column_view offsetsView;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename OffsetT>
    requires OffsetStorageType<OffsetT>
  void operator()() const {
    launchDeviceFor(
        numRows + 1,
        [&] {
          return FillOffsetsFunctor<OffsetT>{cuda::std::span<OffsetT>{
              offsetsView.data<OffsetT>(), static_cast<size_t>(numRows) + 1}};
        },
        stream);
  }

  template <typename OffsetT>
    requires(!OffsetStorageType<OffsetT>)
  void operator()() const {
    CUDF_FAIL("Invalid offset type for decimal sum state");
  }
};

struct averageRoundDecimalSumKernel {
  cudf::column_view sumCol;
  const int64_t* counts;
  const int64_t* overflows;
  cudf::mutable_column_view outView;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename SumT>
    requires DecimalSumStorageType<SumT>
  void operator()() const {
    CUDF_EXPECTS(
        overflows == nullptr || (std::same_as<SumT, __int128_t>),
        "Decimal average overflow requires DECIMAL128 sums");
    auto const n = static_cast<size_t>(numRows);
    launchDeviceFor(
        numRows,
        [&] {
          return AvgRoundFunctor<SumT>{
              cuda::std::span<const SumT>{sumCol.data<SumT>(), n},
              cuda::std::span<const int64_t>{counts, n},
              overflows,
              cuda::std::span<SumT>{outView.data<SumT>(), n}};
        },
        stream);
  }

  template <typename SumT>
    requires(!DecimalSumStorageType<SumT>)
  void operator()() const {
    CUDF_FAIL("Invalid sum type for decimal average");
  }
};

struct packDecimalSumStateKernel {
  cudf::column_view sumCol;
  const int64_t* counts;
  const int64_t* overflows;
  cudf::column_view offsetsView;
  uint8_t* chars;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename SumT, typename OffsetT>
    requires ValidDecimalPackStorageTypes<SumT, OffsetT>
  void operator()() const {
    auto const n = static_cast<size_t>(numRows);
    auto const sums = sumCol.data<SumT>();
    launchDeviceFor(
        numRows,
        [&] {
          return PackStateFunctor<SumT, OffsetT>{
              cuda::std::span<const SumT>{sums, n},
              cuda::std::span<const int64_t>{counts, n},
              overflows,
              cuda::std::span<const OffsetT>{offsetsView.data<OffsetT>(), n},
              chars};
        },
        stream);
  }

  template <typename SumT, typename OffsetT>
    requires(!ValidDecimalPackStorageTypes<SumT, OffsetT>)
  void operator()() const {
    CUDF_FAIL("Invalid types for decimal sum state pack");
  }
};

void fillOffsetsForDecimalSumState(
    cudf::type_id offsetType,
    cudf::mutable_column_view offsetsView,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::type_dispatcher(
      cudf::data_type{offsetType},
      fillOffsetsForDecimalSumStateKernel{offsetsView, numRows, stream});
}

bool unpackDecimalSumState(
    cudf::column_view stateCol,
    cudf::mutable_column_view sumView,
    cudf::mutable_column_view countView,
    cudf::mutable_column_view overflowView,
    cuda::stream_ref stream) {
  auto const numRows = stateCol.size();
  auto const n = static_cast<size_t>(numRows);
  auto stateDeviceView = cudf::column_device_view::create(stateCol, stream);
  int32_t initialInvalidState{0};
  rmm::device_scalar<int32_t> invalidState{initialInvalidState, stream};
  launchDeviceFor(
      numRows,
      [&] {
        return UnpackStateFunctor{
            .state = *stateDeviceView,
            .sums = cuda::std::span<__int128_t>{sumView.data<__int128_t>(), n},
            .counts = cuda::std::span<int64_t>{countView.data<int64_t>(), n},
            .overflows =
                cuda::std::span<int64_t>{overflowView.data<int64_t>(), n},
            .invalidStateFlag = invalidState.data()};
      },
      stream);
  return invalidState.value(stream) == 0;
}

void averageRoundDecimalSum(
    cudf::type_id sumType,
    cudf::column_view sumCol,
    const int64_t* counts,
    const int64_t* overflows,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{sumType},
      averageRoundDecimalSumKernel{
          sumCol, counts, overflows, outView, numRows, stream});
}

void packDecimalSumState(
    cudf::type_id sumType,
    cudf::type_id offsetType,
    cudf::column_view sumCol,
    const int64_t* counts,
    const int64_t* overflows,
    cudf::column_view offsetsView,
    uint8_t* chars,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{sumType},
      cudf::data_type{offsetType},
      packDecimalSumStateKernel{
          sumCol, counts, overflows, offsetsView, chars, numRows, stream});
}

void assembleDecimalSumOverflow(
    cudf::mutable_column_view sum,
    cudf::column_view wraps,
    const cudf::column_view* stateOverflowSum,
    cudf::mutable_column_view overflow,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128 &&
          wraps.type().id() == cudf::type_id::INT64 &&
          overflow.type().id() == cudf::type_id::INT64,
      "Decimal sum overflow requires DECIMAL128 sums and INT64 overflow");
  CUDF_EXPECTS(
      wraps.size() == sum.size() && overflow.size() == sum.size() &&
          (stateOverflowSum == nullptr ||
           stateOverflowSum->size() == sum.size()),
      "Decimal sum overflow columns must have the same size");
  CUDF_EXPECTS(
      stateOverflowSum == nullptr ||
          stateOverflowSum->type().id() == cudf::type_id::INT64,
      "Decimal sum state overflow must be INT64");
  launchDeviceFor(
      sum.size(),
      [&] {
        return AssembleSumOverflowFunctor{
            .sumMask = sum.nullable() ? sum.null_mask() : nullptr,
            .sumOffset = sum.offset(),
            .sums = sum.data<__int128_t>(),
            .wraps = wraps.data<int64_t>(),
            .stateOverflows = stateOverflowSum == nullptr
                ? nullptr
                : stateOverflowSum->data<int64_t>(),
            .stateOverflowMask =
                stateOverflowSum == nullptr || !stateOverflowSum->nullable()
                ? nullptr
                : stateOverflowSum->null_mask(),
            .stateOverflowOffset =
                stateOverflowSum == nullptr ? 0 : stateOverflowSum->offset(),
            .overflows = overflow.data<int64_t>()};
      },
      stream);
}

cudf::size_type adjustDecimalSumForOverflow(
    cudf::column_view sum,
    cudf::column_view overflow,
    cudf::mutable_column_view out,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128 &&
          overflow.type().id() == cudf::type_id::INT64 &&
          out.type().id() == cudf::type_id::DECIMAL128,
      "Decimal sum overflow requires DECIMAL128 sums and INT64 overflow");
  CUDF_EXPECTS(
      overflow.size() == sum.size() && out.size() == sum.size(),
      "Decimal sum overflow columns must have the same size");
  constexpr auto kNoInvalidRow = std::numeric_limits<cudf::size_type>::max();
  rmm::device_scalar<cudf::size_type> firstInvalidRow{kNoInvalidRow, stream};
  launchDeviceFor(
      sum.size(),
      [&] {
        return AdjustSumForOverflowFunctor{
            .sumMask = sum.nullable() ? sum.null_mask() : nullptr,
            .sumOffset = sum.offset(),
            .sums = sum.data<__int128_t>(),
            .overflows = overflow.data<int64_t>(),
            .out = out.data<__int128_t>(),
            .firstInvalidRow = firstInvalidRow.data()};
      },
      stream);
  auto const row = firstInvalidRow.value(stream);
  return row == kNoInvalidRow ? -1 : row;
}

std::pair<cuda::device_buffer<std::byte>, cudf::size_type>
buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return buildStateValidityMaskImpl(sumCol, countCol, stream, mr);
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
