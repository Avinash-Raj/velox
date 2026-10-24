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
// TODO: Track int128 overflow as the CPU does (DecimalUtil::addWithOverflow);
// the `overflow` field is reserved for that and is always 0 until then.
struct DecimalSumState {
  int64_t count; // count of non-null input rows aggregated
  int64_t overflow; // net int128 carries (CPU parity); always 0 on GPU for now
  uint64_t lower; // lower 64 bits of the decimal sum
  int64_t upper; // upper 64 bits of the decimal sum (signed)
};

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
  cuda::std::span<const OffsetT> offsets;
  uint8_t* chars;

  __device__ void operator()(cudf::size_type idx) const {
    int64_t offset = static_cast<int64_t>(offsets[idx]);
    auto* state = reinterpret_cast<DecimalSumState*>(chars + offset);
    int64_t upper;
    uint64_t lower;
    splitToWords(sums[idx], upper, lower);
    state->count = counts[idx];
    state->overflow = 0;
    state->lower = lower;
    state->upper = upper;
  }
};

struct UnpackStateFunctor {
  cudf::column_device_view state;
  cuda::std::span<__int128_t> sums;
  cuda::std::span<int64_t> counts;
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
    sums[idx] = (static_cast<__int128_t>(packed->upper) << 64) | packed->lower;
  }
};

// Half-up sum/count divide for AVG.
template <typename SumT>
struct AvgRoundFunctor {
  cuda::std::span<const SumT> sums;
  cuda::std::span<const int64_t> counts;
  cuda::std::span<SumT> out;

  __device__ void operator()(cudf::size_type idx) const {
    auto count = counts[idx];
    if (count == 0) {
      out[idx] = SumT{0};
      return;
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

// DecimalUtil::kOverflowMultiplier: the weight of one net overflow.
__device__ __forceinline__ constexpr unsigned __int128 overflowMultiplier() {
  return static_cast<unsigned __int128>(1) << 127;
}

// DecimalUtil::kLongDecimalMax, 10^38 - 1.
__device__ __forceinline__ constexpr __int128_t longDecimalMax() {
  return __int128_t{1'000'000'000'000'000'000} * 1'000'000'000'000'000'000 *
      100 -
      1;
}

__device__ __forceinline__ bool isValidRow(
    cudf::bitmask_type const* mask,
    cudf::size_type offset,
    cudf::size_type idx) {
  return mask == nullptr || cudf::bit_is_set(mask, offset + idx);
}

struct SplitDecimal128LimbsFunctor {
  __int128_t const* values;
  int64_t* limb0;
  int64_t* limb1;
  int64_t* limb2;
  int64_t* limb3;

  __device__ void operator()(cudf::size_type idx) const {
    auto const bits = static_cast<unsigned __int128>(values[idx]);
    limb0[idx] = static_cast<uint32_t>(bits);
    limb1[idx] = static_cast<uint32_t>(bits >> 32);
    limb2[idx] = static_cast<uint32_t>(bits >> 64);
    limb3[idx] = static_cast<int32_t>(static_cast<uint32_t>(bits >> 96));
  }
};

struct AssembleDecimal128SumFunctor {
  int64_t const* limb0;
  int64_t const* limb1;
  int64_t const* limb2;
  int64_t const* limb3;
  cudf::bitmask_type const* limbMask;
  cudf::size_type limbOffset;
  int64_t const* overflowIn;
  cudf::bitmask_type const* overflowMask;
  cudf::size_type overflowOffset;
  __int128_t* sum;
  int64_t* overflow;

  __device__ void operator()(cudf::size_type idx) const {
    if (!isValidRow(limbMask, limbOffset, idx)) {
      sum[idx] = 0;
      overflow[idx] = 0;
      return;
    }
    // Accumulate value == carry * 2^127 + low with 0 <= low < 2^127. Limbs 0-2
    // are sums of unsigned 32-bit words and so are non-negative; limb 3 is
    // signed and is split at bit 127 with a floor shift.
    auto const top = limb3[idx];
    int64_t carry =
        (overflowIn && isValidRow(overflowMask, overflowOffset, idx))
        ? overflowIn[idx]
        : 0;
    carry += top >> 31;
    unsigned __int128 low = 0;
    auto add = [&](unsigned __int128 part) {
      low += part;
      carry += static_cast<int64_t>(low >> 127);
      low &= overflowMultiplier() - 1;
    };
    add(static_cast<unsigned __int128>(static_cast<uint64_t>(top) & 0x7fffffff)
        << 96);
    add(static_cast<unsigned __int128>(static_cast<uint64_t>(limb2[idx]))
        << 64);
    add(static_cast<unsigned __int128>(static_cast<uint64_t>(limb1[idx]))
        << 32);
    add(static_cast<unsigned __int128>(static_cast<uint64_t>(limb0[idx])));
    if (carry >= 0 || low == 0) {
      sum[idx] = static_cast<__int128_t>(low);
      overflow[idx] = carry;
    } else {
      // Give sum the sign of the negative value: carry * 2^127 + low ==
      // (carry + 1) * 2^127 - (2^127 - low).
      sum[idx] = -static_cast<__int128_t>(overflowMultiplier() - low);
      overflow[idx] = carry + 1;
    }
  }
};

struct FinalizeDecimal128SumFunctor {
  cudf::column_device_view sum;
  cudf::column_device_view overflow;
  __int128_t* out;
  cudf::size_type* firstInvalidRow;

  __device__ void operator()(cudf::size_type idx) const {
    if (sum.is_null(idx)) {
      out[idx] = 0;
      return;
    }
    auto const value = sum.data<__int128_t>()[idx];
    auto const carry =
        overflow.is_null(idx) ? int64_t{0} : overflow.data<int64_t>()[idx];
    // DecimalUtil::adjustSumForOverflow, then DecimalUtil::valueInRange.
    bool valid = true;
    auto adjusted = value;
    if (carry != 0) {
      valid = (carry == 1 && value < 0) || (carry == -1 && value > 0);
      adjusted = static_cast<__int128_t>(
          static_cast<unsigned __int128>(value) + overflowMultiplier());
    }
    valid =
        valid && adjusted >= -longDecimalMax() && adjusted <= longDecimalMax();
    out[idx] = adjusted;
    if (!valid) {
      atomicMin(firstInvalidRow, idx);
    }
  }
};

// DecimalUtil::divideWithRoundUp with no rescaling, for the instantiations
// DecimalUtil::computeAverage uses.
template <typename R>
__device__ __forceinline__ R
divideWithRoundUp(R& result, R dividend, int64_t divisor, bool noRoundUp) {
  int sign = 1;
  if constexpr (cuda::std::is_signed_v<R>) {
    if (dividend < 0) {
      sign = -1;
      dividend *= -1;
    }
  }
  if (divisor < 0) {
    sign *= -1;
    divisor *= -1;
  }
  R quotient = dividend / divisor;
  R remainder = dividend % divisor;
  if (!noRoundUp && static_cast<int64_t>(remainder) * 2 >= divisor) {
    ++quotient;
  }
  result = quotient * sign;
  return remainder * sign;
}

// DecimalUtil::computeAverage.
__device__ __forceinline__ __int128_t
computeAverage(__int128_t sum, int64_t count, int64_t overflow) {
  __int128_t average;
  if (overflow == 0) {
    divideWithRoundUp<__int128_t>(average, sum, count, false);
    return average;
  }
  unsigned __int128 quotMul;
  unsigned __int128 remMul = divideWithRoundUp<unsigned __int128>(
      quotMul, overflowMultiplier(), count, true);
  remMul *= overflow;
  quotMul *= overflow;
  __int128_t quotSum;
  auto const remSum = divideWithRoundUp<__int128_t>(quotSum, sum, count, true);
  __int128_t remTotal;
  divideWithRoundUp<__int128_t>(
      remTotal, static_cast<__int128_t>(remMul + remSum), count, false);
  return static_cast<__int128_t>(quotMul + quotSum + remTotal);
}

struct AverageDecimal128Functor {
  __int128_t const* sums;
  int64_t const* counts;
  int64_t const* overflows;
  __int128_t* out;

  __device__ void operator()(cudf::size_type idx) const {
    auto const count = counts[idx];
    out[idx] = count == 0 ? __int128_t{0}
                          : computeAverage(sums[idx], count, overflows[idx]);
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
  cudf::mutable_column_view outView;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename SumT>
    requires DecimalSumStorageType<SumT>
  void operator()() const {
    auto const n = static_cast<size_t>(numRows);
    launchDeviceFor(
        numRows,
        [&] {
          return AvgRoundFunctor<SumT>{
              cuda::std::span<const SumT>{sumCol.data<SumT>(), n},
              cuda::std::span<const int64_t>{counts, n},
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
            .invalidStateFlag = invalidState.data()};
      },
      stream);
  return invalidState.value(stream) == 0;
}

void averageRoundDecimalSum(
    cudf::type_id sumType,
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{sumType},
      averageRoundDecimalSumKernel{sumCol, counts, outView, numRows, stream});
}

void packDecimalSumState(
    cudf::type_id sumType,
    cudf::type_id offsetType,
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::column_view offsetsView,
    uint8_t* chars,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{sumType},
      cudf::data_type{offsetType},
      packDecimalSumStateKernel{
          sumCol, counts, offsetsView, chars, numRows, stream});
}

std::pair<cuda::device_buffer<std::byte>, cudf::size_type>
buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return buildStateValidityMaskImpl(sumCol, countCol, stream, mr);
}

void splitDecimal128SumLimbs(
    cudf::column_view values,
    cuda::std::span<const cudf::mutable_column_view> limbs,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(
      values.type().id() == cudf::type_id::DECIMAL128,
      "Decimal128 limb split requires DECIMAL128 input");
  CUDF_EXPECTS(
      limbs.size() == kDecimal128SumLimbs,
      "Decimal128 limb split requires four limb outputs");
  for (auto const& limb : limbs) {
    CUDF_EXPECTS(
        limb.type().id() == cudf::type_id::INT64 &&
            limb.size() == values.size(),
        "Decimal128 limb outputs must be INT64 columns sized to the input");
  }
  launchDeviceFor(
      values.size(),
      [&] {
        return SplitDecimal128LimbsFunctor{
            values.data<__int128_t>(),
            limbs[0].data<int64_t>(),
            limbs[1].data<int64_t>(),
            limbs[2].data<int64_t>(),
            limbs[3].data<int64_t>()};
      },
      stream);
}

void assembleDecimal128SumState(
    cuda::std::span<const cudf::column_view> limbSums,
    const cudf::column_view* overflowSum,
    cudf::mutable_column_view sum,
    cudf::mutable_column_view overflow,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(
      limbSums.size() == kDecimal128SumLimbs,
      "Decimal128 state assembly requires four limb sums");
  auto const numRows = sum.size();
  for (auto const& limb : limbSums) {
    CUDF_EXPECTS(
        limb.type().id() == cudf::type_id::INT64 && limb.size() == numRows,
        "Decimal128 limb sums must be INT64 columns sized to the output");
  }
  CUDF_EXPECTS(
      overflowSum == nullptr ||
          (overflowSum->type().id() == cudf::type_id::INT64 &&
           overflowSum->size() == numRows),
      "Decimal128 overflow sums must be INT64 sized to the output");
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128 &&
          overflow.type().id() == cudf::type_id::INT64 &&
          overflow.size() == numRows,
      "Decimal128 state assembly requires DECIMAL128 sum and INT64 overflow");
  launchDeviceFor(
      numRows,
      [&] {
        return AssembleDecimal128SumFunctor{
            .limb0 = limbSums[0].data<int64_t>(),
            .limb1 = limbSums[1].data<int64_t>(),
            .limb2 = limbSums[2].data<int64_t>(),
            .limb3 = limbSums[3].data<int64_t>(),
            .limbMask =
                limbSums[0].nullable() ? limbSums[0].null_mask() : nullptr,
            .limbOffset = limbSums[0].offset(),
            .overflowIn = overflowSum ? overflowSum->data<int64_t>() : nullptr,
            .overflowMask = overflowSum && overflowSum->nullable()
                ? overflowSum->null_mask()
                : nullptr,
            .overflowOffset = overflowSum ? overflowSum->offset() : 0,
            .sum = sum.data<__int128_t>(),
            .overflow = overflow.data<int64_t>()};
      },
      stream);
}

cudf::size_type finalizeDecimal128Sum(
    cudf::column_view sum,
    cudf::column_view overflow,
    cudf::mutable_column_view out,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128 &&
          overflow.type().id() == cudf::type_id::INT64 &&
          out.type().id() == cudf::type_id::DECIMAL128 &&
          overflow.size() == sum.size() && out.size() == sum.size(),
      "Decimal128 SUM finalization requires matching DECIMAL128/INT64 inputs");
  auto constexpr kNoInvalidRow = std::numeric_limits<cudf::size_type>::max();
  if (sum.size() == 0) {
    return -1;
  }
  auto sumDeviceView = cudf::column_device_view::create(sum, stream);
  auto overflowDeviceView = cudf::column_device_view::create(overflow, stream);
  rmm::device_scalar<cudf::size_type> firstInvalidRow{kNoInvalidRow, stream};
  launchDeviceFor(
      sum.size(),
      [&] {
        return FinalizeDecimal128SumFunctor{
            .sum = *sumDeviceView,
            .overflow = *overflowDeviceView,
            .out = out.data<__int128_t>(),
            .firstInvalidRow = firstInvalidRow.data()};
      },
      stream);
  auto const row = firstInvalidRow.value(stream);
  return row == kNoInvalidRow ? -1 : row;
}

void averageDecimal128WithOverflow(
    cudf::column_view sum,
    cudf::column_view counts,
    cudf::column_view overflow,
    cudf::mutable_column_view out,
    cuda::stream_ref stream) {
  auto const numRows = sum.size();
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128 &&
          counts.type().id() == cudf::type_id::INT64 &&
          overflow.type().id() == cudf::type_id::INT64 &&
          out.type().id() == cudf::type_id::DECIMAL128 &&
          counts.size() == numRows && overflow.size() == numRows &&
          out.size() == numRows,
      "Decimal128 AVG requires matching DECIMAL128 sum and INT64 count/overflow");
  launchDeviceFor(
      numRows,
      [&] {
        return AverageDecimal128Functor{
            sum.data<__int128_t>(),
            counts.data<int64_t>(),
            overflow.data<int64_t>(),
            out.data<__int128_t>()};
      },
      stream);
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
