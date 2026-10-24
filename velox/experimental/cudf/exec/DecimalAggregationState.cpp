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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/DecimalAggregationDevice.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"
#include "velox/type/DecimalUtil.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <limits>
#include <string_view>

namespace facebook::velox::cudf_velox {

DecimalSumStateColumns reduceDecimal64SumCount(
    const cudf::column_view& input,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  DecimalSumStateColumns result;
  result.sum = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::DECIMAL128, input.type().scale()},
      1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  result.count = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64},
      1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  detail::reduceDecimal64SumCount(
      input,
      result.sum->mutable_view(),
      result.count->mutable_view(),
      stream,
      mr);
  auto [nullMask, nullCount] = detail::buildStateValidityMask(
      result.sum->view(), result.count->view(), stream, mr);
  result.sum->set_null_mask(std::move(nullMask), nullCount);
  return result;
}

DecimalSumStateColumns deserializeDecimalSumState(
    const cudf::column_view& stateCol,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      stateCol.type().id() == cudf::type_id::STRING,
      "Decimal sum state requires STRING/VARBINARY column (type is {})",
      cudf::type_to_name(stateCol.type()));
  auto numRows = stateCol.size();
  if (numRows == 0) {
    DecimalSumStateColumns empty;
    empty.sum = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::DECIMAL128, -scale},
        0,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    empty.count = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64},
        0,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    return empty;
  }

  // For fully-null state columns there is nothing to deserialize. Avoid
  // launching unpack kernels over string payload buffers that may be empty.
  if (stateCol.nullable() && stateCol.null_count() == numRows) {
    DecimalSumStateColumns allNull;
    allNull.sum = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::DECIMAL128, -scale},
        numRows,
        cudf::mask_state::ALL_NULL,
        stream,
        mr);
    allNull.count = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64},
        numRows,
        cudf::mask_state::ALL_NULL,
        stream,
        mr);
    return allNull;
  }

  cudf::strings_column_view strings(stateCol);

  auto const nullCount = stateCol.nullable() ? stateCol.null_count() : 0;
  auto const [payloadBegin, payloadEnd] =
      cudf::strings::detail::get_first_and_last_offset(strings, stream);
  auto const payloadSize = payloadEnd - payloadBegin;
  // A null row's payload width is path dependent. serializeDecimalSumState
  // writes kDecimalSumStateSize bytes for every row including nulls, while a
  // velox/Arrow round trip compacts null rows to 0 bytes. Both encodings can
  // occur in the same column: the streaming final aggregation concatenates its
  // buffered (serialized) result with each newly arrived (round tripped) batch,
  // so the payload size lands anywhere between the two extremes. Non-null rows
  // are always kDecimalSumStateSize wide, and that is all
  // unpackDecimalSumState requires -- it skips null rows and addresses payloads
  // through the per-row offsets rather than a fixed stride.
  auto const fullPayloadSize =
      static_cast<int64_t>(numRows) * detail::kDecimalSumStateSize;
  auto const compactPayloadSize =
      static_cast<int64_t>(numRows - nullCount) * detail::kDecimalSumStateSize;
  VELOX_CHECK(
      payloadSize >= compactPayloadSize && payloadSize <= fullPayloadSize &&
          payloadSize % detail::kDecimalSumStateSize == 0,
      "Decimal sum state has an invalid payload size: expected a multiple of {} in [{}, {}], got {}",
      detail::kDecimalSumStateSize,
      compactPayloadSize,
      fullPayloadSize,
      payloadSize);

  auto sumCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::DECIMAL128, -scale},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto countCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);

  auto sumView = sumCol->mutable_view();
  auto countView = countCol->mutable_view();

  VELOX_CHECK(
      detail::unpackDecimalSumState(stateCol, sumView, countView, stream),
      "Decimal sum state requires every non-null row to be {} bytes and {}-byte aligned",
      detail::kDecimalSumStateSize,
      alignof(int64_t));

  if (stateCol.nullable()) {
    auto nullMask =
        cudf::copy_bitmask(stateCol, stream, cudf::memory_resources{mr, mr});
    auto nullCount = stateCol.null_count();
    sumCol->set_null_mask(std::move(nullMask), nullCount);
    auto countMask =
        cudf::copy_bitmask(stateCol, stream, cudf::memory_resources{mr, mr});
    countCol->set_null_mask(std::move(countMask), nullCount);
  }

  DecimalSumStateColumns result;
  result.sum = std::move(sumCol);
  result.count = std::move(countCol);
  return result;
}

std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      countCol.type().id() == cudf::type_id::INT64,
      "Decimal sum state requires INT64 count column (type is {})",
      cudf::type_to_name(countCol.type()));
  auto numRows = sumCol.size();
  VELOX_CHECK_EQ(
      numRows,
      countCol.size(),
      "Decimal sum state requires sum and count to be same size (sum size is {}, count size is {})",
      sumCol.size(),
      countCol.size());
  VELOX_CHECK_LE(
      numRows,
      static_cast<cudf::size_type>(std::numeric_limits<int32_t>::max()),
      "Too many rows to serialize decimal sum state (row count is {})",
      numRows);

  if (numRows == 0) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  auto const rowCount = static_cast<int32_t>(numRows);

  auto const charsBytes =
      static_cast<int64_t>(numRows) * detail::kDecimalSumStateSize;
  auto const threshold = cudf::strings::get_offset64_threshold();
  auto const useLargeOffsets = charsBytes >= threshold;
  // Previously this guard threw std::overflow_error; Velox uses
  // VeloxRuntimeError for this guard.
  VELOX_CHECK(
      !useLargeOffsets || cudf::strings::is_large_strings_enabled(),
      "Size of output ({}) exceeds the column size limit ({})",
      charsBytes,
      threshold);

  auto const offsetsType =
      useLargeOffsets ? cudf::type_id::INT64 : cudf::type_id::INT32;
  auto offsetsCol = cudf::make_fixed_width_column(
      cudf::data_type{offsetsType},
      numRows + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto offsetsView = offsetsCol->mutable_view();

  rmm::device_buffer charsBuf(
      static_cast<size_t>(numRows) * detail::kDecimalSumStateSize, stream, mr);

  auto charsPtr = reinterpret_cast<uint8_t*>(charsBuf.data());
  detail::fillOffsetsForDecimalSumState(
      offsetsType, offsetsView, rowCount, stream);

  const auto sumType = sumCol.type().id();
  VELOX_CHECK(
      sumType == cudf::type_id::DECIMAL64 ||
          sumType == cudf::type_id::DECIMAL128,
      "Unsupported decimal sum column type (type is {})",
      cudf::type_to_name(sumCol.type()));
  detail::packDecimalSumState(
      sumType,
      offsetsType,
      sumCol,
      countCol.data<int64_t>(),
      offsetsView,
      charsPtr,
      rowCount,
      stream);

  auto [nullMask, nullCount] =
      detail::buildStateValidityMask(sumCol, countCol, stream, mr);
  return cudf::make_strings_column(
      static_cast<cudf::size_type>(numRows),
      std::move(offsetsCol),
      std::move(charsBuf),
      nullCount,
      std::move(nullMask));
}

std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      countCol.type().id() == cudf::type_id::INT64,
      "Decimal average requires INT64 count column (type is {})",
      cudf::type_to_name(countCol.type()));
  VELOX_CHECK(
      sumCol.type().id() == cudf::type_id::DECIMAL64 ||
          sumCol.type().id() == cudf::type_id::DECIMAL128,
      "Decimal average requires DECIMAL64 or DECIMAL128 sum column (type is {})",
      cudf::type_to_name(sumCol.type()));
  VELOX_CHECK_EQ(
      sumCol.size(),
      countCol.size(),
      "Decimal average requires sum and count to be same size (sum size is {}, count size is {})",
      sumCol.size(),
      countCol.size());

  auto numRows = sumCol.size();
  auto out = cudf::make_fixed_width_column(
      sumCol.type(), numRows, cudf::mask_state::UNALLOCATED, stream, mr);

  if (numRows > 0) {
    auto const rowCount = static_cast<int32_t>(numRows);
    const auto sumType = sumCol.type().id();
    detail::averageRoundDecimalSum(
        sumType,
        sumCol,
        countCol.data<int64_t>(),
        out->mutable_view(),
        rowCount,
        stream);
  }

  auto [nullMask, nullCount] =
      detail::buildStateValidityMask(sumCol, countCol, stream, mr);
  if (nullCount > 0) {
    out->set_null_mask(std::move(nullMask), nullCount);
  }
  return out;
}

namespace {

static_assert(kDecimal128SumLimbs == detail::kDecimal128SumLimbs);

std::unique_ptr<cudf::column> makeInt64Column(
    cudf::size_type size,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64},
      size,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
}

void copyNullMask(
    const cudf::column_view& source,
    cudf::column& target,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (source.nullable() && source.null_count() > 0) {
    target.set_null_mask(
        cudf::copy_bitmask(source, stream, cudf::memory_resources{mr, mr}),
        source.null_count());
  }
}

void checkInt64(const cudf::column_view& column, std::string_view name) {
  VELOX_CHECK(
      column.type().id() == cudf::type_id::INT64,
      "Decimal128 aggregation state requires an INT64 {} child (type is {})",
      name,
      cudf::type_to_name(column.type()));
}

// Rethrows the CPU DecimalSumAggregate::computeFinalValue error for a row the
// device check rejected, so both engines report identical messages.
[[noreturn]] void throwDecimal128SumOverflow(
    const cudf::column_view& sum,
    const cudf::column_view& overflow,
    cudf::size_type row,
    cuda::stream_ref stream) {
  int128_t value;
  int64_t carry;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &value,
      sum.data<__int128_t>() + row,
      sizeof(value),
      cudaMemcpyDeviceToHost,
      stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &carry,
      overflow.data<int64_t>() + row,
      sizeof(carry),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  auto const adjusted = DecimalUtil::adjustSumForOverflow(value, carry);
  VELOX_USER_CHECK(adjusted.has_value(), "Decimal overflow");
  DecimalUtil::valueInRange(adjusted.value());
  VELOX_FAIL(
      "Device rejected Decimal128 SUM state ({}, {}) that the host accepts",
      value,
      carry);
}

} // namespace

Decimal128StateChildren decimal128StateChildren(
    const cudf::column_view& state,
    bool average,
    cuda::stream_ref stream) {
  auto const numChildren = decimal128StateOverflowChild(average) + 1;
  VELOX_CHECK(
      state.type().id() == cudf::type_id::STRUCT &&
          state.num_children() == numChildren,
      "Decimal128 {} state requires a STRUCT with {} children (type is {})",
      average ? "AVG" : "SUM",
      numChildren,
      cudf::type_to_name(state.type()));
  VELOX_CHECK(
      !state.has_nulls(), "Decimal128 aggregation state rows must be valid");
  const cudf::structs_column_view structs(state);
  Decimal128StateChildren children{
      .sum = structs.get_sliced_child(kDecimal128StateSumChild, stream),
      .count = average ? std::make_optional(structs.get_sliced_child(
                             kDecimal128StateCountChild, stream))
                       : std::nullopt,
      .overflow = structs.get_sliced_child(
          decimal128StateOverflowChild(average), stream)};
  VELOX_CHECK(
      children.sum.type().id() == cudf::type_id::DECIMAL128,
      "Decimal128 aggregation state requires a DECIMAL128 sum child (type is {})",
      cudf::type_to_name(children.sum.type()));
  checkInt64(children.overflow, "overflow");
  if (children.count) {
    checkInt64(*children.count, "count");
  }
  return children;
}

std::vector<std::unique_ptr<cudf::column>> splitDecimal128SumLimbs(
    const cudf::column_view& values,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      values.type().id() == cudf::type_id::DECIMAL128,
      "Decimal128 aggregation requires DECIMAL128 input (type is {})",
      cudf::type_to_name(values.type()));
  std::vector<std::unique_ptr<cudf::column>> limbs;
  std::vector<cudf::mutable_column_view> limbViews;
  for (size_t i = 0; i < kDecimal128SumLimbs; ++i) {
    limbs.push_back(makeInt64Column(values.size(), stream, mr));
    limbViews.push_back(limbs.back()->mutable_view());
  }
  detail::splitDecimal128SumLimbs(
      values,
      cuda::std::span<const cudf::mutable_column_view>{
          limbViews.data(), limbViews.size()},
      stream);
  for (auto& limb : limbs) {
    copyNullMask(values, *limb, stream, mr);
  }
  return limbs;
}

Decimal128SumOverflowColumns assembleDecimal128SumState(
    cuda::std::span<const cudf::column_view> limbSums,
    const cudf::column_view* overflowSum,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(limbSums.size(), kDecimal128SumLimbs);
  auto const numRows = limbSums[0].size();
  Decimal128SumOverflowColumns result;
  result.sum = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::DECIMAL128, -scale},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  result.overflow = makeInt64Column(numRows, stream, mr);
  detail::assembleDecimal128SumState(
      limbSums,
      overflowSum,
      result.sum->mutable_view(),
      result.overflow->mutable_view(),
      stream);
  copyNullMask(limbSums[0], *result.sum, stream, mr);
  return result;
}

std::unique_ptr<cudf::column> finalizeDecimal128Sum(
    const cudf::column_view& sum,
    const cudf::column_view& overflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(sum.size(), overflow.size());
  auto out = cudf::make_fixed_width_column(
      sum.type(), sum.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  auto const invalidRow =
      detail::finalizeDecimal128Sum(sum, overflow, out->mutable_view(), stream);
  if (invalidRow >= 0) {
    throwDecimal128SumOverflow(sum, overflow, invalidRow, stream);
  }
  copyNullMask(sum, *out, stream, mr);
  return out;
}

std::unique_ptr<cudf::column> finalizeDecimal128Average(
    const cudf::column_view& sum,
    const cudf::column_view& count,
    const cudf::column_view& overflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  checkInt64(count, "count");
  checkInt64(overflow, "overflow");
  auto out = cudf::make_fixed_width_column(
      sum.type(), sum.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  detail::averageDecimal128WithOverflow(
      sum, count, overflow, out->mutable_view(), stream);
  auto [nullMask, nullCount] =
      detail::buildStateValidityMask(sum, count, stream, mr);
  if (nullCount > 0) {
    out->set_null_mask(std::move(nullMask), nullCount);
  }
  return out;
}

std::unique_ptr<cudf::column> makeDecimal128StateColumn(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    std::unique_ptr<cudf::column> overflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto const size = sum->size();
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(sum));
  if (count) {
    checkInt64(count->view(), "count");
    children.push_back(std::move(count));
  }
  checkInt64(overflow->view(), "overflow");
  children.push_back(std::move(overflow));
  return cudf::make_structs_column(
      size,
      std::move(children),
      0,
      cudf::create_null_mask(
          0,
          cudf::mask_state::UNALLOCATED,
          stream,
          cudf::memory_resources{mr, get_temp_mr()}),
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox
