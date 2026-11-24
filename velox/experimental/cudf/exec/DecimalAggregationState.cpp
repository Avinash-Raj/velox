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
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <limits>
#include <string_view>
#include <vector>

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
  auto makeStateColumns = [&](cudf::mask_state state) {
    DecimalSumStateColumns columns;
    columns.sum = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::DECIMAL128, -scale},
        numRows,
        state,
        stream,
        mr);
    columns.count = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64}, numRows, state, stream, mr);
    columns.overflow = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64}, numRows, state, stream, mr);
    return columns;
  };
  if (numRows == 0) {
    return makeStateColumns(cudf::mask_state::UNALLOCATED);
  }

  // For fully-null state columns there is nothing to deserialize. Avoid
  // launching unpack kernels over string payload buffers that may be empty.
  if (stateCol.nullable() && stateCol.null_count() == numRows) {
    return makeStateColumns(cudf::mask_state::ALL_NULL);
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

  auto result = makeStateColumns(cudf::mask_state::UNALLOCATED);

  VELOX_CHECK(
      detail::unpackDecimalSumState(
          stateCol,
          result.sum->mutable_view(),
          result.count->mutable_view(),
          result.overflow->mutable_view(),
          stream),
      "Decimal sum state requires every non-null row to be {} bytes and {}-byte aligned",
      detail::kDecimalSumStateSize,
      alignof(int64_t));

  if (stateCol.nullable()) {
    auto nullCount = stateCol.null_count();
    for (auto* column : {&result.sum, &result.count, &result.overflow}) {
      (*column)->set_null_mask(
          cudf::copy_bitmask(stateCol, stream, cudf::memory_resources{mr, mr}),
          nullCount);
    }
  }
  return result;
}

std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    const cudf::column_view* overflowCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      countCol.type().id() == cudf::type_id::INT64,
      "Decimal sum state requires INT64 count column (type is {})",
      cudf::type_to_name(countCol.type()));
  if (overflowCol != nullptr) {
    VELOX_CHECK(
        overflowCol->type().id() == cudf::type_id::INT64,
        "Decimal sum state requires INT64 overflow column (type is {})",
        cudf::type_to_name(overflowCol->type()));
    VELOX_CHECK_EQ(
        overflowCol->size(),
        sumCol.size(),
        "Decimal sum state requires sum and overflow to be same size");
  }
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
  VELOX_CHECK(
      overflowCol == nullptr || sumType == cudf::type_id::DECIMAL128,
      "Decimal sum state overflow requires a DECIMAL128 sum column");
  detail::packDecimalSumState(
      sumType,
      offsetsType,
      sumCol,
      countCol.data<int64_t>(),
      overflowCol == nullptr ? nullptr : overflowCol->data<int64_t>(),
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
    const cudf::column_view* overflowCol,
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
  if (overflowCol != nullptr) {
    VELOX_CHECK(
        overflowCol->type().id() == cudf::type_id::INT64 &&
            sumCol.type().id() == cudf::type_id::DECIMAL128,
        "Decimal average overflow requires DECIMAL128 sums and INT64 overflow (types are {}, {})",
        cudf::type_to_name(sumCol.type()),
        cudf::type_to_name(overflowCol->type()));
    VELOX_CHECK_EQ(
        overflowCol->size(),
        sumCol.size(),
        "Decimal average requires sum and overflow to be same size");
  }

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
        overflowCol == nullptr ? nullptr : overflowCol->data<int64_t>(),
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

DecimalSumStateColumns makeDecimalSumOverflowState(
    std::unique_ptr<cudf::column> sumOverflow,
    const cudf::column_view* stateOverflowSum,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      sumOverflow->type().id() == cudf::type_id::STRUCT &&
          sumOverflow->num_children() == 2,
      "Decimal SUM_OVERFLOW result must be a STRUCT of sum and overflow (type is {})",
      cudf::type_to_name(sumOverflow->type()));
  const auto numRows = sumOverflow->size();
  // SUM_OVERFLOW marks a group without non-null input null in the parent only,
  // so its sum child inherits the parent's nulls.
  auto [sumMask, sumNullCount] = [&] {
    const auto view = sumOverflow->view();
    return cudf::bitmask_and(
        cudf::table_view{{view, view.child(0)}},
        stream,
        cudf::memory_resources{mr, get_temp_mr()});
  }();
  auto contents = sumOverflow->release();
  DecimalSumStateColumns state;
  state.sum = std::move(contents.children[0]);
  VELOX_CHECK(
      state.sum->type().id() == cudf::type_id::DECIMAL128,
      "Decimal SUM_OVERFLOW requires a DECIMAL128 sum (type is {})",
      cudf::type_to_name(state.sum->type()));
  state.sum->set_null_mask(std::move(sumMask), sumNullCount);
  state.overflow = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  detail::assembleDecimalSumOverflow(
      state.sum->mutable_view(),
      contents.children[1]->view(),
      stateOverflowSum,
      state.overflow->mutable_view(),
      stream);
  return state;
}

std::unique_ptr<cudf::column> widenDecimal64SumOverflow(
    std::unique_ptr<cudf::column> sumOverflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      sumOverflow->type().id() == cudf::type_id::STRUCT &&
          sumOverflow->num_children() == 2,
      "Decimal64 SUM_OVERFLOW result must be a STRUCT of sum and overflow (type is {})",
      cudf::type_to_name(sumOverflow->type()));
  const auto view = sumOverflow->view();
  const auto sum = view.child(0);
  VELOX_CHECK(
      sum.type().id() == cudf::type_id::DECIMAL64,
      "Decimal64 SUM_OVERFLOW requires a DECIMAL64 sum (type is {})",
      cudf::type_to_name(sum.type()));
  // SUM_OVERFLOW marks a group without non-null input null in the parent only.
  auto [mask, nullCount] = cudf::bitmask_and(
      cudf::table_view{{view, sum}},
      stream,
      cudf::memory_resources{mr, get_temp_mr()});
  auto out = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::DECIMAL128, sum.type().scale()},
      view.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  detail::widenDecimal64SumOverflow(
      sum, view.child(1), out->mutable_view(), stream);
  if (nullCount > 0) {
    out->set_null_mask(std::move(mask), nullCount);
  }
  return out;
}

std::unique_ptr<cudf::column> adjustDecimalSumForOverflow(
    const cudf::column_view& sumCol,
    const cudf::column_view& overflowCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto out = cudf::make_fixed_width_column(
      sumCol.type(), sumCol.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  const auto invalidRow = detail::adjustDecimalSumForOverflow(
      sumCol, overflowCol, out->mutable_view(), stream);
  if (invalidRow >= 0) {
    const auto sumScalar =
        cudf::get_element(sumCol, invalidRow, stream, get_temp_mr());
    const auto overflowScalar =
        cudf::get_element(overflowCol, invalidRow, stream, get_temp_mr());
    const auto sum =
        static_cast<const cudf::fixed_point_scalar<numeric::decimal128>&>(
            *sumScalar)
            .value(stream);
    const auto overflow =
        static_cast<const cudf::numeric_scalar<int64_t>&>(*overflowScalar)
            .value(stream);
    // Raise the CPU DecimalSumAggregate error for the offending row.
    const auto adjusted = DecimalUtil::adjustSumForOverflow(sum, overflow);
    VELOX_USER_CHECK(adjusted.has_value(), "Decimal overflow");
    DecimalUtil::valueInRange(adjusted.value());
    VELOX_UNREACHABLE();
  }
  if (sumCol.nullable()) {
    out->set_null_mask(
        cudf::copy_bitmask(sumCol, stream, cudf::memory_resources{mr, mr}),
        sumCol.null_count());
  }
  return out;
}

namespace {

void checkInt64(const cudf::column_view& column, std::string_view name) {
  VELOX_CHECK(
      column.type().id() == cudf::type_id::INT64,
      "Decimal128 aggregation state requires an INT64 {} child (type is {})",
      name,
      cudf::type_to_name(column.type()));
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

std::unique_ptr<cudf::column> makeDecimal128StateColumn(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    std::unique_ptr<cudf::column> overflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(sum->type().id() == cudf::type_id::DECIMAL128);
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
