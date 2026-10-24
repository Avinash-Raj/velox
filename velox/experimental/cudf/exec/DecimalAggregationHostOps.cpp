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
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox {

void validateIntermediateColumnType(cudf::column_view const& column) {
  // fmt does not understand cudf::type_id enum class
  auto const colType = static_cast<int>(column.type().id());
  VELOX_CHECK_EQ(
      colType,
      static_cast<int>(cudf::type_id::STRING),
      "Expected serialized decimal aggregation state: Velox VARBINARY represented as cuDF STRING (got type {})",
      colType);
}

cudf::column_view castDecimal64InputToDecimal128(
    cudf::column_view inputCol,
    std::unique_ptr<cudf::column>& holder,
    cuda::stream_ref stream) {
  if (inputCol.type().id() != cudf::type_id::DECIMAL64) {
    return inputCol;
  }
  holder = cudf::cast(
      inputCol,
      cudf::data_type{cudf::type_id::DECIMAL128, inputCol.type().scale()},
      stream,
      get_temp_mr());
  return holder->view();
}

std::unique_ptr<cudf::column> castCountColumnToInt64(
    std::unique_ptr<cudf::column> count,
    cuda::stream_ref stream) {
  if (count->type().id() != cudf::type_id::INT64) {
    count = cudf::cast(
        *count, cudf::data_type{cudf::type_id::INT64}, stream, get_temp_mr());
  }
  return count;
}

std::unique_ptr<cudf::column> serializeDecimalPartialOrIntermediateState(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  count = castCountColumnToInt64(std::move(count), stream);
  return serializeDecimalSumState(sum->view(), count->view(), stream, mr);
}

std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  count = castCountColumnToInt64(std::move(count), stream);
  auto avgCol = computeDecimalAverage(sum->view(), count->view(), stream, mr);
  auto const cudfOutType = veloxToCudfDataType(resultType);
  if (avgCol->type() != cudfOutType) {
    avgCol = cudf::cast(avgCol->view(), cudfOutType, stream, mr);
  }
  return avgCol;
}

namespace {

std::unique_ptr<cudf::column> castToType(
    std::unique_ptr<cudf::column> column,
    const TypePtr& type,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto const cudfType = veloxToCudfDataType(type);
  if (column->type() != cudfType) {
    column = cudf::cast(column->view(), cudfType, stream, mr);
  }
  return column;
}

} // namespace

std::unique_ptr<cudf::column> makeDecimal128AggregateOutput(
    cuda::std::span<const cudf::column_view> limbSums,
    const cudf::column_view* overflowSum,
    std::unique_ptr<cudf::column> count,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const bool average = count != nullptr;
  if (average) {
    count = castToType(std::move(count), BIGINT(), stream, mr);
    VELOX_CHECK(
        !count->has_nulls(), "Decimal128 AVG state counts must not be null");
  }
  if (!resultType->isRow()) {
    auto state = assembleDecimal128SumState(
        limbSums,
        overflowSum,
        getDecimalPrecisionScale(*resultType).second,
        stream,
        get_temp_mr());
    auto result = average
        ? finalizeDecimal128Average(
              state.sum->view(),
              count->view(),
              state.overflow->view(),
              stream,
              mr)
        : finalizeDecimal128Sum(
              state.sum->view(), state.overflow->view(), stream, mr);
    return castToType(std::move(result), resultType, stream, mr);
  }
  VELOX_CHECK_EQ(resultType->size(), decimal128StateOverflowChild(average) + 1);
  auto state = assembleDecimal128SumState(
      limbSums,
      overflowSum,
      getDecimalPrecisionScale(*resultType->childAt(kDecimal128StateSumChild))
          .second,
      stream,
      mr);
  return makeDecimal128StateColumn(
      std::move(state.sum),
      std::move(count),
      std::move(state.overflow),
      stream,
      mr);
}

std::unique_ptr<cudf::column> finalizeDecimal128State(
    const cudf::column_view& state,
    bool average,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto const children = decimal128StateChildren(state, average, stream);
  auto result = average
      ? finalizeDecimal128Average(
            children.sum, *children.count, children.overflow, stream, mr)
      : finalizeDecimal128Sum(children.sum, children.overflow, stream, mr);
  return castToType(std::move(result), resultType, stream, mr);
}

} // namespace facebook::velox::cudf_velox
