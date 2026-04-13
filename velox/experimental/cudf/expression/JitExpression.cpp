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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/AstPrinter.h"
#include "velox/experimental/cudf/expression/JitExpression.h"

namespace facebook::velox::cudf_velox {

JitExpression::JitExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema)
    : expr_{expr, inputRowSchema} {}

void JitExpression::close() {
  expr_.close();
}

void log_int64_t_column_view_to_stdout(const cudf::column_view& view) {
  std::cout << "  Type: " << cudf::ast::type_id_to_string(view.type().id()) << ", scale: " << view.type().scale() << std::endl;
  if (view.type().id() != cudf::type_id::DECIMAL64) {
    std::cout << "  Not an DECIMAL64 column view, cannot log values" << std::endl;
    return;
  }
  auto n = view.size();
  std::vector<int64_t> host(n);
  auto const* d = static_cast<int64_t const*>(view.head<int64_t>());
  cudaMemcpy(host.data(), d, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
  std::cout << "  Raw values:" << std::endl;
  for (cudf::size_type i = 0; i < n; ++i) {
    std::cout << "    [" << i << "] = " << host[i] << std::endl;
  }
}

ColumnOrView JitExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  auto precomputedColumns = precomputeSubexpressions(
      inputColumnViews,
      expr_.precomputeInstructions_,
      expr_.scalars_,
      expr_.inputRowSchema_,
      stream);

  // Make table_view from input columns and precomputed columns
  std::vector<cudf::column_view> allColumnViews(inputColumnViews);
  allColumnViews.reserve(inputColumnViews.size() + precomputedColumns.size());
  for (auto& precomputedCol : precomputedColumns) {
    allColumnViews.push_back(asView(precomputedCol));
  }

  // When the result requires DECIMAL128, upcast any DECIMAL64 input columns
  // before evaluation to prevent intermediate overflow (e.g. MUL of two
  // short decimals whose product exceeds 64-bit range).  We can only do
  // this when the AST tree contains no DECIMAL64 scalar literals, because
  // those are embedded in the tree and cannot be upcast — mixing DECIMAL128
  // columns with DECIMAL64 scalars would cause a JIT compilation error.
  std::vector<std::unique_ptr<cudf::column>> upcastColumns;
  if (finalize) {
    const auto requestedType =
        cudf_velox::veloxToCudfDataType(expr_.expr_->type());
    if (requestedType.id() == cudf::type_id::DECIMAL128) {
      bool hasDecimal64Scalars = std::any_of(
          expr_.scalars_.begin(),
          expr_.scalars_.end(),
          [](const auto& s) {
            return s && s->type().id() == cudf::type_id::DECIMAL64;
          });
      if (!hasDecimal64Scalars) {
        for (auto& view : allColumnViews) {
          if (view.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, view.type().scale()};
            upcastColumns.push_back(cudf::cast(view, castType, stream, mr));
            view = upcastColumns.back()->view();
          }
        }
      }
    }
  }

  cudf::table_view astInputTableView(allColumnViews);

  auto result = [&]() -> ColumnOrView {
    if (auto colRefPtr = dynamic_cast<cudf::ast::column_reference const*>(
            &expr_.cudfTree_.back())) {
      auto columnIndex = colRefPtr->get_column_index();
      if (columnIndex < inputColumnViews.size()) {
        return inputColumnViews[columnIndex];
      } else {
        return std::move(
            precomputedColumns[columnIndex - inputColumnViews.size()]);
      }
    } else {
      if (CudfConfig::getInstance().debugEnabled) {
        LOG(INFO) << cudf::ast::expression_to_string(
            expr_.cudfTree_.back(), astInputTableView);
      }
      return cudf::compute_column_jit(
          astInputTableView, expr_.cudfTree_.back(), stream, mr);
    }
  }();
  if (finalize) {
    const auto requestedType =
        cudf_velox::veloxToCudfDataType(expr_.expr_->type());
    std::cout << "Initial result:" << std::endl;
    log_int64_t_column_view_to_stdout(asView(result));
    std::cout << "Finalizing, requested type is " << cudf::ast::type_id_to_string(requestedType.id()) << ", scale " << requestedType.scale() << std::endl;
    auto resultView = asView(result);
    if (resultView.type() != requestedType) {
      result = cudf::cast(resultView, requestedType, stream, mr);
    }
  } else {
    std::cout << "Not finalizing" << std::endl;
  }
  std::cout << "Final result:" << std::endl;
  log_int64_t_column_view_to_stdout(asView(result));
  return result;
}

bool JitExpression::canEvaluate(std::shared_ptr<velox::exec::Expr> expr) {
  return ASTExpression::canEvaluate(expr);
}

void registerJitEvaluator(int priority) {
  registerCudfExpressionEvaluator(
      kJitEvaluatorName,
      priority,
      [](std::shared_ptr<velox::exec::Expr> expr) {
        return JitExpression::canEvaluate(expr);
      },
      [](std::shared_ptr<velox::exec::Expr> expr, const RowTypePtr& row) {
        return std::make_shared<JitExpression>(std::move(expr), row);
      },
      /*overwrite=*/false);
}

} // namespace facebook::velox::cudf_velox
