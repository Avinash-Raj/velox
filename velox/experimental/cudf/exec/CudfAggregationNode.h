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

#include "velox/core/PlanNode.h"

namespace facebook::velox::cudf_velox {

/// An aggregation with an explicit GPU-only intermediate-state contract.
/// Inherits the core node's shape, but must not execute as a CPU aggregation.
class CudfAggregationNode final : public core::AggregationNode {
 public:
  /// 'node' already contains the rewritten calls and source schema. Each entry
  /// in compactIntermediateTypes corresponds to an aggregate; nullptr retains
  /// that aggregate's ordinary intermediate representation.
  CudfAggregationNode(
      const core::AggregationNode& node,
      std::vector<TypePtr> compactIntermediateTypes);

  bool usesCompactDecimalSum(size_t index) const {
    return compactIntermediateTypes_.at(index) != nullptr;
  }

  /// nullptr for an aggregate that retains its ordinary intermediate type.
  const TypePtr& intermediateType(size_t index) const {
    return compactIntermediateTypes_.at(index);
  }

  /// Returns DECIMAL(38, raw input scale) for plain Decimal64 SUM, otherwise
  /// nullptr. Modifiers on Decimal64 SUM are rejected rather than allowing
  /// producer and consumer fragments to choose different state protocols.
  static TypePtr compactIntermediateType(const Aggregate& aggregate);

  std::string_view name() const override {
    return "CudfAggregation";
  }

  bool canSpill(const core::QueryConfig&) const override {
    return false;
  }

  folly::dynamic serialize() const override;
  static core::PlanNodePtr create(const folly::dynamic& obj, void* context);
  static void registerSerDe();

 private:
  const std::vector<TypePtr> compactIntermediateTypes_;
};

} // namespace facebook::velox::cudf_velox
