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
#include "velox/experimental/cudf/exec/CudfAggregationNode.h"

namespace facebook::velox::cudf_velox {

TypePtr CudfAggregationNode::compactIntermediateType(
    const Aggregate& aggregate) {
  if (aggregate.call->name() !=
          CudfConfig::getInstance().functionNamePrefix + "sum" ||
      aggregate.rawInputTypes.size() != 1 ||
      !aggregate.rawInputTypes[0]->isShortDecimal()) {
    return nullptr;
  }
  VELOX_USER_CHECK(
      !aggregate.distinct && aggregate.sortingKeys.empty() &&
          aggregate.sortingOrders.empty() && !aggregate.mask,
      "Compact Decimal64 SUM does not support DISTINCT, ordering or masks");
  VELOX_USER_CHECK_EQ(aggregate.call->inputs().size(), 1);
  const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(
      aggregate.call->inputs()[0].get());
  VELOX_USER_CHECK(
      field && field->isInputColumn(),
      "Compact Decimal64 SUM requires a column input");
  return DECIMAL(
      38, getDecimalPrecisionScale(*aggregate.rawInputTypes[0]).second);
}

CudfAggregationNode::CudfAggregationNode(
    const core::AggregationNode& node,
    std::vector<TypePtr> compactIntermediateTypes)
    : core::AggregationNode(
          node.id(),
          node.step(),
          node.groupingKeys(),
          node.preGroupedKeys(),
          node.aggregateNames(),
          node.aggregates(),
          node.globalGroupingSets(),
          node.groupId(),
          node.ignoreNullKeys(),
          node.noGroupsSpanBatches(),
          node.mayRetainInput(),
          node.sources()[0]),
      compactIntermediateTypes_(std::move(compactIntermediateTypes)) {
  VELOX_USER_CHECK_NE(step(), Step::kSingle);
  VELOX_USER_CHECK_EQ(compactIntermediateTypes_.size(), aggregates().size());
  bool hasCompactState = false;
  for (size_t i = 0; i < aggregates().size(); ++i) {
    const auto& stateType = intermediateType(i);
    if (!stateType) {
      continue;
    }
    hasCompactState = true;
    const auto& aggregate = aggregates()[i];
    auto expected = compactIntermediateType(aggregate);
    VELOX_USER_CHECK(expected && *stateType == *expected);
    const auto& input = aggregate.call->inputs()[0];
    const auto& inputType =
        step() == Step::kPartial ? aggregate.rawInputTypes[0] : stateType;
    VELOX_USER_CHECK(*input->type() == *inputType);
    VELOX_USER_CHECK(*aggregate.call->type() == *stateType);
    const auto& field = static_cast<const core::FieldAccessTypedExpr&>(*input);
    VELOX_USER_CHECK(
        *sources()[0]->outputType()->findChild(field.name()) == *inputType);
  }
  VELOX_USER_CHECK(hasCompactState);
}

folly::dynamic CudfAggregationNode::serialize() const {
  auto obj = core::AggregationNode::serialize();
  obj["compactIntermediateTypes"] = folly::dynamic::array;
  for (const auto& type : compactIntermediateTypes_) {
    obj["compactIntermediateTypes"].push_back(
        type ? type->serialize() : folly::dynamic(nullptr));
  }
  return obj;
}

core::PlanNodePtr CudfAggregationNode::create(
    const folly::dynamic& obj,
    void* context) {
  auto node = core::AggregationNode::create(obj, context);
  std::vector<TypePtr> types;
  for (const auto& type : obj["compactIntermediateTypes"]) {
    types.push_back(
        type.isNull() ? nullptr : ISerializable::deserialize<Type>(type));
  }
  return std::make_shared<CudfAggregationNode>(
      static_cast<const core::AggregationNode&>(*node), std::move(types));
}

void CudfAggregationNode::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("CudfAggregationNode", CudfAggregationNode::create);
}

} // namespace facebook::velox::cudf_velox
