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

class CudfPlanRewriter {
 public:
  /// Opt-in GPU-only fragment rewrite, before task/driver creation. Every
  /// producer and consumer of a compact UCX edge must use this rewrite and
  /// compatible aggregate rawInputTypes. This cannot negotiate with a remote
  /// worker or prove that its fragment has been rewritten.
  ///
  /// Supports Decimal64 SUM stages connected directly, through LocalPartition,
  /// or through UCX PartitionedOutput/Exchange. Other node classes are
  /// retained; unsupported bridges and non-UCX compact boundaries fail closed.
  /// SINGLE and unrelated aggregate slots retain their existing representation.
  /// The caller must prohibit CPU fallback for CudfAggregationNode.
  static core::PlanNodePtr rewrite(const core::PlanNodePtr& root);
};

} // namespace facebook::velox::cudf_velox
