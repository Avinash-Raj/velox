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

// Times decimal-arithmetic projections through CudfFilterProject over many
// batches. Not a correctness test; build it on each tree being compared.
//
// Env: BENCH_ROWS (rows per batch, default 1048576), BENCH_BATCHES (default
// 200), BENCH_ITERS (timed runs, default 5). opChain has its own:
// BENCH_CHAIN_OPS (comma-separated op counts, default 1,2,4,8,16,32,64),
// BENCH_CHAIN_ROWS (default 65536), BENCH_CHAIN_BATCHES (default 1000).

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cuda_runtime_api.h>

#include <folly/ScopeGuard.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace facebook::velox::cudf_velox {
namespace {

int64_t envOr(const char* name, int64_t fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? fallback : std::atoll(value);
}

class CudfDecimalSyncBenchmark : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
    rows_ = envOr("BENCH_ROWS", 1 << 20);
    batches_ = envOr("BENCH_BATCHES", 200);
    iters_ = envOr("BENCH_ITERS", 5);
    auto& config = CudfConfig::getInstance();
    config.allowCpuFallback = false;
    config.batchSizeMinThreshold = static_cast<int32_t>(rows_);
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    exec::test::OperatorTestBase::TearDown();
  }

  /// Checks one batch against the CPU, then times \p batches batches, or
  /// batches_ if 0.
  void run(
      const std::string& name,
      const RowVectorPtr& input,
      const std::vector<std::string>& projections,
      const std::vector<std::string>& aggregates,
      int64_t batches = 0) {
    if (batches == 0) {
      batches = batches_;
    }
    auto makePlan = [&](int32_t repeat) {
      return exec::test::PlanBuilder()
          .values({input}, false, repeat)
          .project(projections)
          .singleAggregation({}, aggregates)
          .planNode();
    };

    unregisterCudf();
    auto cpu = exec::test::AssertQueryBuilder(makePlan(1)).copyResults(pool());
    registerCudf();
    auto gpu = exec::test::AssertQueryBuilder(makePlan(1)).copyResults(pool());
    facebook::velox::test::assertEqualVectors(cpu, gpu);

    auto plan = makePlan(static_cast<int32_t>(batches));
    exec::test::AssertQueryBuilder(plan).copyResults(pool());

    std::vector<double> millis;
    for (int64_t i = 0; i < iters_; ++i) {
      const auto start = std::chrono::steady_clock::now();
      exec::test::AssertQueryBuilder(plan).copyResults(pool());
      millis.push_back(
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count());
    }
    std::sort(millis.begin(), millis.end());
    const double median = millis[millis.size() / 2];
    std::cout << "[bench] " << name << " rows/batch=" << input->size()
              << " batches=" << batches << " min_ms=" << millis.front()
              << " median_ms=" << median
              << " us/batch=" << median * 1000.0 / batches << std::endl;
  }

  int64_t rows_{0};
  int64_t batches_{0};
  int64_t iters_{0};
};

// TPC-H Q1 pricing: column-scalar and column-column ops, short inputs widening
// to long results.
TEST_F(CudfDecimalSyncBenchmark, q1Pricing) {
  const auto n = static_cast<vector_size_t>(rows_);
  auto input = makeRowVector(
      {"price", "discount", "tax"},
      {
          makeFlatVector<int64_t>(
              n,
              [](auto row) { return 100'00 + row % 9'000'000; },
              nullptr,
              DECIMAL(12, 2)),
          makeFlatVector<int64_t>(
              n, [](auto row) { return row % 11; }, nullptr, DECIMAL(12, 2)),
          makeFlatVector<int64_t>(
              n, [](auto row) { return row % 9; }, nullptr, DECIMAL(12, 2)),
      });
  run("q1Pricing",
      input,
      {"price * (CAST('1' AS DECIMAL(12, 2)) - discount) AS disc_price",
       "price * (CAST('1' AS DECIMAL(12, 2)) - discount) * "
       "(CAST('1' AS DECIMAL(12, 2)) + tax) AS charge"},
      {"sum(disc_price)", "sum(charge)"});
}

// Column-column only, no nulls: no literal decode and no null-mask merge.
TEST_F(CudfDecimalSyncBenchmark, columnColumnLong) {
  const auto n = static_cast<vector_size_t>(rows_);
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int128_t>(
              n,
              [](auto row) { return static_cast<int128_t>(row) * 1'000 + 7; },
              nullptr,
              DECIMAL(30, 4)),
          makeFlatVector<int128_t>(
              n,
              [](auto row) { return static_cast<int128_t>(row % 997) + 1; },
              nullptr,
              DECIMAL(30, 4)),
      });
  run("columnColumnLong",
      input,
      {"a + b AS s", "a - b AS d", "a * b AS p", "a / b AS q"},
      {"sum(s)", "sum(d)", "sum(p)", "sum(q)"});
}

// Divide by a literal: the per-batch zero check on the scalar.
TEST_F(CudfDecimalSyncBenchmark, divideByLiteral) {
  const auto n = static_cast<vector_size_t>(rows_);
  auto input = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>(
          n, [](auto row) { return row * 3 + 1; }, nullptr, DECIMAL(18, 2))});
  run("divideByLiteral",
      input,
      {"a / CAST('7' AS DECIMAL(5, 0)) AS q1",
       "a / CAST('3' AS DECIMAL(5, 1)) AS q2"},
      {"sum(q1)", "sum(q2)"});
}

// One projection chaining N column-column ops, swept over N: each op is one
// checked launch, which the baseline synchronizes on and the deferred status
// does not, so the saving grows with N. DECIMAL(38, 2) on both sides keeps
// every intermediate at that type, so no casts are mixed in, and alternating
// + b and - b keeps the values from growing.
TEST_F(CudfDecimalSyncBenchmark, opChain) {
  const auto rows =
      static_cast<vector_size_t>(envOr("BENCH_CHAIN_ROWS", 1 << 16));
  const auto batches = envOr("BENCH_CHAIN_BATCHES", 1000);
  std::vector<int64_t> opCounts;
  const char* opsEnv = std::getenv("BENCH_CHAIN_OPS");
  std::stringstream opsList(opsEnv == nullptr ? "1,2,4,8,16,32,64" : opsEnv);
  for (std::string count; std::getline(opsList, count, ',');) {
    opCounts.push_back(std::stoll(count));
  }

  // Concatenating batches up to the BENCH_ROWS threshold would hide the
  // per-batch cost being measured.
  auto& config = CudfConfig::getInstance();
  const auto savedThreshold = config.batchSizeMinThreshold;
  config.batchSizeMinThreshold = static_cast<int32_t>(rows);
  SCOPE_EXIT {
    config.batchSizeMinThreshold = savedThreshold;
  };

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int128_t>(
              rows,
              [](auto row) { return static_cast<int128_t>(row) * 1'000 + 7; },
              nullptr,
              DECIMAL(38, 2)),
          makeFlatVector<int128_t>(
              rows,
              [](auto row) { return static_cast<int128_t>(row % 997) + 1; },
              nullptr,
              DECIMAL(38, 2)),
      });
  for (const auto opCount : opCounts) {
    VELOX_CHECK_GT(opCount, 0, "BENCH_CHAIN_OPS entries must be positive");
    std::string chain = "a";
    for (int64_t op = 0; op < opCount; ++op) {
      chain = "(" + chain + (op % 2 == 0 ? " + b)" : " - b)");
    }
    run("opChain/" + std::to_string(opCount),
        input,
        {chain + " AS r"},
        {"sum(r)"},
        batches);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox
