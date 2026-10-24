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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/RoundRobinPartitionFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/DecimalUtil.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <limits>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <typeinfo>

namespace facebook::velox::cudf_velox {
namespace {

int64_t computeAvgRaw(const std::vector<int64_t>& values) {
  int128_t sum = 0;
  for (auto value : values) {
    sum += value;
  }
  int128_t avg = 0;
  facebook::velox::DecimalUtil::computeAverage(avg, sum, values.size(), 0);
  return static_cast<int64_t>(avg);
}

constexpr int kBitsPerWord = 8 * sizeof(cudf::bitmask_type);

std::pair<cuda::device_buffer<std::byte>, cudf::size_type> makeNullMask(
    const std::vector<bool>& valid,
    cuda::stream_ref stream) {
  auto mr = cudf::get_current_device_resource_ref();
  auto numBits = static_cast<cudf::size_type>(valid.size());
  if (numBits == 0) {
    return {
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
        0};
  }
  auto maskBytes = cudf::bitmask_allocation_size_bytes(numBits);
  auto numWords = maskBytes / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host(numWords, 0);
  cudf::size_type nullCount = 0;
  for (cudf::size_type i = 0; i < numBits; ++i) {
    if (valid[i]) {
      auto word = i / kBitsPerWord;
      auto bit = i % kBitsPerWord;
      host[word] |= (cudf::bitmask_type{1} << bit);
    } else {
      ++nullCount;
    }
  }
  auto mask = cudf::create_null_mask(
      numBits, cudf::mask_state::UNINITIALIZED, stream, mr);
  if (!host.empty()) {
    auto status = cudaMemcpyAsync(
        mask.data(),
        host.data(),
        host.size() * sizeof(cudf::bitmask_type),
        cudaMemcpyHostToDevice,
        stream.get());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.sync();
  }
  return {std::move(mask), nullCount};
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* key, const char* value) : key_(key) {
    const char* existing = std::getenv(key);
    if (existing) {
      oldValue_ = std::string(existing);
    }
    if (value) {
      setenv(key, value, 1);
    } else {
      unsetenv(key);
    }
  }

  ~ScopedEnvVar() {
    if (oldValue_) {
      setenv(key_.c_str(), oldValue_->c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
  }

 private:
  std::string key_;
  std::optional<std::string> oldValue_;
};

template <typename T>
std::unique_ptr<cudf::column> makeFixedWidthColumn(
    cudf::data_type type,
    const std::vector<T>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  auto col = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(values.size()),
      cudf::mask_state::UNALLOCATED,
      stream);
  if (!values.empty()) {
    auto status = cudaMemcpyAsync(
        col->mutable_view().data<T>(),
        values.data(),
        values.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream.get());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.sync();
  }
  if (valid) {
    auto [mask, nullCount] = makeNullMask(*valid, stream);
    col->set_null_mask(std::move(mask), nullCount);
  }
  return col;
}

template <typename T>
std::unique_ptr<cudf::column> makeDecimalColumn(
    const std::vector<T>& values,
    int32_t scale,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  cudf::type_id typeId = std::is_same_v<T, int64_t> ? cudf::type_id::DECIMAL64
                                                    : cudf::type_id::DECIMAL128;
  cudf::data_type type{typeId, -scale};
  return makeFixedWidthColumn(type, values, valid, stream);
}

std::unique_ptr<cudf::column> makeInt64Column(
    const std::vector<int64_t>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  return makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT64}, values, valid, stream);
}

template <typename T>
std::vector<T> copyColumnData(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  std::vector<T> host(view.size());
  if (view.size() == 0) {
    return host;
  }
  auto status = cudaMemcpyAsync(
      host.data(),
      view.data<T>(),
      view.size() * sizeof(T),
      cudaMemcpyDeviceToHost,
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();
  return host;
}

std::vector<cudf::bitmask_type> copyNullMask(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  auto numWords = cudf::num_bitmask_words(view.size());
  std::vector<cudf::bitmask_type> host(numWords, 0);
  if (!view.nullable() || numWords == 0) {
    return host;
  }
  auto status = cudaMemcpyAsync(
      host.data(),
      view.null_mask(),
      host.size() * sizeof(cudf::bitmask_type),
      cudaMemcpyDeviceToHost,
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();
  return host;
}

bool isValidAt(const std::vector<cudf::bitmask_type>& mask, size_t idx) {
  if (mask.empty()) {
    return true;
  }
  auto word = idx / kBitsPerWord;
  auto bit = idx % kBitsPerWord;
  return (mask[word] >> bit) & 1;
}

struct CompactRunOptions {
  std::optional<int32_t> gpuBatchRows;
  bool streaming{false};
  int32_t maxDrivers{1};
};

class CudfDecimalTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    window::prestosql::registerAllWindowFunctions();
    CudfConfig::getInstance().allowCpuFallback = false;
    // Ensure a CUDA device is selected and initialized (RMM asserts otherwise).
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
      GTEST_SKIP() << "cudaGetDeviceCount failed: " << static_cast<int>(status)
                   << " (" << cudaGetErrorString(status) << ")";
    }
    if (deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices visible (check CUDA_VISIBLE_DEVICES)";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    CudfConfig::getInstance().allowCpuFallback = false;
    exec::test::OperatorTestBase::TearDown();
  }

  void enableCpuFallback() {
    // The driver adapter captures this setting at registration.
    unregisterCudf();
    CudfConfig::getInstance().allowCpuFallback = true;
    registerCudf();
  }

  RowVectorPtr makeDecimal128Batch(
      const std::vector<int32_t>& keys,
      const std::vector<std::optional<int128_t>>& values) {
    return makeRowVector(
        {"k", "d"},
        {makeFlatVector<int32_t>(keys),
         makeNullableFlatVector<int128_t>(values, DECIMAL(38, 2))});
  }

  // Velox-core results. Disabling cuDF alone still compiles the plan for the
  // GPU when CPU fallback is disallowed.
  RowVectorPtr cpuResults(const core::PlanNodePtr& plan, int32_t maxDrivers) {
    return exec::test::AssertQueryBuilder(plan)
        .maxDrivers(maxDrivers)
        .config(CudfConfig::kCudfEnabled, false)
        .config(CudfConfig::kCudfAllowCpuFallback, true)
        .copyResults(pool());
  }

  // Runs 'original' on the CPU, then its CudfPlanRewriter rewrite on the GPU,
  // and expects the GPU to return the CPU result or throw the CPU error.
  // Returns the CPU error message (empty on success) and the GPU task (null on
  // error).
  std::pair<std::string, std::shared_ptr<exec::Task>> assertCompactMatchesCpu(
      const core::PlanNodePtr& original,
      const CompactRunOptions& options = {}) {
    RowVectorPtr expected;
    std::string error;
    try {
      expected = cpuResults(original, options.maxDrivers);
    } catch (const VeloxUserError& e) {
      error = e.message();
    }

    auto plan = CudfPlanRewriter::rewrite(original);
    EXPECT_EQ(*plan->outputType(), *original->outputType());
    auto& config = CudfConfig::getInstance();
    const auto savedStreaming = config.streamingGroupbyEnabled;
    config.streamingGroupbyEnabled = options.streaming;
    SCOPE_EXIT {
      config.streamingGroupbyEnabled = savedStreaming;
    };
    exec::test::AssertQueryBuilder builder(plan);
    builder.maxDrivers(options.maxDrivers)
        .config(core::QueryConfig::kMaxPartialAggregationMemory, 1);
    if (options.gpuBatchRows) {
      builder.config(CudfFromVelox::kGpuBatchSizeRows, *options.gpuBatchRows);
    }
    if (expected) {
      return {error, builder.assertResults(expected)};
    }
    try {
      builder.copyResults(pool());
      ADD_FAILURE() << "GPU succeeded where the CPU failed with: " << error;
    } catch (const VeloxException& e) {
      EXPECT_EQ(e.message(), error);
      EXPECT_NE(dynamic_cast<const VeloxUserError*>(&e), nullptr);
    }
    return {error, nullptr};
  }

  // assertCompactMatchesCpu over partial -> [LocalPartition] ->
  // [intermediate] -> final plans, GPU batch sizes, and (grouped) streaming.
  // Every aggregate must get a native state. Returns the CPU error, which must
  // be the same for every plan shape.
  std::string assertCompactMatrixMatchesCpu(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<std::string>& keys,
      const std::vector<std::string>& aggregates) {
    std::optional<std::string> error;
    for (bool exchange : {false, true}) {
      for (bool intermediate : {false, true}) {
        for (std::optional<int32_t> batchRows :
             {std::optional<int32_t>{}, std::optional<int32_t>{2}}) {
          for (bool streaming : {false, true}) {
            if (streaming && keys.empty()) {
              continue;
            }
            SCOPED_TRACE(
                fmt::format(
                    "exchange={} intermediate={} batchRows={} streaming={}",
                    exchange,
                    intermediate,
                    batchRows.value_or(0),
                    streaming));
            auto builder = exec::test::PlanBuilder();
            builder.values(batches).partialAggregation(keys, aggregates);
            if (exchange) {
              builder.localPartition(keys);
            }
            if (intermediate) {
              builder.intermediateAggregation();
            }
            auto original = builder.finalAggregation().planNode();
            const auto rewritten = CudfPlanRewriter::rewrite(original);
            auto* final =
                dynamic_cast<const CudfAggregationNode*>(rewritten.get());
            EXPECT_NE(final, nullptr);
            for (size_t i = 0; final && i < aggregates.size(); ++i) {
              EXPECT_TRUE(final->usesCompactDecimalState(i)) << i;
            }
            auto [cpuError, task] = assertCompactMatchesCpu(
                original,
                {.gpuBatchRows = batchRows,
                 .streaming = streaming,
                 .maxDrivers = exchange ? 2 : 1});
            EXPECT_EQ(cpuError, error.value_or(cpuError));
            error = cpuError;
            if (task && !keys.empty() && !exchange) {
              const auto stats = exec::toPlanStats(task->taskStats());
              const auto& customStats = stats.at(original->id()).customStats;
              const auto stat = streaming ? kStreamingGroupbyUsedStat
                                          : kDirectGroupbyFinalizationStat;
              EXPECT_GT(customStats.at(std::string{stat}).sum, 0);
            }
          }
        }
      }
    }
    return error.value_or("");
  }
};

TEST_F(CudfDecimalTest, mixedWidthDecimalDivision) {
  const auto rowType = ROW({
      {"short_decimal", DECIMAL(7, 2)},
      {"long_decimal", DECIMAL(20, 3)},
  });
  auto queryCtx = core::QueryCtx::create();
  core::ExecCtx execCtx(pool(), queryCtx.get());
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const std::vector<bool> shortValid{true, true, true, false, true};
  const std::vector<bool> longValid{true, true, true, true, false};
  // Narrowing this denominator to int64_t would turn it into zero.
  const int128_t largeDenominator = int128_t{1} << 64;
  auto shortDecimal = makeDecimalColumn<int64_t>(
      {1'000, -1'000, 1'000, 1'000, 0}, 2, &shortValid, stream);
  auto longDecimal = makeDecimalColumn<int128_t>(
      {2'000, 2'000, largeDenominator, 2'000, 2'000}, 3, &longValid, stream);
  const std::vector<cudf::column_view> inputs{
      shortDecimal->view(), longDecimal->view()};
  // short_decimal's zero is only reachable as a divisor when the other operand
  // is a scalar: every column operand is null on that row. Value checks for
  // that shape use a divisor column without the zero.
  auto zeroFreeShortDecimal = makeDecimalColumn<int64_t>(
      {1'000, -1'000, 1'000, 1'000, 500}, 2, &shortValid, stream);
  auto zeroFreeLongDecimal = makeDecimalColumn<int128_t>(
      {2'000, 2'000, largeDenominator, 2'000, 2'000}, 3, &longValid, stream);
  const std::vector<cudf::column_view> zeroFreeInputs{
      zeroFreeShortDecimal->view(), longDecimal->view()};

  auto assertDivision =
      [&](const std::string& sql,
          const TypePtr& expectedType,
          const std::vector<std::optional<int128_t>>& expected,
          const std::vector<cudf::column_view>& evalInputs) {
        SCOPED_TRACE(sql);
        auto expression = test_utils::optimizeTypedExpr(
            sql, rowType, queryCtx.get(), &execCtx);
        ASSERT_TRUE(expression->type()->equivalent(*expectedType));
        auto evaluator = createCudfExpression(
            expression, rowType, pool(), queryCtx->queryConfig());
        auto result = evaluator->eval(evalInputs, stream, mr);
        const auto view = asView(result);
        ASSERT_EQ(view.type(), veloxToCudfDataType(expectedType));
        ASSERT_EQ(view.size(), expected.size());
        auto values = [&]() -> std::vector<int128_t> {
          if (expectedType->isShortDecimal()) {
            const auto shortValues = copyColumnData<int64_t>(view, stream);
            return {shortValues.begin(), shortValues.end()};
          }
          return copyColumnData<int128_t>(view, stream);
        }();
        const auto nullMask = copyNullMask(view, stream);
        cudf::size_type expectedNulls = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
          ASSERT_EQ(isValidAt(nullMask, i), expected[i].has_value());
          if (expected[i]) {
            EXPECT_EQ(values[i], *expected[i]);
          } else {
            ++expectedNulls;
          }
        }
        EXPECT_EQ(view.null_count(), expectedNulls);
      };

  assertDivision(
      "short_decimal / long_decimal",
      DECIMAL(11, 3),
      {5'000, -5'000, 0, std::nullopt, std::nullopt},
      inputs);
  // The last row divides by short_decimal's zero, but long_decimal is null
  // so the row is discarded before the divisor is inspected.
  assertDivision(
      "long_decimal / short_decimal",
      DECIMAL(22, 3),
      {200, -200, 1'844'674'407'370'955'162, std::nullopt, std::nullopt},
      inputs);
  assertDivision(
      "short_decimal / CAST('2.000' AS DECIMAL(20, 3))",
      DECIMAL(11, 3),
      {5'000, -5'000, 5'000, std::nullopt, 0},
      inputs);
  assertDivision(
      "long_decimal / CAST('10.00' AS DECIMAL(7, 2))",
      DECIMAL(22, 3),
      {200, 200, 1'844'674'407'370'955'162, 200, std::nullopt},
      inputs);
  assertDivision(
      "CAST('10.00' AS DECIMAL(7, 2)) / long_decimal",
      DECIMAL(11, 3),
      {5'000, 5'000, 0, 5'000, std::nullopt},
      inputs);
  assertDivision(
      "CAST('2.000' AS DECIMAL(20, 3)) / short_decimal",
      DECIMAL(22, 3),
      {200, -200, 200, std::nullopt, 400},
      zeroFreeInputs);
  assertDivision(
      "short_decimal / CAST('18446744073709551.616' AS DECIMAL(20, 3))",
      DECIMAL(11, 3),
      {0, 0, 0, std::nullopt, 0},
      zeroFreeInputs);
  // A zero divisor on a row that survives the null stencil fails the whole
  // expression, as decimal divide does on the CPU.
  {
    const std::string sql = "CAST('2.000' AS DECIMAL(20, 3)) / short_decimal";
    SCOPED_TRACE(sql);
    auto expression =
        test_utils::optimizeTypedExpr(sql, rowType, queryCtx.get(), &execCtx);
    auto evaluator = createCudfExpression(
        expression, rowType, pool(), queryCtx->queryConfig());
    VELOX_ASSERT_USER_THROW(
        evaluator->eval(inputs, stream, mr), "Division by zero");
  }
}

TEST_F(CudfDecimalTest, decimalAvgDecimalInput) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>(
          {100, 200, 300, 400}, // 1.00, 2.00, 3.00, 4.00
          DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS avg_d"})
                  .planNode();

  auto expected = makeRowVector(
      {"avg_d"}, {makeFlatVector<int64_t>({250}, DECIMAL(12, 2))}); // 2.50

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgDecimalInputRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  // Sum = 1.60, count = 7 => 0.22857..., rounds to 0.23 at scale 2.
  std::vector<int64_t> rawValues = {100, 10, 10, 10, 10, 10, 10};
  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>(rawValues, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS avg_d"})
                  .planNode();

  auto expected = makeRowVector(
      {"avg_d"},
      {makeFlatVector<int64_t>({computeAvgRaw(rawValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgPartialFinalVarbinaryRounds) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  std::vector<int32_t> keys = {1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3};
  std::vector<int64_t> values = {100, 10, 10, 10, 10, 10, 10, 100, 1, -100, -1};

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  std::vector<std::pair<int32_t, std::vector<int64_t>>> groups = {
      {1, {100, 10, 10, 10, 10, 10, 10}},
      {2, {100, 1}},
      {3, {-100, -1}},
  };

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>(
              {computeAvgRaw(groups[0].second),
               computeAvgRaw(groups[1].second),
               computeAvgRaw(groups[2].second)},
              DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgIntermediateVarbinaryRounds) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 3}),
          makeFlatVector<int64_t>({100, 10, 100, -100}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1, 2, 3}),
          makeFlatVector<int64_t>({10, 10, 10, 10, 10, 1, -1}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  std::vector<std::pair<int32_t, std::vector<int64_t>>> groups = {
      {1, {100, 10, 10, 10, 10, 10, 10}},
      {2, {100, 1}},
      {3, {-100, -1}},
  };

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>(
              {computeAvgRaw(groups[0].second),
               computeAvgRaw(groups[1].second),
               computeAvgRaw(groups[2].second)},
              DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalPartialFinalVarbinaryRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 10, 10}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .finalAggregation()
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>({computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalIntermediateVarbinaryRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 10, 10}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>({computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>({100, 10, 10, 10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>({computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleDecimal64Overflow) {
  // 12 values of 9e17 (DECIMAL(18,0)) sum to 1.08e19, past 2^63. The sum must
  // accumulate in 128 bits or a DECIMAL64 accumulator wraps; avg is 9e17.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int64_t> values(kNumRows, kBig);

  auto input =
      makeRowVector({"d"}, {makeFlatVector<int64_t>(values, DECIMAL(18, 0))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  auto expected =
      makeRowVector({"a"}, {makeFlatVector<int64_t>({kBig}, DECIMAL(18, 0))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalPartialDecimal64Overflow) {
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int64_t> values(kNumRows, kBig);

  auto input =
      makeRowVector({"d"}, {makeFlatVector<int64_t>(values, DECIMAL(18, 0))});

  auto plan = exec::test::PlanBuilder()
                  .values({input})
                  .partialAggregation({}, {"avg(d) AS a"})
                  .finalAggregation()
                  .planNode();

  auto expected =
      makeRowVector({"a"}, {makeFlatVector<int64_t>({kBig}, DECIMAL(18, 0))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGroupbySingleDecimal64Overflow) {
  // Same overflow within a single group, exercising the groupby raw sum path.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int32_t> keys(kNumRows, 1);
  std::vector<int64_t> values(kNumRows, kBig);

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(18, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k"}, {"avg(d) AS a"})
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1}),
          makeFlatVector<int64_t>({kBig}, DECIMAL(18, 0)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  auto expected = makeRowVector(
      {"a"}, {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalPartialFinalVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"a"}, {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalIntermediateVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"a"}, {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgPartialFinalVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<int64_t>(
              {100, 200, std::nullopt, std::nullopt, 400, std::nullopt},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {150, std::nullopt, 400}, DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgIntermediateVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 400}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {200, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {150, std::nullopt, 400}, DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, streamingDecimalSumAndAvgReusesSegment) {
  auto& config = CudfConfig::getInstance();
  const auto savedEnabled = config.streamingGroupbyEnabled;
  const auto savedMultiplier = config.streamingGroupbyCapacityMultiplier;
  config.streamingGroupbyEnabled = true;
  config.streamingGroupbyCapacityMultiplier = 1.1;
  SCOPE_EXIT {
    config.streamingGroupbyEnabled = savedEnabled;
    config.streamingGroupbyCapacityMultiplier = savedMultiplier;
  };

  auto input = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2, 3}),
      makeNullableFlatVector<int64_t>(
          {999'999'999'999'999'999, -200, std::nullopt, 0}, DECIMAL(18, 2)),
      makeNullableFlatVector<int128_t>(
          {100, -200, std::nullopt, 0}, DECIMAL(20, 2)),
  });
  const std::vector<RowVectorPtr> batches(12, input);
  for (bool includeAvg : {false, true}) {
    SCOPED_TRACE(includeAvg);
    std::vector<std::string> aggregates{"sum(c1)", "sum(c2)"};
    std::vector<VectorPtr> expectedColumns{
        makeFlatVector<int32_t>({0, 1, 2, 3}),
        makeNullableFlatVector<int128_t>(
            {int128_t{999'999'999'999'999'999} * 12, -2400, std::nullopt, 0},
            DECIMAL(38, 2)),
        makeNullableFlatVector<int128_t>(
            {1200, -2400, std::nullopt, 0}, DECIMAL(38, 2))};
    if (includeAvg) {
      aggregates.push_back("avg(c1)");
      aggregates.push_back("avg(c2)");
      expectedColumns.push_back(
          makeNullableFlatVector<int64_t>(
              {999'999'999'999'999'999, -200, std::nullopt, 0},
              DECIMAL(18, 2)));
      expectedColumns.push_back(
          makeNullableFlatVector<int128_t>(
              {100, -200, std::nullopt, 0}, DECIMAL(20, 2)));
    }
    auto plan = exec::test::PlanBuilder()
                    .values(batches)
                    .partialAggregation({"c0"}, aggregates)
                    .finalAggregation()
                    .planNode();
    auto task = exec::test::AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config(CudfFromVelox::kGpuBatchSizeRows, 1)
                    .config(core::QueryConfig::kMaxPartialAggregationMemory, 1)
                    .assertResults(makeRowVector(expectedColumns));
    const auto stats = exec::toPlanStats(task->taskStats());
    const auto& customStats = stats.at(plan->id()).customStats;
    EXPECT_GT(customStats.at(std::string{kStreamingGroupbyUsedStat}).sum, 0);
    EXPECT_EQ(
        customStats.at(std::string{kStreamingGroupbySegmentsStat}).sum, 1);
  }
}

TEST_F(CudfDecimalTest, streamingDecimalAvgUnequalCounts) {
  auto& config = CudfConfig::getInstance();
  const auto savedEnabled = config.streamingGroupbyEnabled;
  config.streamingGroupbyEnabled = true;
  SCOPE_EXIT {
    config.streamingGroupbyEnabled = savedEnabled;
  };
  auto keys = makeFlatVector<int32_t>({0, 0, 1, 1, 2, 2});
  auto first = makeRowVector({
      keys,
      makeNullableFlatVector<int64_t>(
          {101, 102, -101, -102, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
      makeNullableFlatVector<int128_t>(
          {101, 102, -101, -102, std::nullopt, std::nullopt}, DECIMAL(20, 2)),
  });
  auto second = makeRowVector({
      keys,
      makeNullableFlatVector<int64_t>(
          {104, std::nullopt, -104, std::nullopt, std::nullopt, std::nullopt},
          DECIMAL(12, 2)),
      makeNullableFlatVector<int128_t>(
          {104, std::nullopt, -104, std::nullopt, std::nullopt, std::nullopt},
          DECIMAL(20, 2)),
  });
  auto expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2}),
      makeNullableFlatVector<int64_t>(
          {102, -102, std::nullopt}, DECIMAL(12, 2)),
      makeNullableFlatVector<int128_t>(
          {102, -102, std::nullopt}, DECIMAL(20, 2)),
  });
  auto plan = exec::test::PlanBuilder()
                  .values({first, second})
                  .partialAggregation({"c0"}, {"avg(c1)", "avg(c2)"})
                  .finalAggregation()
                  .planNode();
  auto task = exec::test::AssertQueryBuilder(plan)
                  .maxDrivers(1)
                  .config(CudfFromVelox::kGpuBatchSizeRows, 6)
                  .config(core::QueryConfig::kMaxPartialAggregationMemory, 1)
                  .assertResults(expected);
  const auto stats = exec::toPlanStats(task->taskStats());
  EXPECT_GT(
      stats.at(plan->id())
          .customStats.at(std::string{kStreamingGroupbyUsedStat})
          .sum,
      0);
}

TEST_F(CudfDecimalTest, directFinalizationDecimalSumAndAvg) {
  auto input = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2}),
      makeNullableFlatVector<int64_t>({100, 300, std::nullopt}, DECIMAL(12, 2)),
      makeNullableFlatVector<int128_t>(
          {100, 300, std::nullopt}, DECIMAL(20, 2)),
  });
  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeNullableFlatVector<int128_t>({800, std::nullopt}, DECIMAL(38, 2)),
      makeNullableFlatVector<int64_t>({200, std::nullopt}, DECIMAL(12, 2)),
      makeNullableFlatVector<int128_t>({800, std::nullopt}, DECIMAL(38, 2)),
      makeNullableFlatVector<int128_t>({200, std::nullopt}, DECIMAL(20, 2)),
  });
  const std::vector<std::string> aggregates{
      "sum(c1)", "avg(c1)", "sum(c2)", "avg(c2)"};
  for (bool single : {true, false}) {
    SCOPED_TRACE(single);
    auto builder = exec::test::PlanBuilder().values({input, input});
    if (single) {
      builder.singleAggregation({"c0"}, aggregates);
    } else {
      builder.partialAggregation({"c0"}, aggregates).finalAggregation();
    }
    auto plan = builder.planNode();
    auto task = exec::test::AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config(CudfFromVelox::kGpuBatchSizeRows, "3")
                    .config(core::QueryConfig::kMaxPartialAggregationMemory, 1)
                    .assertResults(expected);
    const auto stats = exec::toPlanStats(task->taskStats());
    EXPECT_GT(
        stats.at(plan->id())
            .customStats.at(std::string{kDirectGroupbyFinalizationStat})
            .sum,
        0);
  }
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalVarbinary) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) AS s FROM tmp GROUP BY k");
}

TEST_F(CudfDecimalTest, decimalPartialSumVarbinaryToVeloxRoundTrip) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .planNode();

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  VELOX_CHECK_NOT_NULL(result);
  ASSERT_GT(result->size(), 0);
  ASSERT_EQ(result->childAt(0)->type()->kind(), TypeKind::VARBINARY);
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalEmptyInput) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .filter("k < 0")
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) AS s FROM tmp WHERE k < 0 GROUP BY k");
}

TEST_F(CudfDecimalTest, decimalSumIntermediateVarbinary) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2}),
          makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({2, 3}),
          makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) AS s FROM tmp GROUP BY k");
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalVarbinary) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGlobalIntermediateVarbinary) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGlobalSingle) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>(
          {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

// Masked groupby sum: the mask null-injects raw input so cuDF sum excludes
// masked rows. Group 3 is fully masked out -> NULL. Runs on GPU (fallback off).
TEST_F(CudfDecimalTest, decimalSumMaskedGroupbySingle) {
  auto input = makeRowVector(
      {"k", "d", "m"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeFlatVector<bool>({true, false, true, true, false}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k"}, {"sum(d) AS s"}, {"m"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, sum(d) FILTER (WHERE m) AS s FROM tmp GROUP BY k");
}

// Masked groupby sum across partial + final steps: the mask applies only at the
// raw partial step and propagates through the serialized intermediate state.
TEST_F(CudfDecimalTest, decimalSumMaskedPartialFinal) {
  auto input = makeRowVector(
      {"k", "d", "m"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeFlatVector<bool>({true, false, true, true, false}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"}, {"m"})
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, sum(d) FILTER (WHERE m) AS s FROM tmp GROUP BY k");
}

// Masked global (reduce) sum, including a NULL mask entry which is excluded.
TEST_F(CudfDecimalTest, decimalSumMaskedGlobalSingle) {
  auto input = makeRowVector(
      {"d", "m"},
      {
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeNullableFlatVector<bool>({true, false, true, std::nullopt, true}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"}, {"m"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) FILTER (WHERE m) AS s FROM tmp");
}

// Masked global (reduce) sum where every row is masked out: the input reduces
// to the empty set, so the result is NULL. Runs on GPU (fallback off).
TEST_F(CudfDecimalTest, decimalSumMaskedGlobalAllMasked) {
  auto input = makeRowVector(
      {"d", "m"},
      {
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeFlatVector<bool>({false, false, false, false, false}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"}, {"m"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) FILTER (WHERE m) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGroupbySingleDecimal64Overflow) {
  // One group of 12 values of 9e17 (DECIMAL(18,0)) sums to 1.08e19, past 2^63.
  // sum(decimal(18,0)) -> decimal(38,0), computed in 128 bits, no wrap.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int32_t> keys(kNumRows, 1);
  std::vector<int64_t> values(kNumRows, kBig);

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(18, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  const int128_t expectedSum = static_cast<int128_t>(kBig) * kNumRows;
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k"}, {"sum(d) AS s"})
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1}),
          makeFlatVector<int128_t>({expectedSum}, DECIMAL(38, 0)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalSingleDecimal64Overflow) {
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int64_t> values(kNumRows, kBig);

  auto input =
      makeRowVector({"d"}, {makeFlatVector<int64_t>(values, DECIMAL(18, 0))});

  const int128_t expectedSum = static_cast<int128_t>(kBig) * kNumRows;
  auto plan = exec::test::PlanBuilder()
                  .values({input})
                  .singleAggregation({}, {"sum(d) AS s"})
                  .planNode();

  auto expected = makeRowVector(
      {"s"}, {makeFlatVector<int128_t>({expectedSum}, DECIMAL(38, 0))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);

  auto compactPlan = exec::test::PlanBuilder()
                         .values({input})
                         .partialAggregation({}, {"sum(d) AS s"})
                         .finalAggregation()
                         .planNode();
  exec::test::AssertQueryBuilder(CudfPlanRewriter::rewrite(compactPlan))
      .assertResults(expected);
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalDecimal64Overflow) {
  // Global SUM whose total overflows DECIMAL64; exercises the partial raw sum
  // (serialized to VARBINARY) and the final merge, both in 128 bits.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int64_t> values(kNumRows, kBig);

  auto input =
      makeRowVector({"d"}, {makeFlatVector<int64_t>(values, DECIMAL(18, 0))});

  std::vector<RowVectorPtr> vectors = {input};

  const int128_t expectedSum = static_cast<int128_t>(kBig) * kNumRows;
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"}, {makeFlatVector<int128_t>({expectedSum}, DECIMAL(38, 0))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<int64_t>(
              {100, 200, std::nullopt, std::nullopt, 400, std::nullopt},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int128_t>(
              {static_cast<int128_t>(300),
               std::nullopt,
               static_cast<int128_t>(400)},
              DECIMAL(38, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumIntermediateVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 400}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {200, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int128_t>(
              {static_cast<int128_t>(300),
               std::nullopt,
               static_cast<int128_t>(400)},
              DECIMAL(38, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"},
      {makeNullableFlatVector<int128_t>({std::nullopt}, DECIMAL(38, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalIntermediateVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"},
      {makeNullableFlatVector<int128_t>({std::nullopt}, DECIMAL(38, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, compactSumGroupedMixedAggregates) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
       makeNullableFlatVector<int64_t>(
           {100, -100, std::nullopt, std::nullopt, 400, std::nullopt},
           DECIMAL(12, 2))});
  auto expected = makeRowVector(
      {"k", "s", "a"},
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeNullableFlatVector<int128_t>({0, std::nullopt, 400}, DECIMAL(38, 2)),
       makeNullableFlatVector<int64_t>(
           {0, std::nullopt, 400}, DECIMAL(12, 2))});

  for (bool intermediate : {false, true}) {
    SCOPED_TRACE(intermediate);
    auto builder = exec::test::PlanBuilder();
    builder.values({input})
        .partialAggregation({"k"}, {"sum(d) AS s", "avg(d) AS a"})
        .localPartition({"k"});
    if (intermediate) {
      builder.intermediateAggregation();
    }
    auto original = builder.finalAggregation().planNode();
    auto rewritten = CudfPlanRewriter::rewrite(original);
    auto node = rewritten;
    size_t stages = 0;
    while (auto* aggregation =
               dynamic_cast<const CudfAggregationNode*>(node.get())) {
      ++stages;
      EXPECT_TRUE(aggregation->usesCompactDecimalState(0));
      EXPECT_TRUE(aggregation->usesCompactDecimalState(1));
      EXPECT_EQ(*aggregation->intermediateType(0), *DECIMAL(38, 2));
      EXPECT_EQ(
          *aggregation->intermediateType(1),
          *ROW({{"sum", DECIMAL(38, 2)}, {"count", BIGINT()}}));
      node = node->sources()[0];
      if (auto* partition =
              dynamic_cast<const core::LocalPartitionNode*>(node.get())) {
        EXPECT_EQ(partition->name(), "LocalPartition");
        EXPECT_EQ(*partition->outputType()->childAt(1), *DECIMAL(38, 2));
        EXPECT_EQ(
            *partition->outputType()->childAt(2),
            *ROW({{"sum", DECIMAL(38, 2)}, {"count", BIGINT()}}));
        node = partition->sources()[0];
      }
    }
    EXPECT_EQ(stages, intermediate ? 3 : 2);
    ASSERT_NE(dynamic_cast<const core::ValuesNode*>(node.get()), nullptr);
    EXPECT_EQ(*original->outputType(), *rewritten->outputType());
    exec::test::AssertQueryBuilder(rewritten).maxDrivers(2).assertResults(
        expected);
  }
}

TEST_F(CudfDecimalTest, streamingCompactSumThroughLocalExchange) {
  auto& config = CudfConfig::getInstance();
  const auto savedEnabled = config.streamingGroupbyEnabled;
  config.streamingGroupbyEnabled = true;
  SCOPE_EXIT {
    config.streamingGroupbyEnabled = savedEnabled;
  };

  auto input = makeRowVector({
      makeFlatVector<int32_t>({0, 0, 1, 1, 2, 2}),
      makeNullableFlatVector<int64_t>(
          {100,
           -100,
           std::nullopt,
           std::nullopt,
           999'999'999'999'999'999,
           std::nullopt},
          DECIMAL(18, 2)),
      makeNullableFlatVector<int128_t>(
          {100, -100, std::nullopt, std::nullopt, 200, std::nullopt},
          DECIMAL(20, 2)),
  });
  auto expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2}),
      makeNullableFlatVector<int128_t>(
          {0, std::nullopt, int128_t{999'999'999'999'999'999} * 3},
          DECIMAL(38, 2)),
      makeNullableFlatVector<int128_t>({0, std::nullopt, 600}, DECIMAL(38, 2)),
      makeNullableFlatVector<int64_t>(
          {0, std::nullopt, 999'999'999'999'999'999}, DECIMAL(18, 2)),
  });
  auto plan = CudfPlanRewriter::rewrite(
      exec::test::PlanBuilder()
          .values({input, input, input})
          .partialAggregation({"c0"}, {"sum(c1)", "sum(c2)", "avg(c1)"})
          .localPartition({"c0"})
          .finalAggregation()
          .planNode());
  const auto* final = dynamic_cast<const CudfAggregationNode*>(plan.get());
  ASSERT_NE(final, nullptr);
  ASSERT_TRUE(final->usesCompactDecimalState(0));
  ASSERT_TRUE(final->usesCompactDecimalState(1));
  ASSERT_TRUE(final->usesCompactDecimalState(2));
  ASSERT_EQ(
      *plan->sources()[0]->outputType()->childAt(3),
      *ROW({{"sum", DECIMAL(38, 2)}, {"count", BIGINT()}}));
  ASSERT_EQ(*plan->sources()[0]->outputType()->childAt(1), *DECIMAL(38, 2));
  ASSERT_EQ(
      *plan->sources()[0]->outputType()->childAt(2),
      *ROW({{"sum", DECIMAL(38, 2)}, {"overflow", BIGINT()}}));

  auto task = exec::test::AssertQueryBuilder(plan)
                  .maxDrivers(1)
                  .config(CudfFromVelox::kGpuBatchSizeRows, 6)
                  .config(core::QueryConfig::kMaxPartialAggregationMemory, 1)
                  .assertResults(expected);
  const auto stats = exec::toPlanStats(task->taskStats());
  EXPECT_GT(
      stats.at(plan->id())
          .customStats.at(std::string{kStreamingGroupbyUsedStat})
          .sum,
      0);
}

TEST_F(CudfDecimalTest, compactAvgGroupedBatches) {
  auto& config = CudfConfig::getInstance();
  const auto savedEnabled = config.streamingGroupbyEnabled;
  SCOPE_EXIT {
    config.streamingGroupbyEnabled = savedEnabled;
  };

  constexpr int64_t kBig = 999'999'999'999'999'999;
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({0, 0, 1, 1, 2, 2, 3, 3, 4, 5}),
       makeNullableFlatVector<int64_t>(
           {std::nullopt,
            std::nullopt,
            100,
            -100,
            100,
            101,
            -100,
            -101,
            kBig,
            100},
           DECIMAL(18, 2))});
  auto tail = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5}),
       makeNullableFlatVector<int64_t>(
           {std::nullopt, std::nullopt, std::nullopt, std::nullopt, kBig, 300},
           DECIMAL(18, 2))});
  // The large group's sum exceeds INT64_MAX. The last group has unequal
  // partial counts, so averaging partial averages would give the wrong result.
  std::vector<RowVectorPtr> batches(10, input);
  batches.push_back(tail);
  auto expected = makeRowVector(
      {"k", "a"},
      {makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5}),
       makeNullableFlatVector<int64_t>(
           {std::nullopt, 0, 101, -101, kBig, 118}, DECIMAL(18, 2))});

  for (bool streaming : {false, true}) {
    config.streamingGroupbyEnabled = streaming;
    for (bool intermediate : {false, true}) {
      SCOPED_TRACE(
          fmt::format(
              "streaming={}, intermediate={}", streaming, intermediate));
      auto builder = exec::test::PlanBuilder();
      builder.values(batches)
          .partialAggregation({"k"}, {"avg(d) AS a"})
          .localPartition({"k"});
      if (intermediate) {
        builder.intermediateAggregation();
      }
      auto original = builder.finalAggregation().planNode();
      auto plan = CudfPlanRewriter::rewrite(original);
      auto* final = dynamic_cast<const CudfAggregationNode*>(plan.get());
      ASSERT_NE(final, nullptr);
      EXPECT_TRUE(final->usesCompactDecimalState(0));
      EXPECT_EQ(
          *final->intermediateType(0),
          *ROW({{"sum", DECIMAL(38, 2)}, {"count", BIGINT()}}));
      EXPECT_EQ(*plan->outputType(), *original->outputType());
      auto task =
          exec::test::AssertQueryBuilder(plan)
              .maxDrivers(1)
              .config(CudfFromVelox::kGpuBatchSizeRows, 10)
              .config(core::QueryConfig::kMaxPartialAggregationMemory, 1)
              .assertResults(expected);
      if (streaming) {
        const auto stats = exec::toPlanStats(task->taskStats());
        EXPECT_GT(
            stats.at(plan->id())
                .customStats.at(std::string{kStreamingGroupbyUsedStat})
                .sum,
            0);
      }
    }
  }
}

TEST_F(CudfDecimalTest, compactAvgGroupedEmpty) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  auto plan = CudfPlanRewriter::rewrite(
      exec::test::PlanBuilder()
          .values({input})
          .filter("k < 0")
          .partialAggregation({"k"}, {"avg(d) AS a"})
          .intermediateAggregation()
          .finalAggregation()
          .planNode());
  exec::test::AssertQueryBuilder(plan).assertResults(makeRowVector(
      {"k", "a"},
      {makeFlatVector<int32_t>({}),
       makeFlatVector<int64_t>({}, DECIMAL(12, 2))}));
}

TEST_F(CudfDecimalTest, compactAvgThroughRenamedProject) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1, 1, 2}),
       makeNullableFlatVector<int64_t>(
           {100, 201, std::nullopt}, DECIMAL(12, 2))});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto partial = exec::test::PlanBuilder(ids)
                     .values({input})
                     .partialAggregation({"k"}, {"avg(d) AS a"})
                     .planNode();
  auto original =
      exec::test::PlanBuilder(partial, ids)
          .project({"a AS state", "k AS key"})
          .localPartition({"key"})
          .finalAggregation({"key"}, {"avg(state) AS mean"}, {{DECIMAL(12, 2)}})
          .planNode();
  auto plan = CudfPlanRewriter::rewrite(original);
  const auto stateType = ROW({{"sum", DECIMAL(38, 2)}, {"count", BIGINT()}});
  EXPECT_EQ(*plan->sources()[0]->outputType()->childAt(0), *stateType);
  EXPECT_EQ(
      *plan->sources()[0]->sources()[0]->outputType()->childAt(0), *stateType);
  EXPECT_EQ(*plan->outputType(), *original->outputType());
  exec::test::AssertQueryBuilder(plan).maxDrivers(2).assertResults(
      makeRowVector(
          {"key", "mean"},
          {makeFlatVector<int32_t>({1, 2}),
           makeNullableFlatVector<int64_t>(
               {151, std::nullopt}, DECIMAL(12, 2))}));
}

TEST_F(CudfDecimalTest, compactAvgSlicedStateAndFinalization) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  auto plan = CudfPlanRewriter::rewrite(
      exec::test::PlanBuilder()
          .values({input})
          .partialAggregation({"k"}, {"avg(d) AS a"})
          .finalAggregation()
          .planNode());
  auto* final = dynamic_cast<const CudfAggregationNode*>(plan.get());
  ASSERT_NE(final, nullptr);
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const std::vector<bool> valid{true, true, true, false, true};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(
      makeDecimalColumn<int128_t>({999, 201, -201, 0, 999}, 2, &valid, stream));
  children.push_back(makeInt64Column({99, 2, 2, 0, 99}, nullptr, stream));
  auto state = cudf::make_structs_column(
      5,
      std::move(children),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
      stream,
      mr);
  auto slice = cudf::slice(state->view(), {1, 4}, stream)[0];
  auto keys = makeFixedWidthColumn<int32_t>(
      cudf::data_type{cudf::type_id::INT32}, {1, 2, 3}, nullptr, stream);
  cudf::table_view table({keys->view(), slice});

  auto aggregators = toGroupbyAggregators(
      *final, final->step(), final->outputType(), {nullptr}, {});
  ASSERT_EQ(aggregators.size(), 1);
  std::vector<cudf::groupby::aggregation_request> requests;
  aggregators[0]->addGroupbyRequest(table, requests, stream, mr);
  ASSERT_EQ(requests.size(), 2);
  EXPECT_EQ(
      copyColumnData<int128_t>(requests[0].values, stream),
      (std::vector<int128_t>{201, -201, 0}));
  EXPECT_EQ(requests[0].values.null_count(), 1);
  EXPECT_EQ(
      copyColumnData<int64_t>(requests[1].values, stream),
      (std::vector<int64_t>{2, 2, 0}));
  auto checkAverage = [&](const cudf::column_view& result) {
    EXPECT_EQ(result.type(), cudf::data_type(cudf::type_id::DECIMAL64, -2));
    ASSERT_EQ(result.size(), 3);
    const auto values = copyColumnData<int64_t>(result, stream);
    EXPECT_EQ(values[0], 101);
    EXPECT_EQ(values[1], -101);
    EXPECT_EQ(result.null_count(), 1);
    const auto mask = copyNullMask(result, stream);
    EXPECT_TRUE(isValidAt(mask, 0));
    EXPECT_TRUE(isValidAt(mask, 1));
    EXPECT_FALSE(isValidAt(mask, 2));
  };
  auto result = aggregators[0]->finalize(
      std::make_unique<cudf::column>(slice, stream, mr), stream, mr);
  checkAverage(result->view());

  auto streaming = toStreamingGroupbyAggregators(
      *final,
      final->sources()[0]->outputType(),
      {0, 1},
      final->outputType(),
      {nullptr},
      {});
  ASSERT_TRUE(streaming.has_value());
  ASSERT_EQ(streaming->size(), 1);
  std::vector<cudf::column_view> prepared;
  streaming->at(0)->prepareInput(table, prepared, stream);
  ASSERT_EQ(prepared.size(), 2);
  EXPECT_EQ(
      copyColumnData<int128_t>(prepared[0], stream),
      (std::vector<int128_t>{201, -201, 0}));
  EXPECT_EQ(prepared[0].null_count(), 1);
  EXPECT_EQ(
      copyColumnData<int64_t>(prepared[1], stream),
      (std::vector<int64_t>{2, 2, 0}));
  std::vector<cudf::groupby::streaming_aggregation_request> streamingRequests;
  streaming->at(0)->addStreamingRequest(streamingRequests);
  ASSERT_EQ(streamingRequests.size(), 2);
  // Each key has one state, so the merged results equal the prepared children.
  std::vector<cudf::groupby::aggregation_result> results(2);
  for (size_t i = 0; i < prepared.size(); ++i) {
    results[i].results.push_back(
        std::make_unique<cudf::column>(prepared[i], stream, mr));
  }
  result = streaming->at(0)->makeOutputColumn(results, stream, mr);
  checkAverage(result->view());
}

TEST_F(CudfDecimalTest, compactAvgLeavesGlobalDecimal64Unchanged) {
  auto input = makeRowVector(
      {"k", "d", "l"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>({100}, DECIMAL(20, 2))});
  const auto avgState = ROW(
      {{"sum", DECIMAL(38, 2)}, {"count", BIGINT()}, {"overflow", BIGINT()}});
  const auto sumState = ROW({{"sum", DECIMAL(38, 2)}, {"overflow", BIGINT()}});
  for (const auto& keys :
       {std::vector<std::string>{}, std::vector<std::string>{"k"}}) {
    auto original =
        exec::test::PlanBuilder()
            .values({input})
            .partialAggregation(keys, {"avg(d)", "avg(l)", "sum(l)"})
            .finalAggregation()
            .planNode();
    auto plan = CudfPlanRewriter::rewrite(original);
    const auto& stateType = plan->sources()[0]->outputType();
    EXPECT_EQ(*stateType->childAt(keys.size() + 1), *avgState);
    EXPECT_EQ(*stateType->childAt(keys.size() + 2), *sumState);
    auto* final = dynamic_cast<const CudfAggregationNode*>(plan.get());
    ASSERT_NE(final, nullptr);
    EXPECT_EQ(final->usesCompactDecimalState(0), !keys.empty());
    EXPECT_TRUE(final->usesCompactDecimalState(1));
    EXPECT_TRUE(final->usesCompactDecimalState(2));
    if (keys.empty()) {
      EXPECT_EQ(*stateType->childAt(0), *VARBINARY());
    }
    EXPECT_EQ(*plan->outputType(), *original->outputType());
    exec::test::AssertQueryBuilder(plan).assertResults(cpuResults(original, 1));
  }
}

TEST_F(CudfDecimalTest, compactSumGlobalNullZeroAndEmpty) {
  const std::vector<std::vector<std::optional<int64_t>>> cases = {
      {100, -100, std::nullopt},
      {std::nullopt, std::nullopt},
      {},
      {999999999999LL, 999999999999LL}};
  const std::vector<std::optional<int128_t>> sums = {
      0, std::nullopt, std::nullopt, int128_t{1999999999998LL}};
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    auto input = makeRowVector(
        {"d"}, {makeNullableFlatVector<int64_t>(cases[i], DECIMAL(12, 2))});
    auto expected = makeRowVector(
        {"s"}, {makeNullableFlatVector<int128_t>({sums[i]}, DECIMAL(38, 2))});
    for (bool intermediate : {false, true}) {
      SCOPED_TRACE(intermediate);
      auto builder = exec::test::PlanBuilder();
      builder.values({input}).partialAggregation({}, {"sum(d) AS s"});
      if (intermediate) {
        builder.intermediateAggregation();
      }
      auto rewritten =
          CudfPlanRewriter::rewrite(builder.finalAggregation().planNode());
      ASSERT_NE(
          dynamic_cast<const CudfAggregationNode*>(rewritten.get()), nullptr);
      EXPECT_EQ(
          *rewritten->sources()[0]->outputType()->childAt(0), *DECIMAL(38, 2));
      exec::test::AssertQueryBuilder(rewritten).assertResults(expected);
    }
  }
}

TEST_F(CudfDecimalTest, compactSumGroupedEmpty) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  auto original = exec::test::PlanBuilder()
                      .values({input})
                      .filter("k < 0")
                      .partialAggregation({"k"}, {"sum(d) AS s"})
                      .intermediateAggregation()
                      .finalAggregation()
                      .planNode();
  auto rewritten = CudfPlanRewriter::rewrite(original);
  auto expected = makeRowVector(
      {"k", "s"},
      {makeFlatVector<int32_t>({}),
       makeFlatVector<int128_t>({}, DECIMAL(38, 2))});
  exec::test::AssertQueryBuilder(rewritten).assertResults(expected);
}

TEST_F(CudfDecimalTest, compactSumSurroundingNodePlans) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1, 1, 2}),
       makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2))});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto makeAggregation = [&](bool compact) {
    auto builder = exec::test::PlanBuilder(ids).values({input});
    if (compact) {
      builder.partialAggregation({"k"}, {"sum(d) AS s"}).finalAggregation();
    } else {
      builder.singleAggregation({"k"}, {"sum(d) AS s"});
    }
    return builder.planNode();
  };
  auto checkPreserved = [&](auto&& self,
                            const core::PlanNodePtr& original,
                            const core::PlanNodePtr& rewritten) -> void {
    EXPECT_EQ(original->id(), rewritten->id());
    EXPECT_EQ(*original->outputType(), *rewritten->outputType());
    if (auto* aggregation =
            dynamic_cast<const core::AggregationNode*>(original.get());
        aggregation &&
        aggregation->step() == core::AggregationNode::Step::kFinal) {
      auto* compact = dynamic_cast<const CudfAggregationNode*>(rewritten.get());
      ASSERT_NE(compact, nullptr);
      EXPECT_TRUE(compact->usesCompactDecimalState(0));
      EXPECT_EQ(
          *compact->sources()[0]->outputType()->childAt(1), *DECIMAL(38, 2));
      return;
    }
    EXPECT_EQ(typeid(*original), typeid(*rewritten));
    auto originalOptions = original->serialize();
    auto rewrittenOptions = rewritten->serialize();
    originalOptions.erase("sources");
    rewrittenOptions.erase("sources");
    EXPECT_EQ(originalOptions, rewrittenOptions);
    ASSERT_EQ(original->sources().size(), rewritten->sources().size());
    for (size_t i = 0; i < original->sources().size(); ++i) {
      self(self, original->sources()[i], rewritten->sources()[i]);
    }
  };

  for (bool compactLeft : {false, true}) {
    for (bool compactRight : {false, true}) {
      SCOPED_TRACE(fmt::format("left={}, right={}", compactLeft, compactRight));
      auto left = makeAggregation(compactLeft);
      auto right = exec::test::PlanBuilder(makeAggregation(compactRight), ids)
                       .project({"k AS rk", "s AS rs"})
                       .planNode();
      std::vector<core::PlanNodePtr> plans;
      for (bool partial : {false, true}) {
        plans.push_back(
            exec::test::PlanBuilder(left, ids)
                .topN({"s DESC NULLS FIRST", "k ASC"}, 2, partial)
                .planNode());
        plans.push_back(
            exec::test::PlanBuilder(left, ids)
                .orderBy({"s DESC NULLS FIRST", "k ASC"}, partial)
                .limit(1, 2, partial)
                .planNode());
      }
      plans.push_back(
          exec::test::PlanBuilder(left, ids)
              .hashJoin(
                  {"k"},
                  {"rk"},
                  right,
                  "k > 0",
                  {"rs", "k", "s"},
                  core::JoinType::kLeft)
              .planNode());
      plans.push_back(
          exec::test::PlanBuilder(left, ids)
              .orderBy({"k"}, false)
              .mergeJoin(
                  {"k"},
                  {"rk"},
                  exec::test::PlanBuilder(right, ids)
                      .orderBy({"rk"}, false)
                      .planNode(),
                  "k > 0",
                  {"rs", "k", "s"},
                  core::JoinType::kLeft)
              .planNode());
      plans.push_back(
          exec::test::PlanBuilder(left, ids)
              .nestedLoopJoin(
                  right, "k < rk", {"rs", "k", "s"}, core::JoinType::kLeft)
              .planNode());
      plans.push_back(
          exec::test::PlanBuilder(ids)
              .localMerge(
                  {"s DESC NULLS FIRST", "k ASC"},
                  {exec::test::PlanBuilder(left, ids)
                       .orderBy({"s DESC NULLS FIRST", "k ASC"}, true)
                       .planNode(),
                   exec::test::PlanBuilder(makeAggregation(compactRight), ids)
                       .orderBy({"s DESC NULLS FIRST", "k ASC"}, true)
                       .planNode()})
              .planNode());
      for (const auto& plan : plans) {
        SCOPED_TRACE(plan->toString(true, true));
        const auto original = plan->serialize();
        auto rewritten = CudfPlanRewriter::rewrite(plan);
        checkPreserved(checkPreserved, plan, rewritten);
        EXPECT_EQ(CudfPlanRewriter::rewrite(rewritten), rewritten);
        EXPECT_EQ(plan->serialize(), original);
        if (!compactLeft && !compactRight) {
          EXPECT_EQ(rewritten, plan);
        }
      }
    }
  }
}

TEST_F(CudfDecimalTest, compactSumSurroundingNodeResults) {
  enableCpuFallback();
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1, 1, 2, 2, 3, 4}),
       makeNullableFlatVector<int64_t>(
           {100, -100, 300, 400, std::nullopt, -200}, DECIMAL(12, 2))});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto aggregation = exec::test::PlanBuilder(ids)
                         .values({input})
                         .partialAggregation({"k"}, {"sum(d) AS s"})
                         .finalAggregation()
                         .planNode();
  auto right =
      exec::test::PlanBuilder(ids)
          .values({makeRowVector({"rk"}, {makeFlatVector<int32_t>({1, 2, 4})})})
          .planNode();
  const std::vector<core::PlanNodePtr> plans = {
      exec::test::PlanBuilder(aggregation, ids)
          .topN({"s DESC NULLS FIRST", "k ASC"}, 3, false)
          .planNode(),
      exec::test::PlanBuilder(aggregation, ids)
          .orderBy({"s DESC NULLS FIRST", "k ASC"}, false)
          .limit(1, 2, false)
          .planNode(),
      exec::test::PlanBuilder(aggregation, ids)
          .hashJoin({"k"}, {"rk"}, right, "", {"k", "s"})
          .planNode(),
      exec::test::PlanBuilder(aggregation, ids)
          .nestedLoopJoin(right, "k < rk", {"k", "s", "rk"})
          .planNode()};
  for (size_t i = 0; i < plans.size(); ++i) {
    const auto& plan = plans[i];
    SCOPED_TRACE(plan->toString(true, true));
    auto expected = cpuResults(plan, 1);
    auto query =
        exec::test::AssertQueryBuilder(CudfPlanRewriter::rewrite(plan));
    query.config(CudfConfig::kCudfEnabled, true).maxDrivers(1);
    if (i < 2) {
      facebook::velox::test::assertEqualVectors(
          expected, query.copyResults(pool()));
    } else {
      query.assertResults(expected);
    }
  }
}

TEST_F(CudfDecimalTest, compactSumRejectsSurroundingNodesOnPartialState) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto partial = exec::test::PlanBuilder(ids)
                     .values({input})
                     .partialAggregation({"k"}, {"sum(d) AS s"})
                     .planNode();
  auto right =
      exec::test::PlanBuilder(ids)
          .values({makeRowVector({"rk"}, {makeFlatVector<int32_t>({1})})})
          .planNode();
  const std::vector<core::PlanNodePtr> plans = {
      exec::test::PlanBuilder(partial, ids).topN({"k"}, 1, false).planNode(),
      exec::test::PlanBuilder(partial, ids).limit(0, 1, false).planNode(),
      exec::test::PlanBuilder(partial, ids)
          .hashJoin({"k"}, {"rk"}, right, "", {"k", "s"})
          .planNode(),
      exec::test::PlanBuilder(right, ids)
          .hashJoin({"rk"}, {"k"}, partial, "", {"k", "s"})
          .planNode()};
  for (const auto& plan : plans) {
    SCOPED_TRACE(plan->toString(true, true));
    VELOX_ASSERT_THROW(CudfPlanRewriter::rewrite(plan), "compact");
    auto final =
        exec::test::PlanBuilder(plan, ids)
            .finalAggregation({"k"}, {"sum(s) AS s"}, {{DECIMAL(12, 2)}})
            .planNode();
    VELOX_ASSERT_THROW(
        CudfPlanRewriter::rewrite(final),
        "Unsupported compact aggregation state source");
  }
}

TEST_F(CudfDecimalTest, compactSumUcxFragmentSchemasAndSerde) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  auto producer = exec::test::PlanBuilder()
                      .values({input})
                      .partialAggregation({"k"}, {"sum(d) AS s", "avg(d) AS a"})
                      .partitionedOutput(
                          {"k"},
                          2,
                          false,
                          {},
                          "Presto",
                          std::string{core::TransportKind::kUcx})
                      .planNode();
  auto consumer = exec::test::PlanBuilder()
                      .exchange(
                          producer->outputType(),
                          "Presto",
                          std::string{core::TransportKind::kUcx})
                      .finalAggregation(
                          {"k"},
                          {"sum(s) AS s", "avg(a) AS a"},
                          {{DECIMAL(12, 2)}, {DECIMAL(12, 2)}})
                      .planNode();
  auto gpuProducer = CudfPlanRewriter::rewrite(producer);
  auto gpuConsumer = CudfPlanRewriter::rewrite(consumer);
  auto* output =
      dynamic_cast<const core::PartitionedOutputNode*>(gpuProducer.get());
  auto* exchange =
      dynamic_cast<const core::ExchangeNode*>(gpuConsumer->sources()[0].get());
  ASSERT_NE(output, nullptr);
  ASSERT_NE(exchange, nullptr);
  EXPECT_EQ(output->name(), producer->name());
  EXPECT_EQ(exchange->name(), consumer->sources()[0]->name());
  EXPECT_EQ(output->transportKind(), core::TransportKind::kUcx);
  EXPECT_EQ(exchange->transportKind(), core::TransportKind::kUcx);
  EXPECT_EQ(*output->outputType(), *exchange->outputType());
  EXPECT_EQ(*output->outputType()->childAt(1), *DECIMAL(38, 2));
  EXPECT_EQ(
      *output->outputType()->childAt(2),
      *ROW({{"sum", DECIMAL(38, 2)}, {"count", BIGINT()}}));
  EXPECT_EQ(*producer->outputType()->childAt(1), *VARBINARY());
  EXPECT_EQ(*gpuConsumer->outputType(), *consumer->outputType());

  Type::registerSerDe();
  core::ITypedExpr::registerSerDe();
  core::PlanNode::registerSerDe();
  CudfAggregationNode::registerSerDe();
  auto copy = ISerializable::deserialize<core::PlanNode>(
      gpuConsumer->serialize(), pool());
  auto* aggregation = dynamic_cast<const CudfAggregationNode*>(copy.get());
  ASSERT_NE(aggregation, nullptr);
  EXPECT_TRUE(aggregation->usesCompactDecimalState(0));
  EXPECT_TRUE(aggregation->usesCompactDecimalState(1));
  EXPECT_EQ(*aggregation->intermediateType(0), *DECIMAL(38, 2));
  EXPECT_EQ(
      *aggregation->intermediateType(1),
      *ROW({{"sum", DECIMAL(38, 2)}, {"count", BIGINT()}}));
  EXPECT_EQ(copy->serialize(), gpuConsumer->serialize());
}

TEST_F(CudfDecimalTest, compactSumThroughLocalExchangeProject) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  auto partial = exec::test::PlanBuilder()
                     .values({input})
                     .partialAggregation({"k"}, {"sum(d) AS s"})
                     .planNode();
  auto plan = exec::test::PlanBuilder()
                  .exchange(
                      partial->outputType(),
                      "Presto",
                      std::string{core::TransportKind::kUcx})
                  .project({"s AS state", "k AS key"})
                  .localPartition({"key"})
                  .finalAggregation(
                      {"key"}, {"sum(state) AS total"}, {{DECIMAL(12, 2)}})
                  .planNode();

  auto rewritten = CudfPlanRewriter::rewrite(plan);
  auto* partition = dynamic_cast<const core::LocalPartitionNode*>(
      rewritten->sources()[0].get());
  ASSERT_NE(partition, nullptr);
  auto* project =
      dynamic_cast<const core::ProjectNode*>(partition->sources()[0].get());
  ASSERT_NE(project, nullptr);
  auto* exchange =
      dynamic_cast<const core::ExchangeNode*>(project->sources()[0].get());
  ASSERT_NE(exchange, nullptr);
  EXPECT_EQ(*exchange->outputType()->childAt(1), *DECIMAL(38, 2));
  EXPECT_EQ(*project->outputType()->childAt(0), *DECIMAL(38, 2));
  EXPECT_EQ(*partition->outputType()->childAt(0), *DECIMAL(38, 2));
  EXPECT_EQ(*rewritten->outputType(), *plan->outputType());
}

TEST_F(CudfDecimalTest, compactSumThroughRoundRobinPartition) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1, 1, 2, 2}),
       makeFlatVector<int64_t>({100, 200, 300, 400}, DECIMAL(12, 2))});
  auto plan =
      exec::test::PlanBuilder()
          .values({input})
          .partialAggregation({"k"}, {"sum(d) AS s"})
          .localPartitionRoundRobin()
          .finalAggregation({"k"}, {"sum(s) AS total"}, {{DECIMAL(12, 2)}})
          .planNode();

  auto rewritten = CudfPlanRewriter::rewrite(plan);
  auto* partition = dynamic_cast<const core::LocalPartitionNode*>(
      rewritten->sources()[0].get());
  ASSERT_NE(partition, nullptr);
  EXPECT_NE(
      dynamic_cast<const exec::RoundRobinPartitionFunctionSpec*>(
          &partition->partitionFunctionSpec()),
      nullptr);
  EXPECT_EQ(*partition->outputType()->findChild("s"), *DECIMAL(38, 2));
  EXPECT_EQ(*rewritten->outputType(), *plan->outputType());
  exec::test::AssertQueryBuilder(rewritten).maxDrivers(1).assertResults(
      makeRowVector(
          {"k", "total"},
          {makeFlatVector<int32_t>({1, 2}),
           makeFlatVector<int128_t>({300, 700}, DECIMAL(38, 2))}));
}

TEST_F(CudfDecimalTest, compactSumThroughEnforceSingleRow) {
  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2))});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto partial = exec::test::PlanBuilder(ids)
                     .values({input})
                     .partialAggregation({}, {"sum(d) AS s"})
                     .planNode();
  auto plan = exec::test::PlanBuilder(partial, ids)
                  .enforceSingleRow()
                  .finalAggregation({}, {"sum(s) AS total"}, {{DECIMAL(12, 2)}})
                  .planNode();

  auto rewritten = CudfPlanRewriter::rewrite(plan);
  auto* singleRow = dynamic_cast<const core::EnforceSingleRowNode*>(
      rewritten->sources()[0].get());
  ASSERT_NE(singleRow, nullptr);
  EXPECT_EQ(*singleRow->outputType()->findChild("s"), *DECIMAL(38, 2));
  exec::test::AssertQueryBuilder(rewritten).maxDrivers(1).assertResults(
      makeRowVector(
          {"total"}, {makeFlatVector<int128_t>({600}, DECIMAL(38, 2))}));
}

TEST_F(CudfDecimalTest, compactSumThroughWindow) {
  auto input = makeRowVector(
      {"k", "o", "d"},
      {makeFlatVector<int32_t>({1, 1, 2}),
       makeFlatVector<int32_t>({1, 2, 1}),
       makeFlatVector<int64_t>({100, 200, 400}, DECIMAL(12, 2))});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto partial = exec::test::PlanBuilder(ids)
                     .values({input})
                     .partialAggregation({"k", "o"}, {"sum(d) AS s"})
                     .planNode();
  auto plan =
      exec::test::PlanBuilder(partial, ids)
          .window({"row_number() over (partition by k order by o) AS rn"})
          .finalAggregation({"k"}, {"sum(s) AS total"}, {{DECIMAL(12, 2)}})
          .planNode();

  auto rewritten = CudfPlanRewriter::rewrite(plan);
  auto* window =
      dynamic_cast<const core::WindowNode*>(rewritten->sources()[0].get());
  ASSERT_NE(window, nullptr);
  EXPECT_EQ(*window->inputType()->findChild("s"), *DECIMAL(38, 2));
  EXPECT_EQ(*window->outputType()->findChild("s"), *DECIMAL(38, 2));
  exec::test::AssertQueryBuilder(rewritten).maxDrivers(1).assertResults(
      makeRowVector(
          {"k", "total"},
          {makeFlatVector<int32_t>({1, 2}),
           makeFlatVector<int128_t>({300, 400}, DECIMAL(38, 2))}));
}

TEST_F(CudfDecimalTest, compactSumThroughGroupId) {
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1, 1, 2, 2}),
       makeFlatVector<int64_t>({100, 200, 300, 400}, DECIMAL(12, 2))});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto partial = exec::test::PlanBuilder(ids)
                     .values({input})
                     .partialAggregation({"k"}, {"sum(d) AS s"})
                     .planNode();
  auto plan =
      exec::test::PlanBuilder(partial, ids)
          .groupId({"k"}, {{"k"}, {}}, {"s"})
          .finalAggregation(
              {"k", "group_id"}, {"sum(s) AS total"}, {{DECIMAL(12, 2)}})
          .planNode();

  auto rewritten = CudfPlanRewriter::rewrite(plan);
  auto* groupId =
      dynamic_cast<const core::GroupIdNode*>(rewritten->sources()[0].get());
  ASSERT_NE(groupId, nullptr);
  EXPECT_EQ(*groupId->aggregationInputs()[0]->type(), *DECIMAL(38, 2));
  EXPECT_EQ(
      *groupId->sources()[0]->outputType()->findChild("s"), *DECIMAL(38, 2));
  exec::test::AssertQueryBuilder(rewritten).maxDrivers(1).assertResults(
      makeRowVector(
          {"k", "group_id", "total"},
          {makeNullableFlatVector<int32_t>({1, 2, std::nullopt}),
           makeFlatVector<int64_t>({0, 0, 1}),
           makeFlatVector<int128_t>({300, 700, 1'000}, DECIMAL(38, 2))}));
}

TEST_F(CudfDecimalTest, compactSumRejectsGpuDisabledWithCpuFallback) {
  enableCpuFallback();
  auto input =
      makeRowVector({"d"}, {makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  auto plan = exec::test::PlanBuilder()
                  .values({input})
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();
  auto rewritten = CudfPlanRewriter::rewrite(plan);
  ASSERT_NE(dynamic_cast<const CudfAggregationNode*>(rewritten.get()), nullptr);
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(rewritten)
          .config(CudfConfig::kCudfEnabled, false)
          .copyResults(pool()),
      "Compact aggregation state requires cuDF to be enabled");
}

TEST_F(CudfDecimalTest, compactSumRejectsUnsupportedAggregateWithCpuFallback) {
  enableCpuFallback();
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  auto plan =
      exec::test::PlanBuilder()
          .values({input})
          .partialAggregation({"k"}, {"sum(d) AS s", "array_agg(k) AS a"})
          .finalAggregation()
          .planNode();
  auto rewritten = CudfPlanRewriter::rewrite(plan);
  auto* aggregation = dynamic_cast<const CudfAggregationNode*>(rewritten.get());
  ASSERT_NE(aggregation, nullptr);
  EXPECT_TRUE(aggregation->usesCompactDecimalState(0));
  EXPECT_FALSE(aggregation->usesCompactDecimalState(1));
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(rewritten)
          .config(CudfConfig::kCudfEnabled, true)
          .copyResults(pool()),
      "Compact aggregation state requires GPU execution; CPU fallback is not supported");
}

TEST_F(CudfDecimalTest, compactSumRejectsUnsupportedBoundariesAndModifiers) {
  auto input = makeRowVector(
      {"d", "m"},
      {makeFlatVector<int64_t>({100}, DECIMAL(12, 2)),
       makeFlatVector<bool>({true})});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto partial = exec::test::PlanBuilder(ids)
                     .values({input})
                     .partialAggregation({}, {"sum(d) AS s"})
                     .planNode();
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(partial),
      "Compact aggregation state requires UCX partitioned output");
  auto output =
      exec::test::PlanBuilder(partial, ids).partitionedOutput({}, 1).planNode();
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(output),
      "Compact aggregation state cannot leave a task over non-UCX output");
  auto consumer = exec::test::PlanBuilder()
                      .exchange(partial->outputType(), "Presto")
                      .finalAggregation({}, {"sum(s) AS s"}, {{DECIMAL(12, 2)}})
                      .planNode();
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(consumer),
      "Compact aggregation state requires an unordered UCX exchange");
  auto masked = exec::test::PlanBuilder()
                    .values({input})
                    .partialAggregation({}, {"sum(d) AS s"}, {"m"})
                    .finalAggregation()
                    .planNode();
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(masked),
      "Compact Decimal64 aggregation does not support DISTINCT, ordering or masks");
  auto projected =
      exec::test::PlanBuilder(partial, ids)
          .project({"s", "1 AS computed"})
          .finalAggregation({}, {"sum(s) AS s"}, {{DECIMAL(12, 2)}})
          .planNode();
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(projected),
      "Compact aggregation state project must contain only field references");
}

// CPU LongDecimalWithOverflowState accumulation of 'values'.
std::pair<int128_t, int64_t> cpuDecimal128State(
    const std::vector<int128_t>& values) {
  int128_t sum = 0;
  int64_t overflow = 0;
  for (auto value : values) {
    overflow += DecimalUtil::addWithOverflow(sum, sum, value);
  }
  return {sum, overflow};
}

int128_t cpuDecimal128Average(const std::vector<int128_t>& values) {
  auto [sum, overflow] = cpuDecimal128State(values);
  int128_t average = 0;
  DecimalUtil::computeAverage(average, sum, values.size(), overflow);
  return average;
}

TEST_F(CudfDecimalTest, decimal128StateCarriesOverflowUntilCancellation) {
  // Groups 0 and 1 overflow int128 in opposite directions within each
  // partial batch and cancel back into range only when partials merge. Like
  // the CPU, the GPU must carry the overflow instead of wrapping or failing.
  const auto kMax = DecimalUtil::kLongDecimalMax;
  const std::vector<RowVectorPtr> batches = {
      makeDecimal128Batch(
          {0, 0, 1, 1, 2, 3, 4},
          {kMax, kMax, -kMax, -kMax, std::nullopt, 42, kMax}),
      makeDecimal128Batch(
          {0, 0, 1, 1, 2, 3, 4},
          {-kMax,
           -kMax + 5,
           kMax,
           kMax - 7,
           std::nullopt,
           std::nullopt,
           -kMax}),
      makeDecimal128Batch({0, 1, 4, 5}, {0, 0, 0, kMax}),
  };
  EXPECT_EQ(
      assertCompactMatrixMatchesCpu(batches, {"k"}, {"sum(d)", "avg(d)"}), "");
  auto plan = CudfPlanRewriter::rewrite(
      exec::test::PlanBuilder()
          .values(batches)
          .partialAggregation({"k"}, {"sum(d)", "avg(d)"})
          .finalAggregation()
          .planNode());
  exec::test::AssertQueryBuilder(plan)
      .config(CudfFromVelox::kGpuBatchSizeRows, 7)
      .assertResults(makeRowVector({
          makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5}),
          makeNullableFlatVector<int128_t>(
              {5, -7, std::nullopt, 42, 0, kMax}, DECIMAL(38, 2)),
          makeNullableFlatVector<int128_t>(
              {1, -1, std::nullopt, 42, 0, kMax}, DECIMAL(38, 2)),
      }));

  const std::vector<RowVectorPtr> globalBatches = {
      makeDecimal128Batch({0, 0, 0}, {kMax, kMax, kMax}),
      makeDecimal128Batch({0, 0, 0}, {-kMax, -kMax, -kMax + 11}),
      makeDecimal128Batch({0}, {std::nullopt}),
  };
  EXPECT_EQ(
      assertCompactMatrixMatchesCpu(globalBatches, {}, {"sum(d)", "avg(d)"}),
      "");
}

TEST_F(CudfDecimalTest, decimal128SumOverflowErrorsMatchCpu) {
  const auto kMax = DecimalUtil::kLongDecimalMax;
  // 2^127 - 1.
  const auto kInt128Max = std::numeric_limits<int128_t>::max();
  const auto rangeError = [](int128_t value) {
    return fmt::format(
        "Decimal overflow. Value '{}' is not in the range of Decimal Type",
        value);
  };
  struct Case {
    std::vector<std::vector<int128_t>> batches;
    std::string error;
  };
  // Values in int128 but outside DECIMAL(38) fail the range check. Values that
  // reach +/-2^127 fail adjustSumForOverflow, including exactly -2^127.
  const std::vector<Case> cases = {
      {{{kMax}, {1}}, rangeError(kMax + 1)},
      {{{-kMax}, {-1}}, rangeError(-kMax - 1)},
      {{{kMax, kInt128Max - kMax}}, rangeError(kInt128Max)},
      {{{kMax}, {kInt128Max - kMax + 1}}, "Decimal overflow"},
      {{{-kMax}, {-(kInt128Max - kMax + 1)}}, "Decimal overflow"},
      {{{kMax, kMax}}, "Decimal overflow"},
      {{{-kMax}, {-kMax}}, "Decimal overflow"},
      {{{kMax, kMax, kMax}, {kMax, kMax}}, "Decimal overflow"},
  };
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    std::vector<RowVectorPtr> grouped;
    std::vector<RowVectorPtr> global;
    for (const auto& values : cases[i].batches) {
      std::vector<std::optional<int128_t>> column(values.begin(), values.end());
      global.push_back(
          makeDecimal128Batch(std::vector<int32_t>(values.size(), 0), column));
      // A second, valid group shows only the overflowing group fails.
      std::vector<int32_t> keys(values.size(), 0);
      keys.push_back(1);
      column.push_back(1);
      grouped.push_back(makeDecimal128Batch(keys, column));
    }
    EXPECT_EQ(
        assertCompactMatrixMatchesCpu(grouped, {"k"}, {"sum(d)"}),
        cases[i].error);
    EXPECT_EQ(
        assertCompactMatrixMatchesCpu(global, {}, {"sum(d)"}), cases[i].error);
    // AVG of the same state never fails.
    EXPECT_EQ(assertCompactMatrixMatchesCpu(grouped, {"k"}, {"avg(d)"}), "");
    EXPECT_EQ(assertCompactMatrixMatchesCpu(global, {}, {"avg(d)"}), "");
  }
}

TEST_F(CudfDecimalTest, decimal128AvgOverflowRoundingAndCountsMatchCpu) {
  const auto kMax = DecimalUtil::kLongDecimalMax;
  // Group 0/1: overflowing exact .5 ties of the same sign round away from
  // zero. 2: overflow by more than one 2^127. 3: mixed signs with an odd count.
  // 4: unequal partial counts. 5: nulls. 6: all null. 7: negative overflow.
  const std::vector<RowVectorPtr> batches = {
      makeDecimal128Batch(
          {0, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 7, 7},
          {kMax,
           -kMax,
           kMax,
           kMax,
           kMax,
           kMax,
           kMax,
           1,
           std::nullopt,
           std::nullopt,
           -kMax,
           -kMax,
           -kMax}),
      makeDecimal128Batch(
          {0, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7},
          {kMax - 1,
           -kMax + 1,
           kMax,
           3,
           -1,
           2,
           2,
           2,
           kMax,
           std::nullopt,
           -kMax}),
  };
  EXPECT_EQ(assertCompactMatrixMatchesCpu(batches, {"k"}, {"avg(d)"}), "");
  auto plan = CudfPlanRewriter::rewrite(
      exec::test::PlanBuilder()
          .values(batches)
          .partialAggregation({"k"}, {"avg(d)"})
          .finalAggregation()
          .planNode());
  exec::test::AssertQueryBuilder(plan).assertResults(makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}),
      makeNullableFlatVector<int128_t>(
          {kMax,
           -kMax,
           cpuDecimal128Average({kMax, kMax, kMax, kMax, 3}),
           cpuDecimal128Average({kMax, kMax, -1}),
           2,
           kMax,
           std::nullopt,
           -kMax},
          DECIMAL(38, 2)),
  }));

  for (const auto& values : std::vector<std::vector<int128_t>>{
           {kMax, kMax - 1}, {-kMax, -kMax + 1}, {kMax, kMax, -1}}) {
    std::vector<RowVectorPtr> global;
    for (auto value : values) {
      global.push_back(makeDecimal128Batch({0}, {value}));
    }
    EXPECT_EQ(assertCompactMatrixMatchesCpu(global, {}, {"avg(d)"}), "");
  }
}

TEST_F(CudfDecimalTest, decimal128NullAndEmptyInputsMatchCpu) {
  const auto allNull = makeDecimal128Batch(
      {0, 0, 1}, {std::nullopt, std::nullopt, std::nullopt});
  const auto mixed = makeDecimal128Batch({0, 1}, {std::nullopt, 5});
  const std::vector<std::string> aggregates{"sum(d)", "avg(d)"};
  EXPECT_EQ(assertCompactMatrixMatchesCpu({allNull}, {}, aggregates), "");
  EXPECT_EQ(
      assertCompactMatrixMatchesCpu({allNull, mixed}, {}, aggregates), "");
  EXPECT_EQ(assertCompactMatrixMatchesCpu({allNull}, {"k"}, aggregates), "");
  EXPECT_EQ(
      assertCompactMatrixMatchesCpu({allNull, mixed}, {"k"}, aggregates), "");

  for (bool grouped : {false, true}) {
    for (bool intermediate : {false, true}) {
      SCOPED_TRACE(
          fmt::format("grouped={} intermediate={}", grouped, intermediate));
      auto builder = exec::test::PlanBuilder();
      builder.values({mixed}).filter("k < 0").partialAggregation(
          grouped ? std::vector<std::string>{"k"} : std::vector<std::string>{},
          aggregates);
      if (intermediate) {
        builder.intermediateAggregation();
      }
      auto original = builder.finalAggregation().planNode();
      assertCompactMatchesCpu(original);
      exec::test::AssertQueryBuilder(CudfPlanRewriter::rewrite(original))
          .assertResults(
              grouped ? makeRowVector(
                            {makeFlatVector<int32_t>({}),
                             makeFlatVector<int128_t>({}, DECIMAL(38, 2)),
                             makeFlatVector<int128_t>({}, DECIMAL(38, 2))})
                      : makeRowVector(
                            {makeNullableFlatVector<int128_t>(
                                 {std::nullopt}, DECIMAL(38, 2)),
                             makeNullableFlatVector<int128_t>(
                                 {std::nullopt}, DECIMAL(38, 2))}));
    }
  }
}

TEST_F(CudfDecimalTest, decimal128RandomizedMatchesCpu) {
  const auto kMax = DecimalUtil::kLongDecimalMax;
  std::mt19937_64 rng(20261022);
  auto randomBelow = [&](int128_t bound) {
    const auto bits = (static_cast<unsigned __int128>(rng()) << 64) | rng();
    return static_cast<int128_t>(bits % static_cast<unsigned __int128>(bound));
  };
  constexpr int kGroups = 7;
  constexpr int kPairsPerGroup = 6;
  // Column s: each group first adds big same-signed values, then nearly
  // cancels them, so partial states overflow int128 while every final sum is
  // small. Column a: arbitrary values for AVG.
  std::vector<int32_t> keys;
  std::vector<std::optional<int128_t>> s;
  std::vector<std::optional<int128_t>> a;
  std::vector<std::vector<int128_t>> big(kGroups);
  for (int pass = 0; pass < 2; ++pass) {
    for (int i = 0; i < kPairsPerGroup; ++i) {
      for (int g = 0; g < kGroups; ++g) {
        if (pass == 0) {
          auto value = kMax / 2 + randomBelow(kMax / 2);
          big[g].push_back(g % 2 ? -value : value);
          s.push_back(big[g].back());
        } else {
          s.push_back(-big[g][i] + randomBelow(2001) - 1000);
        }
        keys.push_back(g);
        if (randomBelow(10) == 0) {
          a.push_back(std::nullopt);
        } else {
          const auto value = randomBelow(kMax);
          a.push_back(rng() % 2 ? value : -value);
        }
      }
    }
  }
  for (int g = 0; g < kGroups; ++g) {
    keys.push_back(g);
    s.push_back(std::nullopt);
    a.push_back(std::nullopt);
  }
  // With mixed signs the CPU state depends on input order. That changes AVG
  // only on exact .5 ties, which an odd count rules out.
  for (int g = 0; g < kGroups; ++g) {
    int count = 0;
    size_t last = 0;
    for (size_t row = 0; row < keys.size(); ++row) {
      if (keys[row] == g) {
        count += a[row].has_value();
        last = row;
      }
    }
    if (count % 2 == 0) {
      a[last] = a[last] ? std::nullopt : std::optional<int128_t>{kMax};
    }
  }

  std::vector<RowVectorPtr> batches;
  constexpr size_t kBatchRows = 10;
  for (size_t begin = 0; begin < keys.size(); begin += kBatchRows) {
    const auto end = std::min(begin + kBatchRows, keys.size());
    batches.push_back(makeRowVector(
        {"k", "s", "a"},
        {makeFlatVector<int32_t>(
             std::vector<int32_t>(keys.begin() + begin, keys.begin() + end)),
         makeNullableFlatVector<int128_t>(
             std::vector<std::optional<int128_t>>(
                 s.begin() + begin, s.begin() + end),
             DECIMAL(38, 4)),
         makeNullableFlatVector<int128_t>(
             std::vector<std::optional<int128_t>>(
                 a.begin() + begin, a.begin() + end),
             DECIMAL(38, 4))}));
  }
  const std::vector<std::string> aggregates{"sum(s)", "avg(a)"};
  EXPECT_EQ(assertCompactMatrixMatchesCpu(batches, {"k"}, aggregates), "");
  EXPECT_EQ(assertCompactMatrixMatchesCpu(batches, {}, aggregates), "");
}

TEST_F(CudfDecimalTest, decimal128PreservesDecimal64AndResultTypes) {
  const auto kMax = DecimalUtil::kLongDecimalMax;
  auto input = makeRowVector(
      {"k", "d", "l"},
      {makeFlatVector<int32_t>({1, 1, 2, 2, 3}),
       makeNullableFlatVector<int64_t>(
           {999'999'999'999'999'999,
            999'999'999'999'999'999,
            -5,
            6,
            std::nullopt},
           DECIMAL(18, 2)),
       makeFlatVector<int128_t>(
           {kMax, -kMax, kMax, 2 - kMax, 7}, DECIMAL(38, 3))});
  for (const auto& keys :
       {std::vector<std::string>{}, std::vector<std::string>{"k"}}) {
    SCOPED_TRACE(keys.size());
    auto original =
        exec::test::PlanBuilder()
            .values({input, input})
            .partialAggregation(keys, {"sum(d)", "avg(d)", "sum(l)", "avg(l)"})
            .localPartition(keys)
            .intermediateAggregation()
            .finalAggregation()
            .planNode();
    auto plan = CudfPlanRewriter::rewrite(original);
    auto* final = dynamic_cast<const CudfAggregationNode*>(plan.get());
    ASSERT_NE(final, nullptr);
    EXPECT_EQ(*final->intermediateType(0), *DECIMAL(38, 2));
    EXPECT_EQ(final->usesCompactDecimalState(1), !keys.empty());
    EXPECT_EQ(
        *final->intermediateType(2),
        *ROW({{"sum", DECIMAL(38, 3)}, {"overflow", BIGINT()}}));
    EXPECT_EQ(
        *final->intermediateType(3),
        *ROW(
            {{"sum", DECIMAL(38, 3)},
             {"count", BIGINT()},
             {"overflow", BIGINT()}}));
    const auto& resultType = plan->outputType();
    EXPECT_EQ(*resultType, *original->outputType());
    EXPECT_EQ(*resultType->childAt(keys.size()), *DECIMAL(38, 2));
    EXPECT_EQ(*resultType->childAt(keys.size() + 1), *DECIMAL(18, 2));
    EXPECT_EQ(*resultType->childAt(keys.size() + 2), *DECIMAL(38, 3));
    EXPECT_EQ(*resultType->childAt(keys.size() + 3), *DECIMAL(38, 3));
    auto* partition = dynamic_cast<const core::LocalPartitionNode*>(
        plan->sources()[0]->sources()[0].get());
    ASSERT_NE(partition, nullptr);
    EXPECT_EQ(
        *partition->outputType()->childAt(keys.size() + 2),
        *final->intermediateType(2));
    EXPECT_EQ(
        *partition->outputType()->childAt(keys.size() + 3),
        *final->intermediateType(3));
    assertCompactMatchesCpu(original, {.maxDrivers = 2});
  }
}

TEST_F(CudfDecimalTest, decimal128UcxFragmentSchemasAndSerde) {
  auto input = makeRowVector(
      {"k", "l"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int128_t>({100}, DECIMAL(20, 3))});
  auto producer = exec::test::PlanBuilder()
                      .values({input})
                      .partialAggregation({"k"}, {"sum(l) AS s", "avg(l) AS a"})
                      .partitionedOutput(
                          {"k"},
                          2,
                          false,
                          {},
                          "Presto",
                          std::string{core::TransportKind::kUcx})
                      .planNode();
  auto consumer = exec::test::PlanBuilder()
                      .exchange(
                          producer->outputType(),
                          "Presto",
                          std::string{core::TransportKind::kUcx})
                      .finalAggregation(
                          {"k"},
                          {"sum(s) AS s", "avg(a) AS a"},
                          {{DECIMAL(20, 3)}, {DECIMAL(20, 3)}})
                      .planNode();
  auto gpuProducer = CudfPlanRewriter::rewrite(producer);
  auto gpuConsumer = CudfPlanRewriter::rewrite(consumer);
  auto* output =
      dynamic_cast<const core::PartitionedOutputNode*>(gpuProducer.get());
  auto* exchange =
      dynamic_cast<const core::ExchangeNode*>(gpuConsumer->sources()[0].get());
  ASSERT_NE(output, nullptr);
  ASSERT_NE(exchange, nullptr);
  EXPECT_EQ(*output->outputType(), *exchange->outputType());
  EXPECT_EQ(
      *output->outputType()->childAt(1),
      *ROW({{"sum", DECIMAL(38, 3)}, {"overflow", BIGINT()}}));
  EXPECT_EQ(
      *output->outputType()->childAt(2),
      *ROW(
          {{"sum", DECIMAL(38, 3)},
           {"count", BIGINT()},
           {"overflow", BIGINT()}}));
  EXPECT_EQ(*producer->outputType()->childAt(1), *VARBINARY());
  EXPECT_EQ(*gpuConsumer->outputType(), *consumer->outputType());

  Type::registerSerDe();
  core::ITypedExpr::registerSerDe();
  core::PlanNode::registerSerDe();
  CudfAggregationNode::registerSerDe();
  auto copy = ISerializable::deserialize<core::PlanNode>(
      gpuConsumer->serialize(), pool());
  EXPECT_EQ(copy->serialize(), gpuConsumer->serialize());
  auto* aggregation = dynamic_cast<const CudfAggregationNode*>(copy.get());
  ASSERT_NE(aggregation, nullptr);
  EXPECT_EQ(
      *aggregation->intermediateType(0), *output->outputType()->childAt(1));
  EXPECT_EQ(
      *aggregation->intermediateType(1), *output->outputType()->childAt(2));
}

TEST_F(CudfDecimalTest, decimal128RejectsMismatchedStatesAndModifiers) {
  auto input = makeRowVector(
      {"k", "d", "l", "m"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>({100}, DECIMAL(20, 3)),
       makeFlatVector<bool>({true})});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto partial = [&](const std::string& aggregate) {
    return exec::test::PlanBuilder(ids)
        .values({input})
        .partialAggregation({"k"}, {aggregate + " AS s"})
        .localPartition({"k"})
        .planNode();
  };
  // A consumer declaring another scale or width than its producer.
  for (const auto& [aggregate, rawType] :
       std::vector<std::pair<std::string, TypePtr>>{
           {"sum(l)", DECIMAL(20, 2)},
           {"avg(l)", DECIMAL(20, 2)},
           {"sum(d)", DECIMAL(20, 2)},
           {"avg(d)", DECIMAL(20, 2)},
           {"sum(l)", DECIMAL(12, 3)}}) {
    SCOPED_TRACE(aggregate + " " + rawType->toString());
    const auto function = aggregate.substr(0, 3);
    auto plan = exec::test::PlanBuilder(partial(aggregate), ids)
                    .finalAggregation({"k"}, {function + "(s)"}, {{rawType}})
                    .planNode();
    VELOX_ASSERT_THROW(
        CudfPlanRewriter::rewrite(plan),
        "Incompatible compact aggregation state schemas");
  }

  auto masked = exec::test::PlanBuilder()
                    .values({input})
                    .partialAggregation({"k"}, {"sum(l)"}, {"m"})
                    .finalAggregation()
                    .planNode();
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(masked),
      "Compact Decimal128 aggregation does not support DISTINCT, ordering or masks");
  auto consumer = exec::test::PlanBuilder()
                      .exchange(partial("avg(l)")->outputType(), "Presto")
                      .finalAggregation({"k"}, {"avg(s)"}, {{DECIMAL(20, 3)}})
                      .planNode();
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(consumer),
      "Compact aggregation state requires an unordered UCX exchange");

  // A node whose declared state is not the Decimal128 layout.
  auto plan = CudfPlanRewriter::rewrite(
      exec::test::PlanBuilder()
          .values({input})
          .partialAggregation({"k"}, {"sum(l)"})
          .finalAggregation()
          .planNode());
  auto* final = dynamic_cast<const CudfAggregationNode*>(plan.get());
  ASSERT_NE(final, nullptr);
  EXPECT_THROW(
      std::make_shared<CudfAggregationNode>(
          *final, std::vector<TypePtr>{DECIMAL(38, 3)}),
      VeloxUserError);

  // State columns with the wrong layout are rejected at execution.
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(makeDecimalColumn<int128_t>({1}, 3, nullptr, stream));
  children.push_back(makeInt64Column({1}, nullptr, stream));
  auto state = cudf::make_structs_column(
      1,
      std::move(children),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
      stream,
      mr);
  decimal128StateChildren(state->view(), false, stream);
  VELOX_ASSERT_THROW(
      decimal128StateChildren(state->view(), true, stream),
      "Decimal128 AVG state requires a STRUCT with 3 children");
}

TEST_F(CudfDecimalTest, decimal128StateAssemblyAndFinalizationMatchCpu) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const auto kMax = DecimalUtil::kLongDecimalMax;
  const auto kInt128Max = std::numeric_limits<int128_t>::max();
  // Same-signed groups, whose CPU state is the canonical one.
  const std::vector<std::vector<int128_t>> groups = {
      {kInt128Max, 1},
      {-kInt128Max, -1},
      {kInt128Max, kInt128Max, kInt128Max},
      {-kInt128Max, -kInt128Max, -kInt128Max},
      {kMax, kMax, kMax, kMax, kMax},
      {-1},
      {0},
      {12345, 67890},
      {kMax},
      {-kMax, -2},
      {},
  };
  const std::vector<int64_t> extraOverflow{2, -2, 1, -1, 4, -3, 0, 1, 0, 0, 0};

  std::vector<int128_t> flat;
  for (const auto& group : groups) {
    flat.insert(flat.end(), group.begin(), group.end());
  }
  auto values = makeDecimalColumn<int128_t>(flat, 0, nullptr, stream);
  auto limbs = splitDecimal128SumLimbs(values->view(), stream, mr);
  ASSERT_EQ(limbs.size(), kDecimal128SumLimbs);
  std::vector<std::vector<int64_t>> hostLimbs;
  for (const auto& limb : limbs) {
    hostLimbs.push_back(copyColumnData<int64_t>(limb->view(), stream));
  }
  std::vector<std::vector<int64_t>> limbSums(kDecimal128SumLimbs);
  std::vector<bool> valid;
  size_t row = 0;
  for (const auto& group : groups) {
    std::vector<int64_t> sums(kDecimal128SumLimbs, 0);
    for (auto value : group) {
      const auto bits = static_cast<unsigned __int128>(value);
      for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(
            hostLimbs[i][row],
            static_cast<int64_t>(static_cast<uint32_t>(bits >> (32 * i))));
      }
      EXPECT_EQ(hostLimbs[3][row], static_cast<int32_t>(bits >> 96));
      for (size_t i = 0; i < kDecimal128SumLimbs; ++i) {
        sums[i] += hostLimbs[i][row];
      }
      ++row;
    }
    for (size_t i = 0; i < kDecimal128SumLimbs; ++i) {
      limbSums[i].push_back(sums[i]);
    }
    valid.push_back(!group.empty());
  }
  std::vector<std::unique_ptr<cudf::column>> limbColumns;
  std::vector<cudf::column_view> limbViews;
  for (const auto& sums : limbSums) {
    limbColumns.push_back(makeInt64Column(sums, &valid, stream));
    limbViews.push_back(limbColumns.back()->view());
  }
  auto extra = makeInt64Column(extraOverflow, nullptr, stream);
  for (bool withOverflow : {false, true}) {
    SCOPED_TRACE(withOverflow);
    const auto extraView = extra->view();
    auto state = assembleDecimal128SumState(
        {limbViews.data(), limbViews.size()},
        withOverflow ? &extraView : nullptr,
        0,
        stream,
        mr);
    const auto sums = copyColumnData<int128_t>(state.sum->view(), stream);
    const auto overflows =
        copyColumnData<int64_t>(state.overflow->view(), stream);
    const auto mask = copyNullMask(state.sum->view(), stream);
    for (size_t g = 0; g < groups.size(); ++g) {
      SCOPED_TRACE(g);
      EXPECT_EQ(isValidAt(mask, g), !groups[g].empty());
      if (groups[g].empty()) {
        EXPECT_EQ(overflows[g], 0);
        continue;
      }
      auto [cpuSum, cpuOverflow] = cpuDecimal128State(groups[g]);
      EXPECT_EQ(sums[g], cpuSum);
      EXPECT_EQ(
          overflows[g], cpuOverflow + (withOverflow ? extraOverflow[g] : 0));

      // SUM finalization fails exactly where the CPU's does, with its message.
      auto sumColumn =
          makeDecimalColumn<int128_t>({sums[g]}, 0, nullptr, stream);
      auto overflowColumn = makeInt64Column({overflows[g]}, nullptr, stream);
      std::string cpuError;
      int128_t cpuValue = 0;
      try {
        auto adjusted =
            DecimalUtil::adjustSumForOverflow(sums[g], overflows[g]);
        VELOX_USER_CHECK(adjusted.has_value(), "Decimal overflow");
        DecimalUtil::valueInRange(adjusted.value());
        cpuValue = adjusted.value();
      } catch (const VeloxUserError& e) {
        cpuError = e.message();
      }
      try {
        auto result = finalizeDecimal128Sum(
            sumColumn->view(), overflowColumn->view(), stream, mr);
        EXPECT_EQ(cpuError, "");
        EXPECT_EQ(
            copyColumnData<int128_t>(result->view(), stream)[0], cpuValue);
      } catch (const VeloxUserError& e) {
        EXPECT_EQ(e.message(), cpuError);
      }
    }
  }

  // The AVG kernel is DecimalUtil::computeAverage, including for states that
  // are not canonical.
  const std::vector<std::tuple<int128_t, int64_t, int64_t>> averages = {
      {kInt128Max, 3, 1},
      {-kInt128Max, 3, -1},
      {-5, 2, 1},
      {5, 2, -1},
      {1, 2, 0},
      {-1, 2, 0},
      {3, 2, 0},
      {-3, 2, 0},
      {0, 4, 2},
      {0, 4, -2},
      {kMax, 7, 3},
      {-kMax, 7, -3},
      {123'456'789, 1'000'000'007, 5},
      {-123'456'789, 1'000'000'007, -5},
      {0, 0, 0},
  };
  std::vector<int128_t> averageSums;
  std::vector<int64_t> counts;
  std::vector<int64_t> overflows;
  for (const auto& [sum, count, overflow] : averages) {
    averageSums.push_back(sum);
    counts.push_back(count);
    overflows.push_back(overflow);
  }
  auto sumColumn = makeDecimalColumn<int128_t>(averageSums, 0, nullptr, stream);
  auto countColumn = makeInt64Column(counts, nullptr, stream);
  auto overflowColumn = makeInt64Column(overflows, nullptr, stream);
  auto result = finalizeDecimal128Average(
      sumColumn->view(),
      countColumn->view(),
      overflowColumn->view(),
      stream,
      mr);
  const auto actual = copyColumnData<int128_t>(result->view(), stream);
  const auto mask = copyNullMask(result->view(), stream);
  for (size_t i = 0; i < averages.size(); ++i) {
    SCOPED_TRACE(i);
    const auto& [sum, count, overflow] = averages[i];
    EXPECT_EQ(isValidAt(mask, i), count != 0);
    if (count != 0) {
      int128_t expected = 0;
      DecimalUtil::computeAverage(expected, sum, count, overflow);
      EXPECT_EQ(actual[i], expected);
    }
  }
}

TEST_F(CudfDecimalTest, reduceDecimal64SumCountSlicesAndNulls) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const std::vector<int64_t> values = {999, 100, 999, -100, 999};
  const std::vector<bool> valid = {false, true, false, true, false};
  auto input = makeDecimalColumn<int64_t>(values, 2, &valid, stream);
  auto slices = cudf::slice(input->view(), {1, 4, 2, 3, 0, 0}, stream);
  const std::vector<int64_t> expectedCounts = {2, 0, 0};
  for (size_t i = 0; i < slices.size(); ++i) {
    SCOPED_TRACE(i);
    auto result = reduceDecimal64SumCount(slices[i], stream, mr);
    EXPECT_EQ(
        result.sum->type(), cudf::data_type(cudf::type_id::DECIMAL128, -2));
    EXPECT_EQ(copyColumnData<int128_t>(result.sum->view(), stream)[0], 0);
    EXPECT_EQ(
        copyColumnData<int64_t>(result.count->view(), stream)[0],
        expectedCounts[i]);
    EXPECT_EQ(result.sum->null_count(), expectedCounts[i] == 0 ? 1 : 0);
    EXPECT_EQ(result.count->null_count(), 0);
  }

  constexpr int64_t kBig = 900'000'000'000'000'000;
  auto nonNullInput = makeDecimalColumn<int64_t>(
      std::vector<int64_t>(12, kBig), 2, nullptr, stream);
  auto result = reduceDecimal64SumCount(nonNullInput->view(), stream, mr);
  EXPECT_EQ(
      copyColumnData<int128_t>(result.sum->view(), stream)[0],
      static_cast<int128_t>(kBig) * 12);
  EXPECT_EQ(copyColumnData<int64_t>(result.count->view(), stream)[0], 12);
  EXPECT_EQ(result.sum->null_count(), 0);
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateDecimal64) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, -200, 300};
  std::vector<int64_t> counts = {1, 2, 0};
  std::vector<bool> sumValid = {true, false, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto sumAndCount =
      deserializeDecimalSumState(stateCol->view(), 2, stream, mr);
  auto stateMask = copyNullMask(stateCol->view(), stream);
  auto sumMask = copyNullMask(sumAndCount.sum->view(), stream);
  EXPECT_EQ(stateMask, sumMask);

  auto outSum = copyColumnData<__int128_t>(sumAndCount.sum->view(), stream);
  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(sumMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateDecimal128) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {2, 1, 0};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, false, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto sumAndCount =
      deserializeDecimalSumState(stateCol->view(), 3, stream, mr);
  auto stateMask = copyNullMask(stateCol->view(), stream);
  auto sumMask = copyNullMask(sumAndCount.sum->view(), stream);
  EXPECT_EQ(stateMask, sumMask);

  auto outSum = copyColumnData<__int128_t>(sumAndCount.sum->view(), stream);
  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(sumMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], sums[i]);
    }
  }
}

// Reproduces the Q18 failure scenario: serializes a decimal sum state with
// partial nulls, round-trips through Arrow (which compacts null rows to 0-byte
// payloads), then deserializes. Without the null-count fix in
// deserializeDecimalSumState, the payload size check would fire because
// chars_size == (numRows - nullCount) * 32, not numRows * 32.
TEST_F(CudfDecimalTest, decimalDeserializeSumStatePartialNullCompact) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // Serialize a 3-row state where row 1 is null (sum-null, count == 0).
  std::vector<int64_t> sums = {100, 0, 300};
  std::vector<int64_t> counts = {1, 0, 2};
  std::vector<bool> sumValid = {true, false, true};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  // Round-trip through Arrow (cuDF -> Velox VARBINARY -> cuDF STRING).
  // Arrow stores null rows with 0-byte payloads, so the resulting cuDF STRING
  // column has chars_size == (numRows - nullCount) * 32.
  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);

  // Verify the column now has the compact layout.
  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(
      strings.chars_size(stream),
      static_cast<int64_t>(sums.size()) * 32); // 32 == kDecimalSumStateSize

  auto result = deserializeDecimalSumState(compactStateView, 2, stream, mr);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_TRUE(isValidAt(outMask, 0));
  EXPECT_FALSE(isValidAt(outMask, 1));
  EXPECT_TRUE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(100));
  EXPECT_EQ(outCount[0], 1);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(300));
  EXPECT_EQ(outCount[2], 2);
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateUncompactedSlice) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  std::vector<int64_t> sums = {10, 20, 0, 40, 50};
  std::vector<int64_t> counts = {1, 2, 0, 4, 5};
  std::vector<bool> sumValid = {true, true, false, true, true};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  cudf::strings_column_view strings(stateCol->view());
  EXPECT_EQ(
      strings.chars_size(stream),
      static_cast<int64_t>(sums.size()) * 32); // 32 == kDecimalSumStateSize

  auto slices = cudf::slice(stateCol->view(), {1, 4});
  ASSERT_EQ(slices.size(), 1);
  ASSERT_EQ(slices.front().offset(), 1);

  auto result = deserializeDecimalSumState(slices.front(), 2, stream, mr);
  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  ASSERT_EQ(outSum.size(), 3);
  EXPECT_TRUE(isValidAt(outMask, 0));
  EXPECT_FALSE(isValidAt(outMask, 1));
  EXPECT_TRUE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(20));
  EXPECT_EQ(outCount[0], 2);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(40));
  EXPECT_EQ(outCount[2], 4);
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateArrowCompactedSlice) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // Slice the sum/count inputs first to pin serialization offset handling.
  // Round-trip through Arrow so the null row has a 0-byte payload, then slice
  // the state so deserialization must shift both offsets and validity bits.
  std::vector<int64_t> sums = {-999, 10, 20, 0, 40, 50, 999};
  std::vector<int64_t> counts = {9, 1, 2, 0, 4, 5, 9};
  std::vector<bool> sumValid = {true, true, true, false, true, true, true};
  auto sumParent = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countParent = makeInt64Column(counts, nullptr, stream);
  auto sumSlices = cudf::slice(sumParent->view(), {1, 6});
  auto countSlices = cudf::slice(countParent->view(), {1, 6});
  ASSERT_EQ(sumSlices.size(), 1);
  ASSERT_EQ(countSlices.size(), 1);
  ASSERT_EQ(sumSlices.front().offset(), 1);
  ASSERT_EQ(countSlices.front().offset(), 1);
  auto stateCol = serializeDecimalSumState(
      sumSlices.front(), countSlices.front(), stream, mr);

  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);
  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(
      strings.chars_size(stream), static_cast<int64_t>(stateCol->size()) * 32);

  auto slices = cudf::slice(compactStateView, {1, 4});
  ASSERT_EQ(slices.size(), 1);
  ASSERT_EQ(slices.front().offset(), 1);

  auto result = deserializeDecimalSumState(slices.front(), 2, stream, mr);
  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  ASSERT_EQ(outSum.size(), 3);
  EXPECT_TRUE(isValidAt(outMask, 0));
  EXPECT_FALSE(isValidAt(outMask, 1));
  EXPECT_TRUE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(20));
  EXPECT_EQ(outCount[0], 2);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(40));
  EXPECT_EQ(outCount[2], 4);
}

// Reproduces the TPC-DS Q1 failure in CudfGroupbyFINAL: the streaming final
// aggregation concatenates its buffered result -- which came straight from
// serializeDecimalSumState and so keeps a 32-byte payload for null rows -- with
// each newly arrived batch, which was round tripped through velox and so has
// its null rows compacted to 0 bytes. The concatenated state column mixes both
// encodings, so its payload size is neither numRows * 32 nor
// (numRows - nullCount) * 32 and a two-way equality check rejects it.
TEST_F(CudfDecimalTest, decimalDeserializeSumStateMixedNullEncodings) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  auto serialize = [&](const std::vector<int64_t>& sums,
                       const std::vector<int64_t>& counts,
                       const std::vector<bool>& sumValid) {
    auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
    auto countCol = makeInt64Column(counts, nullptr, stream);
    return serializeDecimalSumState(
        sumCol->view(), countCol->view(), stream, mr);
  };

  // Buffered side: row 1 is null and keeps its 32-byte payload.
  auto fullState = serialize({100, 0, 300}, {1, 0, 2}, {true, false, true});

  // Incoming side: same shape, but the velox round trip compacts its null row
  // to a 0-byte payload.
  auto incomingState = serialize({400, 0, 600}, {3, 0, 4}, {true, false, true});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{incomingState->view()}},
      pool(),
      ROW({{"s", VARBINARY()}}),
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);

  auto mixed = cudf::concatenate(
      std::vector<cudf::column_view>{
          fullState->view(), compactTable->view().column(0)},
      stream,
      mr);

  // Four non-null rows, plus the one null payload the buffered side kept: the
  // payload size sits strictly between the compact and full extremes.
  cudf::strings_column_view mixedStrings(mixed->view());
  auto const numRows = static_cast<int64_t>(mixed->size());
  auto const nullCount = static_cast<int64_t>(mixed->null_count());
  EXPECT_EQ(numRows, 6);
  EXPECT_EQ(nullCount, 2);
  EXPECT_GT(
      mixedStrings.chars_size(stream),
      (numRows - nullCount) * 32); // 32 == kDecimalSumStateSize
  EXPECT_LT(mixedStrings.chars_size(stream), numRows * 32);

  auto result = deserializeDecimalSumState(mixed->view(), 2, stream, mr);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_FALSE(isValidAt(outMask, 1));
  EXPECT_FALSE(isValidAt(outMask, 4));
  for (auto row : {0, 2, 3, 5}) {
    EXPECT_TRUE(isValidAt(outMask, row));
  }
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(100));
  EXPECT_EQ(outCount[0], 1);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(300));
  EXPECT_EQ(outCount[2], 2);
  EXPECT_EQ(outSum[3], static_cast<__int128_t>(400));
  EXPECT_EQ(outCount[3], 3);
  EXPECT_EQ(outSum[5], static_cast<__int128_t>(600));
  EXPECT_EQ(outCount[5], 4);
}

// Trailing null: the offset for the last row equals chars_size, so the kernel
// would read 32 bytes past the buffer end without the null-mask guard.
TEST_F(CudfDecimalTest, decimalDeserializeSumStateTrailingNullCompact) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // 3-row state: [valid, valid, null].
  std::vector<int64_t> sums = {100, 200, 0};
  std::vector<int64_t> counts = {1, 2, 0};
  std::vector<bool> sumValid = {true, true, false};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);

  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(strings.chars_size(stream), static_cast<int64_t>(sums.size()) * 32);

  auto result = deserializeDecimalSumState(compactStateView, 2, stream, mr);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_TRUE(isValidAt(outMask, 0));
  EXPECT_TRUE(isValidAt(outMask, 1));
  EXPECT_FALSE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(100));
  EXPECT_EQ(outCount[0], 1);
  EXPECT_EQ(outSum[1], static_cast<__int128_t>(200));
  EXPECT_EQ(outCount[1], 2);
}

// Leading null: offset 0 overlaps the next valid row's data so the read is
// in-bounds, but verify the null mask propagates correctly.
TEST_F(CudfDecimalTest, decimalDeserializeSumStateLeadingNullCompact) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // 3-row state: [null, valid, valid].
  std::vector<int64_t> sums = {0, 200, 300};
  std::vector<int64_t> counts = {0, 2, 3};
  std::vector<bool> sumValid = {false, true, true};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);

  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(strings.chars_size(stream), static_cast<int64_t>(sums.size()) * 32);

  auto result = deserializeDecimalSumState(compactStateView, 2, stream, mr);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_FALSE(isValidAt(outMask, 0));
  EXPECT_TRUE(isValidAt(outMask, 1));
  EXPECT_TRUE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[1], static_cast<__int128_t>(200));
  EXPECT_EQ(outCount[1], 2);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(300));
  EXPECT_EQ(outCount[2], 3);
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateAllNull) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  constexpr cudf::size_type numRows = 4;

  auto offsetsCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32},
      numRows + 1,
      cudf::mask_state::UNALLOCATED,
      stream);
  auto* offsetsPtr = offsetsCol->mutable_view().data<int32_t>();
  auto status = cudaMemsetAsync(
      offsetsPtr,
      0,
      static_cast<size_t>(numRows + 1) * sizeof(int32_t),
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();

  std::vector<bool> valid(numRows, false);
  auto [nullMask, nullCount] = makeNullMask(valid, stream);
  rmm::device_buffer charsBuf(0, stream);
  auto stateCol = cudf::make_strings_column(
      numRows,
      std::move(offsetsCol),
      std::move(charsBuf),
      nullCount,
      std::move(nullMask));

  auto sumAndCount =
      deserializeDecimalSumState(stateCol->view(), 2, stream, mr);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();

  EXPECT_EQ(outSumView.size(), numRows);
  EXPECT_EQ(outCountView.size(), numRows);
  EXPECT_EQ(outSumView.null_count(), numRows);
  EXPECT_EQ(outCountView.null_count(), numRows);

  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);
  for (size_t i = 0; i < static_cast<size_t>(numRows); ++i) {
    EXPECT_FALSE(isValidAt(outSumMask, i));
    EXPECT_FALSE(isValidAt(outCountMask, i));
  }
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateRejectsInvalidRowWidth) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // The total payload size is valid, but neither row contains a complete
  // serialized state. This exercises the per-row check rather than the
  // aggregate payload-size check.
  auto offsetsCol = makeFixedWidthColumn<int32_t>(
      cudf::data_type{cudf::type_id::INT32}, {0, 16, 64}, nullptr, stream);
  rmm::device_buffer charsBuf(64, stream);
  auto stateCol = cudf::make_strings_column(
      2,
      std::move(offsetsCol),
      std::move(charsBuf),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));

  VELOX_ASSERT_THROW(
      deserializeDecimalSumState(stateCol->view(), 2, stream, mr),
      "Decimal sum state requires every non-null row to be 32 bytes");
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateRejectsMisalignedRow) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  auto sumCol = makeDecimalColumn<int64_t>({200}, 2, nullptr, stream);
  auto countCol = makeInt64Column({2}, nullptr, stream);
  auto alignedState =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  cudf::strings_column_view alignedStrings(alignedState->view());

  // The only non-null row has the correct width, but starts four bytes into
  // the chars buffer. The aggregate payload checks still accept the layout:
  // 32 non-null bytes in a 64-byte payload across three rows.
  auto offsetsCol = makeFixedWidthColumn<int32_t>(
      cudf::data_type{cudf::type_id::INT32}, {0, 4, 36, 64}, nullptr, stream);
  rmm::device_buffer charsBuf(64, stream);
  auto status = cudaMemcpyAsync(
      static_cast<uint8_t*>(charsBuf.data()) + 4,
      alignedStrings.chars_begin(stream),
      32, // kDecimalSumStateSize
      cudaMemcpyDeviceToDevice,
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  std::vector<bool> valid = {false, true, false};
  auto [nullMask, nullCount] = makeNullMask(valid, stream);
  auto stateCol = cudf::make_strings_column(
      3,
      std::move(offsetsCol),
      std::move(charsBuf),
      nullCount,
      std::move(nullMask));

  VELOX_ASSERT_THROW(
      deserializeDecimalSumState(stateCol->view(), 2, stream, mr),
      "Decimal sum state requires every non-null row to be 32 bytes and 8-byte aligned");
}

TEST_F(CudfDecimalTest, decimalSerializeSumStateUsesInt64OffsetsWhenEnabled) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  ScopedEnvVar enableLargeStrings("LIBCUDF_LARGE_STRINGS_ENABLED", "1");
  ScopedEnvVar threshold("LIBCUDF_LARGE_STRINGS_THRESHOLD", "1");

  std::vector<int64_t> sums = {100, -200};
  std::vector<int64_t> counts = {1, 1};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, nullptr, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  cudf::strings_column_view strings(stateCol->view());
  EXPECT_EQ(strings.offsets().type().id(), cudf::type_id::INT64);
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripUsesInt64Offsets) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  ScopedEnvVar enableLargeStrings("LIBCUDF_LARGE_STRINGS_ENABLED", "1");
  ScopedEnvVar threshold("LIBCUDF_LARGE_STRINGS_THRESHOLD", "1");

  std::vector<int64_t> sums = {100, -200, 300, 400};
  std::vector<int64_t> counts = {1, 0, 2, 3};
  std::vector<bool> sumValid = {true, true, false, true};
  std::vector<bool> countValid = {true, true, true, false};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  cudf::strings_column_view strings(stateCol->view());
  EXPECT_EQ(strings.offsets().type().id(), cudf::type_id::INT64);

  auto sumAndCount =
      deserializeDecimalSumState(stateCol->view(), 2, stream, mr);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(outSumView.size(), sums.size());
  EXPECT_EQ(outCountView.size(), counts.size());
  EXPECT_EQ(outSumMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(outSumMask, i), expectedValid);
    EXPECT_EQ(isValidAt(outCountMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, 105, 250, -125};
  std::vector<int64_t> counts = {4, 2, 0, 2};
  std::vector<bool> sumValid = {true, true, true, true};
  std::vector<bool> countValid = {true, false, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(avgMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64Slice) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> parentSums = {999, 100, 105, 250, -125, -999};
  std::vector<int64_t> parentCounts = {9, 4, 2, 0, 2, 9};
  std::vector<bool> parentSumValid = {true, true, true, true, true, true};
  std::vector<bool> parentCountValid = {true, true, false, true, true, true};

  auto sumParent =
      makeDecimalColumn<int64_t>(parentSums, 2, &parentSumValid, stream);
  auto countParent = makeInt64Column(parentCounts, &parentCountValid, stream);
  auto sumSlices = cudf::slice(sumParent->view(), {1, 5});
  auto countSlices = cudf::slice(countParent->view(), {1, 5});
  ASSERT_EQ(sumSlices.size(), 1);
  ASSERT_EQ(countSlices.size(), 1);
  ASSERT_EQ(sumSlices.front().offset(), 1);
  ASSERT_EQ(countSlices.front().offset(), 1);

  auto avgCol =
      computeDecimalAverage(sumSlices.front(), countSlices.front(), stream, mr);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  std::vector<int64_t> sums = {100, 105, 250, -125};
  std::vector<int64_t> counts = {4, 2, 0, 2};
  std::vector<bool> countValid = {true, false, true, true};
  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  ASSERT_EQ(outAvg.size(), sums.size());
  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(avgMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {3, 2, 0};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(avgMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64MostNegativeSum) {
  // Negating INT64_MIN in the signed type is overflow UB; the magnitude and
  // sign must be handled in the unsigned domain. avg of one INT64_MIN is
  // itself.
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  std::vector<int64_t> sums = {kMin};
  std::vector<int64_t> counts = {1};
  std::vector<bool> valid = {true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 0, &valid, stream);
  auto countCol = makeInt64Column(counts, &valid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);
  EXPECT_EQ(outAvg[0], kMin);
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128MostNegativeSum) {
  // Same regression at the __int128 boundary. avg of one -2^127 is itself.
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const __int128_t kMin =
      static_cast<__int128_t>(static_cast<unsigned __int128>(1) << 127);
  std::vector<__int128_t> sums = {kMin};
  std::vector<int64_t> counts = {1};
  std::vector<bool> valid = {true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 0, &valid, stream);
  auto countCol = makeInt64Column(counts, &valid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);
  EXPECT_EQ(outAvg[0], kMin);
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64AllValid) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, 200, -150};
  std::vector<int64_t> counts = {4, 5, 3};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128AllValid) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(90000),
      static_cast<__int128_t>(-5000),
      static_cast<__int128_t>(1),
  };
  std::vector<int64_t> counts = {3, 5, 1};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64NonNullableInputs) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {80, -40, 1000};
  std::vector<int64_t> counts = {2, 4, 10};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, nullptr, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  ASSERT_FALSE(sumCol->nullable());
  ASSERT_FALSE(countCol->nullable());

  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128NonNullableInputs) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(600),
      static_cast<__int128_t>(-99),
  };
  std::vector<int64_t> counts = {3, 9};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 4, nullptr, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  ASSERT_FALSE(sumCol->nullable());
  ASSERT_FALSE(countCol->nullable());

  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripDecimal64) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, -200, 300, 400};
  std::vector<int64_t> counts = {1, 0, 2, 3};
  std::vector<bool> sumValid = {true, true, false, true};
  std::vector<bool> countValid = {true, true, true, false};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto stateMask = copyNullMask(stateCol->view(), stream);

  auto sumAndCount =
      deserializeDecimalSumState(stateCol->view(), 2, stream, mr);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(stateMask, outSumMask);
  EXPECT_EQ(stateMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(stateMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripDecimal128) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {2, 1, 0};
  std::vector<bool> sumValid = {true, false, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto stateMask = copyNullMask(stateCol->view(), stream);

  auto sumAndCount =
      deserializeDecimalSumState(stateCol->view(), 3, stream, mr);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(stateMask, outSumMask);
  EXPECT_EQ(stateMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(stateMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], sums[i]);
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, cudfVarbinaryArrowRoundTrip) {
  auto input = makeRowVector(
      {"bin"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto roundTrip =
      with_arrow::toVeloxColumn(cudfTable->view(), pool(), "rt_", stream, mr);

  ASSERT_EQ(roundTrip->childAt(0)->type()->kind(), TypeKind::VARCHAR);
  VELOX_ASSERT_THROW(
      roundTrip->setType(ROW({{"rt_0", VARBINARY()}})),
      "Cannot change vector type");

  auto expected = makeRowVector(
      {"rt_0"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")}, VARCHAR())});

  facebook::velox::test::assertEqualVectors(expected, roundTrip);
}

TEST_F(CudfDecimalTest, cudfVarbinaryArrowRoundTripWithExpectedType) {
  auto input = makeRowVector(
      {"bin"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  auto expectedType = ROW({{"bin", VARBINARY()}});

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto roundTrip = with_arrow::toVeloxColumn(
      cudfTable->view(), pool(), expectedType, "rt_", stream, mr);

  ASSERT_EQ(roundTrip->childAt(0)->type()->kind(), TypeKind::VARBINARY);

  auto expected = makeRowVector(
      {"rt_0"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  facebook::velox::test::assertEqualVectors(expected, roundTrip);
}

TEST_F(CudfDecimalTest, cudfVarbinaryRowTypeMismatch) {
  auto input = makeRowVector(
      {"l_returnflag",
       "l_linestatus",
       "avg_51",
       "avg_52",
       "avg_53",
       "count_54",
       "sum_47",
       "sum_48",
       "sum_49",
       "sum_50"},
      {makeFlatVector<std::string>({"A", "B"}, VARCHAR()),
       makeFlatVector<std::string>({"F", "O"}, VARCHAR()),
       makeFlatVector<std::string>({"x", "y"}, VARBINARY()),
       makeFlatVector<std::string>({"p", "q"}, VARBINARY()),
       makeFlatVector<std::string>({"m", "n"}, VARBINARY()),
       makeFlatVector<int64_t>({10, 20}, BIGINT()),
       makeFlatVector<std::string>({"u", "v"}, VARBINARY()),
       makeFlatVector<std::string>({"r", "s"}, VARBINARY()),
       makeFlatVector<std::string>({"t", "w"}, VARBINARY()),
       makeFlatVector<std::string>({"c", "d"}, VARBINARY())});

  auto expectedType = ROW({
      {"l_returnflag", VARCHAR()},
      {"l_linestatus", VARCHAR()},
      {"avg_51", VARBINARY()},
      {"avg_52", VARBINARY()},
      {"avg_53", VARBINARY()},
      {"count_54", BIGINT()},
      {"sum_47", VARBINARY()},
      {"sum_48", VARBINARY()},
      {"sum_49", VARBINARY()},
      {"sum_50", VARBINARY()},
  });

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto roundTrip =
      with_arrow::toVeloxColumn(cudfTable->view(), pool(), "rt_", stream, mr);

  ASSERT_EQ(roundTrip->childAt(2)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(3)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(4)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(6)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(7)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(8)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(9)->type()->kind(), TypeKind::VARCHAR);

  VELOX_ASSERT_THROW(
      roundTrip->setType(expectedType), "Cannot change vector type");
}

} // namespace
} // namespace facebook::velox::cudf_velox
