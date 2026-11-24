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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>

namespace facebook::velox::cudf_velox {

/// Decimal SUM state. The exact value of a row is sum + overflow * 2^127, the
/// CPU LongDecimalWithOverflowState contract, so a DECIMAL128 sum may wrap and
/// later cancel.
struct DecimalSumStateColumns {
  std::unique_ptr<cudf::column> sum;
  std::unique_ptr<cudf::column> count;
  // INT64 net int128 overflow count, or nullptr when the sum cannot overflow.
  std::unique_ptr<cudf::column> overflow;
};

/** Directly reduces DECIMAL64 input into one DECIMAL128 sum and INT64 count. */
DecimalSumStateColumns reduceDecimal64SumCount(
    const cudf::column_view& input,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Decodes intermediate decimal SUM aggregate state stored as a cuDF STRING
 * column (fixed-size packed bytes per row, converted from Velox VARBINARY) into
 * three device columns: a DECIMAL128 sum with scale -scale (matching Velox
 * intermediate state), an INT64 partial row count and an INT64 net int128
 * overflow count. Handles empty input, all-null state without touching payload
 * buffers, and propagates the source null mask to all outputs when present.
 *
 * @param stateCol STRING column of packed sum/count/overflow payloads.
 * @param scale decimal scale used to set the output sum column's scale to
 *        -scale.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return decoded sum, count and overflow columns.
 */
DecimalSumStateColumns deserializeDecimalSumState(
    const cudf::column_view& stateCol,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Encodes partial decimal SUM state (DECIMAL64 or DECIMAL128 sums plus INT64
 * counts and overflow) into a single STRING column (later converted to Velox
 * VARBINARY): per-row fixed-width payloads and string offsets (INT32 or INT64
 * depending on total char size and cuDF large-strings settings). The output
 * null mask matches buildStateValidityMask: a row is invalid if the sum or
 * count is null, or the count is zero.
 *
 * @param sumCol per-row partial sums (DECIMAL64 or DECIMAL128).
 * @param countCol per-row INT64 partial row counts.
 * @param overflowCol per-row INT64 net int128 overflow counts, or nullptr to
 *        serialize zero overflow.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return STRING column of serialized state.
 */
std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    const cudf::column_view* overflowCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Finalizes AVG from intermediate SUM state: divides each sum (plus its
 * overflow * 2^127) by its count on device with decimal-specific rounding (see
 * averageRoundDecimalSum), producing a column of the same decimal type as the
 * sum. Rows are null where buildStateValidityMask marks them invalid (null
 * sum/count or zero count), matching serializeDecimalSumState.
 *
 * @param sumCol per-row partial sums (DECIMAL64 or DECIMAL128).
 * @param countCol per-row INT64 partial row counts.
 * @param overflowCol per-row INT64 net int128 overflow counts, or nullptr when
 *        there is none. Requires DECIMAL128 sums.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return per-row decimal average column.
 */
std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    const cudf::column_view* overflowCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Converts the result of cuDF SUM_OVERFLOW with sum_overflow_output::COUNT
 * into decimal SUM state. The result is STRUCT{sum DECIMAL128, overflow INT64}
 * with exact value overflow * 2^128 + sum; the state holds the same value as
 * sum + overflow * 2^127 (see DecimalSumStateColumns). A null result row, which
 * is a group without non-null input, gives a null sum.
 *
 * @param sumOverflow SUM_OVERFLOW(COUNT) result column.
 * @param stateOverflowSum per-row SUM of the merged states' overflow, or
 *        nullptr for raw input; null rows count as zero.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for the outputs.
 * @return sum and overflow columns; count is unset.
 */
DecimalSumStateColumns makeDecimalSumOverflowState(
    std::unique_ptr<cudf::column> sumOverflow,
    const cudf::column_view* stateOverflowSum,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Converts the result of cuDF SUM_OVERFLOW with sum_overflow_output::COUNT
 * over DECIMAL64 input, STRUCT{sum DECIMAL64, overflow INT64} with exact value
 * overflow * 2^64 + sum, into a DECIMAL128 sum at the same scale. A null
 * result row, which is a group without non-null input, gives a null sum.
 *
 * @param sumOverflow SUM_OVERFLOW(COUNT) result column.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for the output.
 * @return DECIMAL128 sums.
 */
std::unique_ptr<cudf::column> widenDecimal64SumOverflow(
    std::unique_ptr<cudf::column> sumOverflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Finalizes DECIMAL128 SUM state like the CPU DecimalSumAggregate: throws the
 * same VeloxUserError when a row's sum + overflow * 2^127 does not fit
 * DECIMAL(38).
 *
 * @param sumCol DECIMAL128 state sums; null rows produce null.
 * @param overflowCol INT64 net int128 overflow counts.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for the output.
 * @return DECIMAL128 sums with the scale of sumCol.
 */
std::unique_ptr<cudf::column> adjustDecimalSumForOverflow(
    const cudf::column_view& sumCol,
    const cudf::column_view& overflowCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

// Native DECIMAL128 SUM state is ROW(sum DECIMAL(38, s), overflow BIGINT) and
// AVG state is ROW(sum DECIMAL(38, s), count BIGINT, overflow BIGINT). The
// value of a state is sum + overflow * 2^127, the CPU
// LongDecimalWithOverflowState contract, so intermediate states may exceed
// DECIMAL(38) or int128 and later cancel. The parent is always valid; sum is
// null for a group without non-null input; count and overflow are never null.
inline constexpr cudf::size_type kDecimal128StateSumChild = 0;
inline constexpr cudf::size_type kDecimal128StateCountChild = 1;

inline constexpr cudf::size_type decimal128StateOverflowChild(bool average) {
  return average ? 2 : 1;
}

struct Decimal128StateChildren {
  cudf::column_view sum;
  // Set for AVG state only.
  std::optional<cudf::column_view> count;
  cudf::column_view overflow;
};

/**
 * Validates a native DECIMAL128 SUM/AVG state column and returns its sliced
 * children.
 *
 * @param state STRUCT state column.
 * @param average true for the AVG layout, false for SUM.
 * @param stream CUDA stream for device work.
 */
Decimal128StateChildren decimal128StateChildren(
    const cudf::column_view& state,
    bool average,
    cuda::stream_ref stream);

/**
 * Builds a native DECIMAL128 SUM/AVG state column from its children.
 *
 * @param sum DECIMAL128 sums.
 * @param count INT64 counts for AVG, nullptr for SUM.
 * @param overflow INT64 overflow counts.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for the struct.
 */
std::unique_ptr<cudf::column> makeDecimal128StateColumn(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    std::unique_ptr<cudf::column> overflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
