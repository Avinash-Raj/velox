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

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/buffer>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace facebook::velox::cudf_velox::detail {

// Size in bytes of each row's packed decimal SUM intermediate state in the
// strings payload (count, net int128 overflow, and 128-bit sum split into
// words).
constexpr size_t kDecimalSumStateSize = 32;

/**
 * Reduces a DECIMAL64 column directly into one DECIMAL128 sum and one INT64
 * non-null count without materializing a DECIMAL128 input column.
 */
void reduceDecimal64SumCount(
    cudf::column_view input,
    cudf::mutable_column_view sum,
    cudf::mutable_column_view count,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Writes strings-style prefix offsets: offset[i] == i * kDecimalSumStateSize.
 *
 * @param offsetType INT32 or INT64; selects offset storage width via
 *        cudf::type_dispatcher.
 * @param offsetsView output offsets column of numRows + 1 elements.
 * @param numRows number of payload rows.
 * @param stream CUDA stream for the launch.
 */
void fillOffsetsForDecimalSumState(
    cudf::type_id offsetType,
    cudf::mutable_column_view offsetsView,
    cudf::size_type numRows,
    cuda::stream_ref stream);

/**
 * Encodes each row's partial sum, count and overflow into the fixed-width
 * device layout used for VARBINARY interchange.
 *
 * @param sumType DECIMAL64 or DECIMAL128; selects sum storage width.
 * @param offsetType INT32 or INT64; selects offset storage width. sumType and
 *        offsetType are dispatched via cudf::double_type_dispatcher.
 * @param sumCol per-row sums.
 * @param counts per-row int64 counts.
 * @param overflows per-row net int128 overflow counts, or nullptr to write
 *        zero.
 * @param offsetsView per-row byte offsets into chars.
 * @param chars output payload buffer.
 * @param numRows number of rows.
 * @param stream CUDA stream for the launch.
 */
void packDecimalSumState(
    cudf::type_id sumType,
    cudf::type_id offsetType,
    cudf::column_view sumCol,
    const int64_t* counts,
    const int64_t* overflows,
    cudf::column_view offsetsView,
    uint8_t* chars,
    cudf::size_type numRows,
    cuda::stream_ref stream);

/**
 * Decodes each non-null row of a serialized decimal SUM state STRING column.
 *
 * @param stateCol logical STRING view of packed states. Sliced views are
 *        supported; their row offset is applied to payload and null-mask
 *        access by cudf::column_device_view.
 * @param sumView output per-row DECIMAL128 sums, sized to stateCol.
 * @param countView output per-row counts, sized to stateCol.
 * @param overflowView output per-row net int128 overflow counts, sized to
 *        stateCol.
 * @param stream CUDA stream for the launch.
 * @return true if every non-null row contains a valid serialized state.
 */
bool unpackDecimalSumState(
    cudf::column_view stateCol,
    cudf::mutable_column_view sumView,
    cudf::mutable_column_view countView,
    cudf::mutable_column_view overflowView,
    cuda::stream_ref stream);

/**
 * Per-row half-up integer divide of sum + overflow * 2^127 by count, matching
 * DecimalUtil::computeAverage; count == 0 writes zero (validity is applied
 * separately).
 *
 * @param sumType DECIMAL64 or DECIMAL128; selects sum storage width via
 *        cudf::type_dispatcher<cudf::dispatch_storage_type>.
 * @param sumCol per-row sums.
 * @param counts per-row counts.
 * @param overflows per-row net int128 overflow counts, or nullptr when there
 *        is none. Requires DECIMAL128 sums.
 * @param outView output per-row averages.
 * @param numRows number of rows.
 * @param stream CUDA stream for the launch.
 */
void averageRoundDecimalSum(
    cudf::type_id sumType,
    cudf::column_view sumCol,
    const int64_t* counts,
    const int64_t* overflows,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
    cuda::stream_ref stream);

/**
 * Rewrites the result of cuDF SUM_OVERFLOW with sum_overflow_output::COUNT,
 * whose exact value is wraps * 2^128 + sum, as the CPU
 * LongDecimalWithOverflowState pair, whose exact value is
 * sum + overflow * 2^127. Adds stateOverflowSum to the overflow when merging
 * states. The pair is canonical: sum and overflow share the sign of the exact
 * value and |sum| < 2^127, which is what DecimalUtil::addWithOverflow keeps.
 * Null rows of sum get sum 0 and overflow 0.
 *
 * @param sum DECIMAL128 sums modulo 2^128, rewritten in place.
 * @param wraps INT64 net number of 2^128 wraps per row.
 * @param stateOverflowSum per-row sums of the merged states' overflow, or
 *        nullptr for raw input; null rows count as zero.
 * @param overflow output INT64 overflow, sized to sum.
 * @param stream CUDA stream for the launch.
 */
void assembleDecimalSumOverflow(
    cudf::mutable_column_view sum,
    cudf::column_view wraps,
    const cudf::column_view* stateOverflowSum,
    cudf::mutable_column_view overflow,
    cuda::stream_ref stream);

/**
 * Applies DecimalUtil::adjustSumForOverflow and the DECIMAL(38) range check of
 * the CPU DecimalSumAggregate to each non-null row and writes the adjusted
 * value. Null rows write zero.
 *
 * @param sum DECIMAL128 state sums.
 * @param overflow INT64 net int128 overflow counts.
 * @param out output DECIMAL128 values, sized to sum.
 * @param stream CUDA stream for the launch.
 * @return index of the first row that fails either check, or -1.
 */
cudf::size_type adjustDecimalSumForOverflow(
    cudf::column_view sum,
    cudf::column_view overflow,
    cudf::mutable_column_view out,
    cuda::stream_ref stream);

/**
 * Builds a null mask for rows where sum and count are both valid and count is
 * non-zero, for serializing state or finalizing averages.
 *
 * @param sumCol decoded sum column.
 * @param countCol decoded count column.
 * @param stream CUDA stream for the launch.
 * @param mr memory resource for the returned mask.
 * @return {null mask buffer, null count}.
 */
std::pair<cuda::device_buffer<std::byte>, cudf::size_type>
buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::detail
