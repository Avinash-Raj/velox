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
#include <cuda/std/span>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace facebook::velox::cudf_velox::detail {

// Size in bytes of each row's packed decimal SUM intermediate state in the
// strings payload (count, overflow placeholder, and 128-bit sum split into
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
 * Encodes each row's partial sum and count into the fixed-width device layout
 * used for VARBINARY interchange.
 *
 * @param sumType DECIMAL64 or DECIMAL128; selects sum storage width.
 * @param offsetType INT32 or INT64; selects offset storage width. sumType and
 *        offsetType are dispatched via cudf::double_type_dispatcher.
 * @param sumCol per-row sums.
 * @param counts per-row int64 counts.
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
 * @param stream CUDA stream for the launch.
 * @return true if every non-null row contains a valid serialized state.
 */
bool unpackDecimalSumState(
    cudf::column_view stateCol,
    cudf::mutable_column_view sumView,
    cudf::mutable_column_view countView,
    cuda::stream_ref stream);

/**
 * Per-row half-up integer divide of sum by count; count == 0 writes zero
 * (validity is applied separately).
 *
 * @param sumType DECIMAL64 or DECIMAL128; selects sum storage width via
 *        cudf::type_dispatcher<cudf::dispatch_storage_type>.
 * @param sumCol per-row sums.
 * @param counts per-row counts.
 * @param outView output per-row averages.
 * @param numRows number of rows.
 * @param stream CUDA stream for the launch.
 */
void averageRoundDecimalSum(
    cudf::type_id sumType,
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
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

// Number of 32-bit limbs a DECIMAL128 value is split into so that cuDF's INT64
// SUM accumulates it exactly. Limbs 0-2 are unsigned and limb 3 is signed, so
// each INT64 limb sum stays exact for fewer than 2^31 rows per group.
constexpr int kDecimal128SumLimbs = 4;

/**
 * Writes the kDecimal128SumLimbs 32-bit limbs of each DECIMAL128 value, least
 * significant first, widened to INT64. Null rows are left unspecified; callers
 * apply the input null mask.
 *
 * @param values DECIMAL128 input; sliced views are supported.
 * @param limbs kDecimal128SumLimbs INT64 outputs sized to values.
 * @param stream CUDA stream for the launch.
 */
void splitDecimal128SumLimbs(
    cudf::column_view values,
    cuda::std::span<const cudf::mutable_column_view> limbs,
    cuda::stream_ref stream);

/**
 * Reassembles per-row limb sums plus net overflow sums into the canonical
 * (sum, overflow) state, whose exact value is sum + overflow * 2^127. sum and
 * overflow share the sign of that value and |sum| < 2^127, which is the form
 * DecimalUtil::addWithOverflow produces for same-signed inputs. Rows whose
 * limb sums are null get sum = 0 and overflow = 0.
 *
 * @param limbSums kDecimal128SumLimbs INT64 limb sums sharing one null mask.
 * @param overflowSum optional INT64 overflow sums; null rows count as zero.
 * @param sum output DECIMAL128 sums.
 * @param overflow output INT64 net overflow counts.
 * @param stream CUDA stream for the launch.
 */
void assembleDecimal128SumState(
    cuda::std::span<const cudf::column_view> limbSums,
    const cudf::column_view* overflowSum,
    cudf::mutable_column_view sum,
    cudf::mutable_column_view overflow,
    cuda::stream_ref stream);

/**
 * Applies the CPU DecimalSumAggregate final check to each valid row: the value
 * sum + overflow * 2^127 must fit int128 (DecimalUtil::adjustSumForOverflow)
 * and DECIMAL(38) (DecimalUtil::valueInRange). Writes that value to out.
 *
 * @param sum DECIMAL128 state sums; null rows are skipped.
 * @param overflow INT64 net overflow counts.
 * @param out output DECIMAL128 values sized to sum.
 * @param stream CUDA stream for the launch.
 * @return index of the first row that fails either check, or -1.
 */
cudf::size_type finalizeDecimal128Sum(
    cudf::column_view sum,
    cudf::column_view overflow,
    cudf::mutable_column_view out,
    cuda::stream_ref stream);

/**
 * Per-row DecimalUtil::computeAverage over sum + overflow * 2^127 and count,
 * including its half-up rounding; count == 0 writes zero (validity is applied
 * separately).
 *
 * @param sum DECIMAL128 state sums.
 * @param counts INT64 counts.
 * @param overflow INT64 net overflow counts.
 * @param out output DECIMAL128 averages sized to sum.
 * @param stream CUDA stream for the launch.
 */
void averageDecimal128WithOverflow(
    cudf::column_view sum,
    cudf::column_view counts,
    cudf::column_view overflow,
    cudf::mutable_column_view out,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox::detail
