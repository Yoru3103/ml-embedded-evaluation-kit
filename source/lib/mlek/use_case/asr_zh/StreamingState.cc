/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/asr_zh/StreamingState.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>

namespace arm::app::asr_zh {
std::vector<size_t> StateShape(size_t slot, size_t leftContext)
{
    if (slot == 73) {
        return {1, 128, 3, 19};
    }
    if (slot == 74) {
        return {1};
    }
    if (slot == 0 || slot > 74) {
        return {};
    }
    constexpr std::array<size_t, 6> kDownsample{1, 2, 4, 8, 4, 2};
    constexpr std::array<size_t, 6> kHeads{4, 4, 4, 8, 4, 4};
    constexpr std::array<size_t, 6> kKernel{31, 31, 15, 15, 15, 31};
    const size_t stage     = (slot - 1) / 12;
    const size_t dimension = stage == 0 ? 192 : 256;
    if (!leftContext || leftContext % kDownsample[stage] != 0) {
        return {};
    }
    const size_t left = leftContext / kDownsample[stage];
    switch ((slot - 1) % 6) {
    case 0:
        return {left, 1, kHeads[stage] * 32};
    case 1:
        return {1, 1, left, dimension * 3 / 4};
    case 2:
    case 3:
        return {left, 1, kHeads[stage] * 12};
    default:
        return {1, dimension, kKernel[stage] / 2};
    }
}

size_t StateBytes(size_t slot, StateStorage storage, size_t leftContext)
{
    const auto shape          = StateShape(slot, leftContext);
    const size_t elementBytes = slot == 74 || storage == StateStorage::FLOAT32 ? 4
                                : storage == StateStorage::INT16               ? 2
                                                                               : 1;
    return shape.empty() ? 0
                         : elementBytes * std::accumulate(shape.begin(),
                                                          shape.end(),
                                                          size_t{1},
                                                          std::multiplies<size_t>());
}

bool StreamingState::Bind(size_t slot,
                          void* input,
                          size_t inputBytes,
                          const void* output,
                          size_t outputBytes,
                          StateStorage storage,
                          int32_t zeroPoint,
                          size_t leftContext)
{
    const size_t bytes = StateBytes(slot, storage, leftContext);
    if (!bytes || !input || !output || inputBytes != bytes || outputBytes != bytes ||
        (storage == StateStorage::INT8 && (zeroPoint < -128 || zeroPoint > 127)) ||
        (storage != StateStorage::INT8 && zeroPoint != 0) ||
        (slot == 74 && storage != StateStorage::FLOAT32)) {
        return false;
    }
    m_buffers[slot - 1] = {input, output, bytes, zeroPoint, storage};
    return true;
}

bool StreamingState::Ready() const
{
    return std::all_of(m_buffers.begin(), m_buffers.end(), [](const Buffer& buffer) {
        return buffer.input && buffer.output;
    });
}

bool StreamingState::Reset()
{
    if (!Ready()) {
        return false;
    }
    for (const auto& buffer : m_buffers) {
        std::memset(buffer.input, buffer.resetValue, buffer.bytes);
    }
    return true;
}

bool StreamingState::Advance()
{
    if (!Ready()) {
        return false;
    }
    for (const auto& buffer : m_buffers) {
        if (buffer.storage == StateStorage::INT16) {
            const auto* output = static_cast<const int16_t*>(buffer.output);
            auto* input        = static_cast<int16_t*>(buffer.input);
            const size_t count = buffer.bytes / sizeof(int16_t);
            for (size_t i = 0; i < count; ++i) {
                // PT2E's input quantizer saturates to [-32767, 32767].
                input[i] = output[i] == std::numeric_limits<int16_t>::min()
                               ? static_cast<int16_t>(std::numeric_limits<int16_t>::min() + 1)
                               : output[i];
            }
        } else {
            std::memmove(buffer.input, buffer.output, buffer.bytes);
        }
    }
    return true;
}
} // namespace arm::app::asr_zh
