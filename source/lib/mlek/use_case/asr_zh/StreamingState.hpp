/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_ASR_ZH_STREAMING_STATE_HPP
#define MLEK_ASR_ZH_STREAMING_STATE_HPP
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace arm::app::asr_zh {
/** Exact state shapes for the fixed chunk=32, left=256 checkpoint; slot is 1..74. */
std::vector<size_t> StateShape(size_t slot);
size_t StateBytes(size_t slot, bool quantized = false);

/** Bind separate persistent input and planned output buffers before processing audio. */
class StreamingState {
public:
    static constexpr size_t kCount = 74;
    bool Bind(size_t slot,
              void* input,
              size_t inputBytes,
              const void* output,
              size_t outputBytes,
              bool quantized    = false,
              int32_t zeroPoint = 0);
    bool Ready() const;
    bool Reset();
    bool Advance();

private:
    struct Buffer {
        void* input{nullptr};
        const void* output{nullptr};
        size_t bytes{0};
        int resetValue{0};
    };
    std::array<Buffer, kCount> m_buffers{};
};
} // namespace arm::app::asr_zh
#endif
