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
enum class StateStorage { FLOAT32, INT8, INT16 };

/** Exact state shapes for chunk=32; slot is 1..74. */
std::vector<size_t> StateShape(size_t slot, size_t leftContext = 256);
size_t
StateBytes(size_t slot, StateStorage storage = StateStorage::FLOAT32, size_t leftContext = 256);
inline size_t StateBytes(size_t slot, bool quantized)
{ return StateBytes(slot, quantized ? StateStorage::INT8 : StateStorage::FLOAT32); }

/** Bind separate persistent input and planned output buffers before processing audio. */
class StreamingState {
public:
    static constexpr size_t kCount = 74;
    bool Bind(size_t slot,
              void* input,
              size_t inputBytes,
              const void* output,
              size_t outputBytes,
              StateStorage storage = StateStorage::FLOAT32,
              int32_t zeroPoint    = 0,
              size_t leftContext   = 256);
    bool Bind(size_t slot,
              void* input,
              size_t inputBytes,
              const void* output,
              size_t outputBytes,
              bool quantized,
              int32_t zeroPoint = 0)
    {
        return this->Bind(slot,
                          input,
                          inputBytes,
                          output,
                          outputBytes,
                          quantized ? StateStorage::INT8 : StateStorage::FLOAT32,
                          zeroPoint);
    }
    bool Ready() const;
    bool Reset();
    bool Advance();

private:
    struct Buffer {
        void* input{nullptr};
        const void* output{nullptr};
        size_t bytes{0};
        int resetValue{0};
        StateStorage storage{StateStorage::FLOAT32};
    };
    std::array<Buffer, kCount> m_buffers{};
};
} // namespace arm::app::asr_zh
#endif
