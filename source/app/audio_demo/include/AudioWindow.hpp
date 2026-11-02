/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_AUDIO_WINDOW_HPP
#define MLEK_AUDIO_WINDOW_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace arm::app::fvp_audio {
/** Fixed-size sliding PCM window retaining absolute positions after dropped input. */
template <size_t WindowSamples, size_t ExtraSamples>
class AudioWindow {
public:
    void Reset(uint64_t firstSample)
    {
        m_start = firstSample;
        m_count = 0;
    }

    bool Append(const int16_t* pcm, size_t samples)
    {
        if (!pcm || samples > m_pcm.size() - m_count) {
            return false;
        }
        std::copy_n(pcm, samples, m_pcm.data() + m_count);
        m_count += samples;
        return true;
    }

    void Advance(size_t samples)
    {
        const size_t consumed = std::min(samples, m_count);
        m_count -= consumed;
        std::memmove(m_pcm.data(), m_pcm.data() + consumed, m_count * sizeof(int16_t));
        m_start += consumed;
    }

    const int16_t* Data() const
    { return m_pcm.data(); }
    size_t Count() const
    { return m_count; }
    uint64_t Start() const
    { return m_start; }
    bool Ready() const
    { return m_count >= WindowSamples; }

private:
    std::array<int16_t, WindowSamples + ExtraSamples> m_pcm{};
    size_t m_count{0};
    uint64_t m_start{0};
};
} // namespace arm::app::fvp_audio
#endif
