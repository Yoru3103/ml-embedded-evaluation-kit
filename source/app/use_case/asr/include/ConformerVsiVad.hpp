/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CONFORMER_VSI_VAD_HPP
#define CONFORMER_VSI_VAD_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#ifndef ASR_VSI_VAD_RMS
#    define ASR_VSI_VAD_RMS 800
#endif
#ifndef ASR_VSI_SILENCE_MS
#    define ASR_VSI_SILENCE_MS 1000
#endif
#ifndef ASR_VSI_PRE_ROLL_MS
#    define ASR_VSI_PRE_ROLL_MS 600
#endif

namespace arm::app::conformer_vsi {
static_assert(ASR_VSI_VAD_RMS > 0 && ASR_VSI_VAD_RMS < 32768, "Invalid VSI RMS threshold");
static_assert(ASR_VSI_SILENCE_MS >= 20 && ASR_VSI_SILENCE_MS <= 5000,
              "Invalid VSI silence duration");
constexpr size_t kSampleRate = 16000;
static_assert(ASR_VSI_PRE_ROLL_MS >= 200 && ASR_VSI_PRE_ROLL_MS <= 2000,
              "Invalid VSI pre-roll duration");
constexpr size_t kPreRollSamples  = kSampleRate * ASR_VSI_PRE_ROLL_MS / 1000;
constexpr size_t kPostRollSamples = kSampleRate * ASR_VSI_SILENCE_MS / 1000;
constexpr size_t kMaxSamples      = 30 * kSampleRate;
constexpr size_t kMinSamples      = 512;
constexpr size_t kStartSamples    = kSampleRate * 120 / 1000;
constexpr size_t kSpeechSamples   = kSampleRate / 4;
constexpr size_t kHistorySamples  = kPreRollSamples + kStartSamples - 1;

/** Configurable RMS threshold; accumulate in 64 bits for full-scale PCM. */
inline bool HasSpeechEnergy(const int16_t* pcm, size_t samples, bool active = false)
{
    uint64_t energy = 0;
    for (size_t i = 0; i < samples; ++i) {
        const int64_t value = pcm[i];
        energy += static_cast<uint64_t>(value * value);
    }
    const uint64_t threshold = active ? (ASR_VSI_VAD_RMS + 1) / 2 : ASR_VSI_VAD_RMS;
    return samples && energy >= threshold * threshold * samples;
}

/** Pure sample-domain segmenter. Caller owns kMaxSamples storage, position continuity,
 * transport flags and decoding. Consume a completed segment before Reset()/next Push().
 * Idle silence is retained only as pre-roll and never produces a segment.
 */
class Vad {
public:
    explicit Vad(int16_t* audio) : m_audio(audio) {}

    void Reset()
    {
        m_preSize  = 0;
        m_preNext  = 0;
        m_size     = 0;
        m_silence  = 0;
        m_first    = 0;
        m_startRun = 0;
        m_speech   = 0;
    }

    /** Return true at sentence silence or capacity; positions include pre/post-roll. */
    bool Push(int16_t sample, bool speech, uint64_t position)
    {
        if (!Active()) {
            m_startRun = speech ? m_startRun + 1 : 0;
            if (m_startRun < kStartSamples) {
                m_pre[m_preNext] = sample;
                m_preNext        = (m_preNext + 1) % kHistorySamples;
                if (m_preSize < kHistorySamples) {
                    ++m_preSize;
                }
                return false;
            }
            m_speech           = m_startRun - 1;
            m_first            = position - m_preSize;
            const size_t first = (m_preNext + kHistorySamples - m_preSize) % kHistorySamples;
            for (size_t i = 0; i < m_preSize; ++i) {
                m_audio[m_size++] = m_pre[(first + i) % kHistorySamples];
            }
        }
        m_audio[m_size++] = sample;
        m_speech += speech ? 1 : 0;
        m_silence = speech ? 0 : m_silence + 1;
        return m_silence >= kPostRollSamples || m_size == kMaxSamples;
    }

    bool Active() const
    { return m_size != 0; }
    size_t Size() const
    { return m_size; }
    uint64_t First() const
    { return m_first; }
    /** Reject clicks and very short bursts, including at EOS. */
    bool Qualified() const
    { return m_speech >= kSpeechSamples; }

private:
    int16_t* m_audio;
    std::array<int16_t, kHistorySamples> m_pre{};
    size_t m_preSize  = 0;
    size_t m_preNext  = 0;
    size_t m_size     = 0;
    size_t m_silence  = 0;
    uint64_t m_first  = 0;
    size_t m_startRun = 0;
    size_t m_speech   = 0;
};
} // namespace arm::app::conformer_vsi
#endif
