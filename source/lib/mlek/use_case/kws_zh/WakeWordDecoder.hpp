/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_KWS_ZH_WAKE_WORD_DECODER_HPP
#define MLEK_KWS_ZH_WAKE_WORD_DECODER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace arm::app::kws_zh {

struct WakeWordEvent {
    size_t m_keyword;
    float m_score;
    uint64_t m_sample;
};

/** Decode independent sigmoid scores, deduplicating overlapping windows. */
class WakeWordDecoder {
public:
    static constexpr size_t kKeywords = 2;
    WakeWordDecoder(float threshold, size_t minimumFrames, uint64_t cooldownSamples);
    void Reset();
    std::vector<WakeWordEvent>
    Decode(const float* scores, size_t frames, uint64_t windowStart, uint64_t validSamples);

private:
    float m_threshold;
    size_t m_minimumFrames;
    uint64_t m_cooldown;
    std::array<size_t, kKeywords> m_runs{};
    std::array<uint64_t, kKeywords> m_lastEvent{};
    std::array<bool, kKeywords> m_hasEvent{};
    uint64_t m_lastSample{0};
    bool m_hasSample{false};
};

} // namespace arm::app::kws_zh
#endif
