/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/kws_zh/WakeWordDecoder.hpp"
#include "mlek/use_case/kws_zh/KaldiFbank.hpp"
#include <algorithm>
#include <cmath>

namespace arm::app::kws_zh {

WakeWordDecoder::WakeWordDecoder(float threshold, size_t minimumFrames, uint64_t cooldownSamples) :
    m_threshold(threshold), m_minimumFrames(std::max(size_t{1}, minimumFrames)),
    m_cooldown(cooldownSamples)
{}

void WakeWordDecoder::Reset()
{
    m_runs.fill(0);
    m_lastEvent.fill(0);
    m_hasEvent.fill(false);
    m_lastSample = 0;
    m_hasSample  = false;
}

std::vector<WakeWordEvent> WakeWordDecoder::Decode(const float* scores,
                                                   size_t frames,
                                                   uint64_t windowStart,
                                                   uint64_t validSamples)
{
    std::vector<WakeWordEvent> events;
    if (!scores) {
        return events;
    }
    for (size_t frame = 0; frame < frames; ++frame) {
        const uint64_t sample =
            windowStart + frame * KaldiFbank::kFrameShift + KaldiFbank::kFrameLength;
        if (sample > validSamples || (m_hasSample && sample <= m_lastSample)) {
            continue;
        }
        if (m_hasSample && sample - m_lastSample != KaldiFbank::kFrameShift) {
            m_runs.fill(0);
        }
        for (size_t keyword = 0; keyword < kKeywords; ++keyword) {
            const float score = scores[frame * kKeywords + keyword];
            if (!std::isfinite(score) || score < m_threshold) {
                m_runs[keyword] = 0;
                continue;
            }
            ++m_runs[keyword];
            if (m_runs[keyword] == m_minimumFrames &&
                (!m_hasEvent[keyword] || sample - m_lastEvent[keyword] >= m_cooldown)) {
                events.push_back({keyword, score, sample});
                m_lastEvent[keyword] = sample;
                m_hasEvent[keyword]  = true;
            }
        }
        m_lastSample = sample;
        m_hasSample  = true;
    }
    return events;
}

} // namespace arm::app::kws_zh
