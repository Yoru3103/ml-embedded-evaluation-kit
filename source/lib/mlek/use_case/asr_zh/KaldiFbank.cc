/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/asr_zh/KaldiFbank.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace arm::app::asr_zh {
KaldiFbank::KaldiFbank() : m_frame(kFftLength), m_fft(kFftLength)
{
    constexpr double kPi = 3.14159265358979323846;
    for (size_t i = 0; i < kFrameLength; ++i) {
        m_window[i] = static_cast<float>(
            std::pow(0.5 - 0.5 * std::cos(2.0 * kPi * i / (kFrameLength - 1)), 0.85));
    }
    const auto mel   = [](float hz) { return 1127.f * std::log(1.f + hz / 700.f); };
    const float low  = mel(20.f);
    const float step = (mel(7600.f) - low) / (kBins + 1);
    for (size_t bin = 0; bin < kBins; ++bin) {
        const float left   = low + bin * step;
        const float centre = low + (bin + 1) * step;
        const float right  = low + (bin + 2) * step;
        for (size_t fft = 0; fft < kFftLength / 2; ++fft) {
            const float frequency = mel(16000.f / kFftLength * fft);
            if (frequency > left && frequency < right) {
                if (m_filters[bin].empty()) {
                    m_first[bin] = fft;
                }
                m_filters[bin].push_back(frequency <= centre
                                             ? (frequency - left) / (centre - left)
                                             : (right - frequency) / (right - centre));
            }
        }
    }
    math::MathUtils::FftInitF32(kFftLength, m_fftInstance);
}

bool KaldiFbank::Begin(const int16_t* audio, size_t samples)
{
    if (!audio || !samples || samples > static_cast<size_t>(INT32_MAX)) {
        return false;
    }
    m_audio    = audio;
    m_samples  = samples;
    m_frames   = (samples + kFrameShift / 2) / kFrameShift;
    m_start    = 0;
    m_computed = 0;
    m_features.fill(kPadding);
    return m_frames > 0 && m_fftInstance.m_initialised;
}

bool KaldiFbank::ComputeFrame(size_t frame, float* output)
{
    if (!m_audio || frame >= m_frames || !output) {
        return false;
    }
    std::fill(m_frame.begin(), m_frame.end(), 0.f);
    const int64_t start = static_cast<int64_t>(frame * kFrameShift) -
                          static_cast<int64_t>((kFrameLength - kFrameShift) / 2);
    float sum           = 0.f;
    for (size_t i = 0; i < kFrameLength; ++i) {
        int64_t sample = start + static_cast<int64_t>(i);
        while (sample < 0 || sample >= static_cast<int64_t>(m_samples)) {
            sample = sample < 0 ? -sample - 1 : 2 * static_cast<int64_t>(m_samples) - sample - 1;
        }
        m_frame[i] = static_cast<float>(m_audio[sample]) * (1.f / 32768.f);
        sum += m_frame[i];
    }
    const float mean = sum / kFrameLength;
    for (size_t i = 0; i < kFrameLength; ++i) {
        m_frame[i] -= mean;
    }
    for (size_t i = kFrameLength - 1; i > 0; --i) {
        m_frame[i] -= 0.97f * m_frame[i - 1];
    }
    m_frame[0] -= 0.97f * m_frame[0];
    for (size_t i = 0; i < kFrameLength; ++i) {
        m_frame[i] *= m_window[i];
    }
    math::MathUtils::FftF32(m_frame, m_fft, m_fftInstance);
    std::array<float, kFftLength / 2> power{};
    power[0] = m_fft[0] * m_fft[0];
    for (size_t i = 1; i < power.size(); ++i) {
        power[i] = m_fft[2 * i] * m_fft[2 * i] + m_fft[2 * i + 1] * m_fft[2 * i + 1];
    }
    for (size_t bin = 0; bin < kBins; ++bin) {
        float energy = 0.f;
        for (size_t i = 0; i < m_filters[bin].size(); ++i) {
            energy += m_filters[bin][i] * power[m_first[bin] + i];
        }
        output[bin] = std::log(std::max(energy, std::numeric_limits<float>::epsilon()));
    }
    ++m_computed;
    return true;
}

bool KaldiFbank::Next(float* output, size_t elements)
{
    if (!output || elements != m_features.size() || !m_audio || m_start >= m_frames) {
        return false;
    }
    size_t first = 0;
    if (m_start != 0) {
        constexpr size_t kOverlap = kChunkFrames - kAdvance;
        std::memmove(m_features.data(),
                     m_features.data() + kAdvance * kBins,
                     kOverlap * kBins * sizeof(float));
        first = kOverlap;
    }
    for (size_t i = first; i < kChunkFrames; ++i) {
        auto* destination = m_features.data() + i * kBins;
        if (m_start + i < m_frames) {
            if (!ComputeFrame(m_start + i, destination)) {
                return false;
            }
        } else {
            std::fill_n(destination, kBins, kPadding);
        }
    }
    std::copy(m_features.begin(), m_features.end(), output);
    m_start += kAdvance;
    return true;
}
} // namespace arm::app::asr_zh
