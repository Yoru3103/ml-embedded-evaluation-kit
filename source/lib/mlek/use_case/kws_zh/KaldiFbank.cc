/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/kws_zh/KaldiFbank.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace arm::app::kws_zh {

KaldiFbank::KaldiFbank() : m_frame(kFftLength), m_fft(kFftLength)
{
    constexpr float kPi = 3.14159265358979323846f;
    for (size_t i = 0; i < kFrameLength; ++i) {
        m_window[i] = std::pow(0.5f - 0.5f * std::cos(2.f * kPi * i / (kFrameLength - 1)), 0.85f);
    }
    const float melLow  = 1127.f * std::log(1.f + 20.f / 700.f);
    const float melHigh = 1127.f * std::log(1.f + 8000.f / 700.f);
    const float delta   = (melHigh - melLow) / (kBins + 1);
    for (size_t bin = 0; bin < kBins; ++bin) {
        const float left   = melLow + bin * delta;
        const float centre = left + delta;
        const float right  = centre + delta;
        for (size_t fftBin = 0; fftBin < kFftLength / 2; ++fftBin) {
            const float frequency  = static_cast<float>(fftBin * kSampleRate) / kFftLength;
            const float mel        = 1127.f * std::log(1.f + frequency / 700.f);
            const float up         = (mel - left) / (centre - left);
            const float down       = (right - mel) / (right - centre);
            m_filters[bin][fftBin] = std::max(0.f, std::min(up, down));
        }
    }
    math::MathUtils::FftInitF32(kFftLength, m_fftInstance);
}

bool KaldiFbank::Compute(const int16_t* audio, size_t count, float* output, size_t elements)
{
    if ((!audio && count) || !output || elements != kFrames * kBins ||
        !m_fftInstance.m_initialised) {
        return false;
    }
    std::array<float, kFftLength / 2 + 1> power{};
    for (size_t frame = 0; frame < kFrames; ++frame) {
        std::fill(m_frame.begin(), m_frame.end(), 0.f);
        float mean = 0.f;
        for (size_t i = 0; i < kFrameLength; ++i) {
            const size_t sample = frame * kFrameShift + i;
            m_frame[i]          = sample < count ? static_cast<float>(audio[sample]) : 0.f;
            mean += m_frame[i];
        }
        mean /= kFrameLength;
        for (size_t i = 0; i < kFrameLength; ++i) {
            m_frame[i] -= mean;
        }
        for (size_t i = kFrameLength - 1; i > 0; --i) {
            m_frame[i] -= 0.97f * m_frame[i - 1];
        }
        m_frame[0] *= 1.f - 0.97f;
        for (size_t i = 0; i < kFrameLength; ++i) {
            m_frame[i] *= m_window[i];
        }
        math::MathUtils::FftF32(m_frame, m_fft, m_fftInstance);
        power[0]              = m_fft[0] * m_fft[0];
        power[kFftLength / 2] = m_fft[1] * m_fft[1];
        for (size_t i = 1; i < kFftLength / 2; ++i) {
            power[i] = m_fft[2 * i] * m_fft[2 * i] + m_fft[2 * i + 1] * m_fft[2 * i + 1];
        }
        for (size_t bin = 0; bin < kBins; ++bin) {
            float energy = 0.f;
            for (size_t i = 0; i < power.size(); ++i) {
                energy += m_filters[bin][i] * power[i];
            }
            output[frame * kBins + bin] =
                std::log(std::max(energy, std::numeric_limits<float>::epsilon()));
        }
    }
    return true;
}

} // namespace arm::app::kws_zh
