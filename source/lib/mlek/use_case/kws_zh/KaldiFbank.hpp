/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_KWS_ZH_KALDI_FBANK_HPP
#define MLEK_KWS_ZH_KALDI_FBANK_HPP

#include "mlek/math/PlatformMath.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace arm::app::kws_zh {

/** Kaldi-compatible, unnormalised log-power Fbank for the WeKWS checkpoint. */
class KaldiFbank {
public:
    static constexpr size_t kSampleRate    = 16000;
    static constexpr size_t kFrameLength   = 400;
    static constexpr size_t kFrameShift    = 160;
    static constexpr size_t kFftLength     = 512;
    static constexpr size_t kBins          = 40;
    static constexpr size_t kFrames        = 198;
    static constexpr size_t kWindowSamples = (kFrames - 1) * kFrameShift + kFrameLength;

    KaldiFbank();

    /** Extract a fixed window, padding missing PCM samples with zero. */
    bool Compute(const int16_t* audio, size_t count, float* output, size_t elements);

private:
    std::array<float, kFrameLength> m_window{};
    std::array<std::array<float, kFftLength / 2 + 1>, kBins> m_filters{};
    std::vector<float> m_frame;
    std::vector<float> m_fft;
    math::FftInstance m_fftInstance{};
};

} // namespace arm::app::kws_zh
#endif
