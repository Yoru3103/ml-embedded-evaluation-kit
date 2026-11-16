/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_ASR_ZH_KALDI_FBANK_HPP
#define MLEK_ASR_ZH_KALDI_FBANK_HPP
#include "mlek/math/PlatformMath.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace arm::app::asr_zh {
/** Fixed-checkpoint Kaldi Fbank. PCM remains caller-owned for the utterance. */
class KaldiFbank {
public:
    static constexpr size_t kBins        = 80;
    static constexpr size_t kFrameShift  = 160;
    static constexpr size_t kFrameLength = 400;
    static constexpr size_t kFftLength   = 512;
    static constexpr size_t kChunkFrames = 77;
    static constexpr size_t kAdvance     = 64;
    static constexpr float kPadding      = -23.02585093f;

    KaldiFbank();
    bool Begin(const int16_t* audio, size_t samples);
    /** Return the next padded input; calculate every real feature frame only once. */
    bool Next(float* output, size_t elements);
    size_t Frames() const
    { return m_frames; }
    size_t ComputedFrames() const
    { return m_computed; }

private:
    bool ComputeFrame(size_t frame, float* output);
    const int16_t* m_audio{nullptr};
    size_t m_samples{0};
    size_t m_frames{0};
    size_t m_start{0};
    size_t m_computed{0};
    std::array<float, kFrameLength> m_window{};
    std::array<std::vector<float>, kBins> m_filters;
    std::array<size_t, kBins> m_first{};
    std::array<float, kChunkFrames * kBins> m_features{};
    std::vector<float> m_frame;
    std::vector<float> m_fft;
    math::FftInstance m_fftInstance{};
};
} // namespace arm::app::asr_zh
#endif
