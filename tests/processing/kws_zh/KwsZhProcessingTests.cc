/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/kws_zh/KaldiFbank.hpp"
#include "mlek/use_case/kws_zh/WakeWordDecoder.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

using arm::app::kws_zh::KaldiFbank;
using arm::app::kws_zh::WakeWordDecoder;

/** Export real C++ features for independent torchaudio numerical comparison. */
static bool ExportFeatures(const char* inputPath, const char* outputPath)
{
    std::ifstream input(inputPath, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 0 || input.tellg() % 2 != 0) {
        return false;
    }
    std::vector<int16_t> pcm(static_cast<size_t>(input.tellg()) / sizeof(int16_t));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(int16_t));
    KaldiFbank fbank;
    std::vector<float> features(KaldiFbank::kFrames * KaldiFbank::kBins);
    if (!fbank.Compute(pcm.data(), pcm.size(), features.data(), features.size())) {
        return false;
    }
    std::ofstream output(outputPath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(features.data()), features.size() * sizeof(float));
    return static_cast<bool>(output);
}

/** Exercise overlap deduplication, consecutive frames, cooldown and padding bounds. */
static bool TestDecoder()
{
    std::vector<float> scores(KaldiFbank::kFrames * 2, 0.f);
    for (size_t start : {size_t{0}, size_t{5}, size_t{103}}) {
        for (size_t frame = start; frame < start + 3; ++frame) {
            scores[frame * 2] = 0.9f;
        }
    }
    WakeWordDecoder decoder{0.5f, 3, 16000};
    const auto first = decoder.Decode(scores.data(), KaldiFbank::kFrames, 0, 31920);
    if (first.size() != 2 || first[0].m_sample != 720 || first[1].m_keyword != 0 ||
        first[1].m_sample != 17200 ||
        !decoder.Decode(scores.data(), KaldiFbank::kFrames, 0, 31920).empty()) {
        return false;
    }
    decoder.Reset();
    const auto shortClip = decoder.Decode(scores.data(), KaldiFbank::kFrames, 0, 1000);
    if (shortClip.size() != 1) {
        return false;
    }
    decoder.Reset();
    scores[2]          = std::numeric_limits<float>::quiet_NaN();
    const auto invalid = decoder.Decode(scores.data(), 3, 0, 1000);
    if (!invalid.empty()) {
        return false;
    }
    std::fill(scores.begin(), scores.end(), 0.f);
    for (size_t frame = 100; frame < 103; ++frame) {
        scores[2 * frame + 1] = 0.8f;
    }
    decoder.Reset();
    decoder.Decode(scores.data(), KaldiFbank::kFrames, 0, 31920);
    const auto overlap = decoder.Decode(scores.data(), KaldiFbank::kFrames, 16000, 48000);
    return overlap.size() == 1 && overlap[0].m_keyword == 1 && overlap[0].m_sample == 32720;
}

/** Check Kaldi silence floor and zero-padding equivalence. */
static bool TestFbank()
{
    KaldiFbank fbank;
    std::vector<float> features(KaldiFbank::kFrames * KaldiFbank::kBins);
    if (!fbank.Compute(nullptr, 0, features.data(), features.size())) {
        return false;
    }
    const float floor = std::log(std::numeric_limits<float>::epsilon());
    for (float value : features) {
        if (std::abs(value - floor) > 1.e-5f) {
            return false;
        }
    }
    if (fbank.Compute(nullptr, 1, features.data(), features.size()) ||
        fbank.Compute(nullptr, 0, features.data(), features.size() - 1)) {
        return false;
    }
    std::vector<int16_t> shortAudio(300);
    for (size_t i = 0; i < shortAudio.size(); ++i) {
        shortAudio[i] = static_cast<int16_t>(static_cast<int>((i * 73) % 2000) - 1000);
    }
    fbank.Compute(shortAudio.data(), shortAudio.size(), features.data(), features.size());
    shortAudio.resize(KaldiFbank::kWindowSamples, 0);
    std::vector<float> padded(features.size());
    fbank.Compute(shortAudio.data(), shortAudio.size(), padded.data(), padded.size());
    return features == padded;
}

int main(int argc, char** argv)
{
    if (argc == 4 && std::strcmp(argv[1], "--features") == 0) {
        return ExportFeatures(argv[2], argv[3]) ? 0 : 1;
    }
    if (!TestDecoder() || !TestFbank()) {
        std::fprintf(stderr, "Chinese KWS processing tests failed\n");
        return 1;
    }
    std::puts("Chinese KWS processing tests passed");
    return 0;
}
