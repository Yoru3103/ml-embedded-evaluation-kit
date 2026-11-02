/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ConformerVsiVad.hpp"
#include <cassert>
#include <vector>
using namespace arm::app::conformer_vsi;
int main()
{
    std::vector<int16_t> audio(kMaxSamples);
    Vad vad(audio.data());
    uint64_t pos = 100000;
    for (size_t i = 0; i < kPreRollSamples + 1000; ++i)
        assert(!vad.Push(0, false, pos++));
    assert(!vad.Active());
    const auto speechFirst = pos;
    for (size_t i = 0; i < 4800; ++i)
        assert(!vad.Push(1000, true, pos++));
    assert(vad.Active() && vad.Qualified());
    assert(vad.First() == speechFirst - kPreRollSamples);
    for (size_t i = 0; i < kPostRollSamples; ++i)
        assert(vad.Push(0, false, pos++) == (i + 1 == kPostRollSamples));
    assert(vad.Size() == kPreRollSamples + 4800 + kPostRollSamples);
    assert(vad.First() + vad.Size() == pos);
    vad.Reset();
    assert(!vad.Active());
    assert(!vad.Push(1000, true, pos++));
    assert(!vad.Active());
    // A single loud block must not start a capture or flush a noise-only result.
    for (size_t i = 1; i < 320; ++i)
        assert(!vad.Push(1000, true, pos++));
    assert(!vad.Active() && !vad.Qualified());
    vad.Reset();
    for (size_t i = 0; i < kMaxSamples; ++i)
        assert(vad.Push(1000, true, pos++) == (i + 1 == kMaxSamples));
    assert(vad.Size() == kMaxSamples);
    vad.Reset();
    int16_t full[320];
    for (auto& x : full)
        x = -32768;
    assert(HasSpeechEnergy(full, 320));
    int16_t zero[320]{};
    assert(!HasSpeechEnergy(zero, 320));
    assert(!HasSpeechEnergy(zero, 0));
    for (auto& x : full)
        x = ASR_VSI_VAD_RMS - 1;
    assert(!HasSpeechEnergy(full, 320));
    for (auto& x : full)
        x = ASR_VSI_VAD_RMS;
    assert(HasSpeechEnergy(full, 320));
    for (auto& x : full)
        x = ASR_VSI_VAD_RMS / 2;
    assert(!HasSpeechEnergy(full, 320));
    assert(HasSpeechEnergy(full, 320, true));

    // Preserve a quiet leading syllable and a quiet tail byte-for-byte.
    vad.Reset();
    pos = 0;
    std::vector<int16_t> expected;
    for (size_t i = 0; i < kPreRollSamples; ++i) {
        const int16_t sample = static_cast<int16_t>(i % 100 + 1);
        expected.push_back(sample);
        assert(!vad.Push(sample, false, pos++));
    }
    for (size_t i = 0; i < 4800; ++i) {
        expected.push_back(1200);
        assert(!vad.Push(1200, true, pos++));
    }
    for (size_t i = 0; i < kPostRollSamples; ++i) {
        expected.push_back(99);
        assert(vad.Push(99, false, pos++) == (i + 1 == kPostRollSamples));
    }
    assert(vad.First() == 0 && vad.Size() == expected.size());
    assert(std::equal(expected.begin(), expected.end(), audio.begin()));
}
