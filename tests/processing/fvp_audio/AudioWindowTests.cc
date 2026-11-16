/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "AudioWindow.hpp"
#include "GapRecovery.hpp"
#include <array>
#include <cstdlib>
#include <iostream>

static void Require(bool condition)
{
    if (!condition) {
        std::cerr << "Audio window invariant failed\n";
        std::exit(1);
    }
}

int main()
{
    using arm::app::fvp_audio::AudioWindow;
    using arm::app::fvp_audio::GapRecovery;
    static AudioWindow<31920, 320> window;
    std::array<int16_t, 320> pcm{};
    /* Includes block/window non-divisibility: the first window leaves 80 samples
     * beyond its feature boundary, and overlaps must retain these samples. */
    for (size_t hop : {size_t{3200}, size_t{16000}}) {
        window.Reset(0);
        uint64_t inputSamples = 0;
        size_t windows        = 0;
        uint64_t lastStart    = 0;
        for (size_t block = 0; block < 150; ++block) {
            for (size_t i = 0; i < pcm.size(); ++i) {
                pcm[i] = static_cast<int16_t>((inputSamples + i) % 30000);
            }
            Require(window.Append(pcm.data(), pcm.size()));
            inputSamples += pcm.size();
            if (window.Ready()) {
                Require(window.Start() == windows * hop);
                for (size_t i = 0; i < window.Count(); ++i) {
                    Require(window.Data()[i] == static_cast<int16_t>((window.Start() + i) % 30000));
                }
                lastStart = window.Start();
                window.Advance(hop);
                ++windows;
            }
        }
        Require(windows == (hop == 3200 ? 6 : 2) && lastStart == 16000);
        Require(window.Start() + window.Count() == inputSamples);
    }
    // Cross 100 seconds of source time, suffer a long overrun episode, then recover.
    // Each gap still invalidates the audio context; only one UI transition is needed.
    window.Reset(0);
    GapRecovery recovery{31920};
    uint64_t sample         = 0;
    size_t notices          = 0;
    size_t recoveredWindows = 0;
    for (size_t block = 0; block < 8000; ++block) {
        const bool gap = block >= 5000 && block < 6500;
        if (gap) {
            sample += 1600;
            window.Reset(sample);
        }
        notices += recovery.Observe(gap, pcm.size()) ? 1 : 0;
        Require(window.Append(pcm.data(), pcm.size()));
        if (window.Ready()) {
            Require(!recovery.Active());
            if (block >= 6500) {
                ++recoveredWindows;
            }
            window.Advance(16000);
        }
        sample += pcm.size();
    }
    Require(notices == 1 && recoveredWindows > 0 && sample > 16000 * 100);
    Require(!window.Append(pcm.data(), 40000));
    Require(!window.Append(nullptr, 1));
    window.Reset(80000);
    Require(!window.Ready() && window.Count() == 0 && window.Start() == 80000);
    Require(window.Append(pcm.data(), 123));
    window.Advance(999);
    Require(window.Count() == 0 && window.Start() == 80123);
    std::cout << "Audio window overlap, capacity and gap tests: PASS\n";
}
