/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#include "TokenSampler.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using arm::app::tinystories::kMaxSamplingCandidates;
using arm::app::tinystories::SamplingConfig;
using arm::app::tinystories::SelectToken;
using arm::app::tinystories::XorShift32;

namespace {
SamplingConfig Greedy()
{
    SamplingConfig config{};
    config.greedy = true;
    return config;
}

SamplingConfig Sample(float temperature, float topP, size_t topK)
{
    SamplingConfig config{};
    config.greedy      = false;
    config.temperature = temperature;
    config.topP        = topP;
    config.topK        = topK;
    return config;
}
} // namespace

int main()
{
    // Greedy keeps the first maximum and ignores the generator.
    {
        const std::vector<float> logits{0.1F, 2.5F, -1.0F, 2.5F};
        XorShift32 rng(7);
        int32_t token = -1;
        assert(SelectToken(logits.data(), logits.size(), Greedy(), rng, token));
        assert(token == 1);
    }
    // Non-finite logits are rejected for both modes.
    {
        const std::vector<float> logits{0.0F, std::numeric_limits<float>::quiet_NaN()};
        XorShift32 rng(1);
        int32_t token = -1;
        assert(!SelectToken(logits.data(), logits.size(), Greedy(), rng, token));
        assert(!SelectToken(logits.data(), logits.size(), Sample(1.0F, 0.9F, 8), rng, token));
    }
    // Invalid configurations are rejected.
    {
        const std::vector<float> logits{0.0F};
        XorShift32 rng(1);
        int32_t token = -1;
        assert(!SelectToken(logits.data(), logits.size(), Sample(0.0F, 0.9F, 8), rng, token));
        assert(!SelectToken(logits.data(), logits.size(), Sample(1.0F, 0.0F, 8), rng, token));
        assert(!SelectToken(logits.data(), logits.size(), Sample(1.0F, 1.1F, 8), rng, token));
        assert(!SelectToken(logits.data(), logits.size(), Sample(1.0F, 0.9F, 0), rng, token));
    }
    // A one-hot distribution always selects the dominant token.
    {
        std::vector<float> logits(32000, -10.0F);
        logits[1234] = 10.0F;
        XorShift32 rng(1);
        for (int repeat = 0; repeat < 16; ++repeat) {
            int32_t token = -1;
            assert(SelectToken(logits.data(), logits.size(), Sample(1.0F, 0.9F, 64), rng, token));
            assert(token == 1234);
        }
    }
    // Top-p truncation must never return an excluded tail token.
    {
        std::vector<float> logits(32000, -20.0F);
        logits[10] = 5.0F;
        logits[11] = 4.9F;
        logits[12] = 4.8F;
        XorShift32 rng(0x12345678U);
        for (int repeat = 0; repeat < 256; ++repeat) {
            int32_t token = -1;
            assert(SelectToken(logits.data(), logits.size(), Sample(1.0F, 0.5F, 64), rng, token));
            assert(token >= 10 && token <= 12);
        }
    }
    // Temperature controls sharpness: low temperature concentrates on the top token.
    {
        std::vector<float> logits(32000, -20.0F);
        logits[100] = 5.0F;
        logits[101] = 4.9F;
        XorShift32 rng(99);
        int32_t token = -1;
        assert(SelectToken(logits.data(), logits.size(), Sample(0.01F, 1.0F, 64), rng, token));
        assert(token == 100);
    }
    // Greedy and sampling use the same generator interface; sequences are stable.
    {
        std::vector<float> logits(32000);
        for (size_t index = 0; index < logits.size(); ++index) {
            logits[index] = static_cast<float>((index * 37) % 101) * 0.01F;
        }
        XorShift32 first(42);
        XorShift32 second(42);
        for (int repeat = 0; repeat < 128; ++repeat) {
            int32_t left  = -1;
            int32_t right = -1;
            assert(SelectToken(logits.data(), logits.size(), Sample(1.0F, 0.9F, 64), first, left));
            assert(
                SelectToken(logits.data(), logits.size(), Sample(1.0F, 0.9F, 64), second, right));
            assert(left == right);
        }
    }
    // Generation must stay inside the vocabulary even for a flat distribution.
    {
        std::vector<float> logits(32000, 0.0F);
        XorShift32 rng(5);
        for (int repeat = 0; repeat < 64; ++repeat) {
            int32_t token = -1;
            assert(SelectToken(logits.data(), logits.size(), Sample(1.0F, 0.9F, 64), rng, token));
            assert(token >= 0 && token < 32000);
        }
    }
    std::puts("TokenSampler: greedy, rejection, one-hot, top-p, temperature, determinism passed");
    return 0;
}
