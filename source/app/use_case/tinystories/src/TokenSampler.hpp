/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace arm::app::tinystories {

/** Largest candidate set kept before top-p truncation. */
constexpr size_t kMaxSamplingCandidates = 64;

/** Token selection settings shared by the test and demo builds. */
struct SamplingConfig {
    bool greedy       = true;
    float temperature = 1.0F;
    float topP        = 0.9F;
    size_t topK       = kMaxSamplingCandidates;
};

/** Small deterministic generator; sampling must be reproducible from a seed. */
class XorShift32 {
public:
    explicit XorShift32(uint32_t seed = 1) : m_state(seed == 0 ? 1 : seed) {}

    uint32_t Next()
    {
        uint32_t value = m_state;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        m_state = value;
        return value;
    }

    /** Uniform value in [0, 1). */
    float Uniform()
    { return static_cast<float>(Next() >> 8) * (1.0F / 16777216.0F); }

private:
    uint32_t m_state;
};

/**
 * Select the next token from one logits vector.
 *
 * Greedy selection keeps the historical argmax behaviour. Sampling keeps the
 * top-k logits, applies temperature, truncates the smallest tail of the
 * distribution until the retained mass reaches top-p and draws from it.
 *
 * @param logits  FP32 logits, one per vocabulary entry.
 * @param count   Number of logits.
 * @param config  Greedy or sampling settings; invalid values are rejected.
 * @param rng     Generator state advanced only when sampling.
 * @param token   Selected token index.
 * @returns       False for a null/empty vector, a non-finite logit or an
 *                unsupported configuration.
 */
bool SelectToken(const float* logits,
                 size_t count,
                 const SamplingConfig& config,
                 XorShift32& rng,
                 int32_t& token);

} // namespace arm::app::tinystories
