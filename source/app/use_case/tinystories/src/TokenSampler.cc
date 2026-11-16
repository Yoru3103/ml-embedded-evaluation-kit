/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#include "TokenSampler.hpp"

#include <cmath>
#include <limits>

namespace arm::app::tinystories {
namespace {
    struct Candidate {
        float value;
        int32_t index;
    };

    /** Keep the candidates ordered by descending logit; equal values keep input order. */
    void InsertCandidate(
        Candidate* candidates, size_t& count, size_t capacity, float value, int32_t index)
    {
        if (count == capacity && value <= candidates[count - 1].value) {
            return;
        }
        size_t position = (count < capacity) ? count : capacity - 1;
        while (position > 0 && candidates[position - 1].value < value) {
            candidates[position] = candidates[position - 1];
            --position;
        }
        candidates[position] = Candidate{value, index};
        if (count < capacity) {
            ++count;
        }
    }
} // namespace

bool SelectToken(const float* logits,
                 size_t count,
                 const SamplingConfig& config,
                 XorShift32& rng,
                 int32_t& token)
{
    if (logits == nullptr || count == 0 || !std::isfinite(config.temperature) ||
        config.temperature <= 0.0F || !std::isfinite(config.topP) || config.topP <= 0.0F ||
        config.topP > 1.0F || config.topK == 0) {
        return false;
    }

    Candidate candidates[kMaxSamplingCandidates]{};
    size_t candidateCount = 0;
    const size_t capacity =
        (config.topK < kMaxSamplingCandidates) ? config.topK : kMaxSamplingCandidates;
    float maximum    = -std::numeric_limits<float>::infinity();
    int32_t topIndex = 0;
    for (size_t index = 0; index < count; ++index) {
        const float value = logits[index];
        if (!std::isfinite(value)) {
            return false;
        }
        if (value > maximum) {
            maximum  = value;
            topIndex = static_cast<int32_t>(index);
        }
        if (!config.greedy) {
            InsertCandidate(
                candidates, candidateCount, capacity, value, static_cast<int32_t>(index));
        }
    }
    if (config.greedy) {
        token = topIndex;
        return true;
    }

    double weights[kMaxSamplingCandidates]{};
    double total                    = 0.0;
    const double inverseTemperature = 1.0 / static_cast<double>(config.temperature);
    for (size_t index = 0; index < candidateCount; ++index) {
        weights[index] =
            std::exp((static_cast<double>(candidates[index].value) - static_cast<double>(maximum)) *
                     inverseTemperature);
        total += weights[index];
    }
    if (!(total > 0.0)) {
        token = candidates[0].index;
        return true;
    }

    const double target = static_cast<double>(config.topP) * total;
    double cumulative   = 0.0;
    size_t kept         = 0;
    for (; kept < candidateCount; ++kept) {
        cumulative += weights[kept];
        if (cumulative >= target) {
            ++kept;
            break;
        }
    }
    if (kept == 0) {
        kept = 1;
    }

    const double draw = static_cast<double>(rng.Uniform()) * cumulative;
    double running    = 0.0;
    for (size_t index = 0; index < kept; ++index) {
        running += weights[index];
        if (draw < running) {
            token = candidates[index].index;
            return true;
        }
    }
    token = candidates[kept - 1].index;
    return true;
}

} // namespace arm::app::tinystories
