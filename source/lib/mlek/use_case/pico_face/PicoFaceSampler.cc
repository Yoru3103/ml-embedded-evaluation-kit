/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/pico_face/PicoFaceSampler.hpp"
#include "mlek/use_case/pico_face/PicoFaceProcessing.hpp"
#include <cmath>

namespace arm::app::pico_face {
namespace {
    constexpr uint64_t kMultiplier = 6364136223846793005ULL;
    constexpr uint64_t kIncrement  = 1442695040888963407ULL;
} // namespace

NoiseGenerator::NoiseGenerator(uint32_t seed)
{
    m_state = m_state * kMultiplier + kIncrement;
    m_state += seed;
    m_state = m_state * kMultiplier + kIncrement;
}

uint32_t NoiseGenerator::Next()
{
    uint64_t value = m_state;
    unsigned count = static_cast<unsigned>(value >> 59);
    m_state        = value * kMultiplier + kIncrement;
    value ^= value >> 18;
    uint32_t output = static_cast<uint32_t>(value >> 27);
    return (output >> count) | (output << ((32 - count) & 31));
}

float NoiseGenerator::Gaussian()
{
    int32_t value = 0;
    for (unsigned i = 0; i < 12; ++i) {
        value += static_cast<int32_t>(Next() >> 20);
    }
    return static_cast<float>(value - 24576) / 4096.0f;
}

bool PicoFaceSampler::Generate(PicoFaceModel& model,
                               const GenerationConfig& config,
                               const float* conditions,
                               const float* schedule,
                               uint8_t* rgb)
{
    if (!conditions || !schedule || !rgb || config.classId >= 5 ||
        !std::isfinite(config.guidance) || config.guidance < 0.0f ||
        (config.steps != 1 && config.steps != 2 && config.steps != 4 && config.steps != 8)) {
        return false;
    }
    NoiseGenerator noise(config.seed);
    for (auto& value : m_latent) {
        value = noise.Gaussian();
    }
    const unsigned stride = 8 / config.steps;
    for (unsigned i = 0; i < config.steps; ++i) {
        const unsigned k      = i * stride;
        const float next      = i + 1 < config.steps ? schedule[k + stride] : 0.0f;
        const float dt        = schedule[k] - next;
        const auto* condition = conditions + (config.classId * 8 + k) * kConditionElements;
        if (!model.DitStep(m_latent.data(), condition, m_velocity.data())) {
            return false;
        }
        if (config.classId != 4 && config.guidance > 0.0f) {
            condition = conditions + (4 * 8 + k) * kConditionElements;
            if (!model.DitStep(m_latent.data(), condition, m_unconditional.data())) {
                return false;
            }
            for (size_t j = 0; j < kLatentElements; ++j) {
                m_velocity[j] =
                    m_unconditional[j] + config.guidance * (m_velocity[j] - m_unconditional[j]);
            }
        }
        for (size_t j = 0; j < kLatentElements; ++j) {
            m_latent[j] -= dt * m_velocity[j];
        }
    }
    const float* image = model.Decode(m_latent.data());
    if (!image) {
        return false;
    }
    ConvertToRgb(image, rgb, kImageWidth * kImageWidth);
    return true;
}
} // namespace arm::app::pico_face
