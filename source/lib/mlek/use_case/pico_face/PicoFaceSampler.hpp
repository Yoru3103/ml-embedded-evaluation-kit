/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace arm::app::pico_face {
constexpr size_t kLatentElements    = 8 * 16 * 16;
constexpr size_t kConditionElements = 128;
constexpr size_t kImageWidth        = 128;
constexpr size_t kImageElements     = 3 * kImageWidth * kImageWidth;

/** Runtime options, matching the upstream four-step boot demonstration. */
struct GenerationConfig {
    uint32_t seed{3};
    unsigned steps{4};
    unsigned classId{1};
    float guidance{4.0f};
};

/** Framework-independent synchronous model boundary. */
class PicoFaceModel {
public:
    virtual ~PicoFaceModel()                                                     = default;
    virtual bool DitStep(float* latent, const float* condition, float* velocity) = 0;
    /** Return a CHW float image, valid until the next model call. */
    virtual const float* Decode(float* latent) = 0;
};

/** Deterministic PCG32/CLT-12 generator used by the upstream reference. */
class NoiseGenerator {
public:
    explicit NoiseGenerator(uint32_t seed);
    float Gaussian();

private:
    uint32_t Next();
    uint64_t m_state{0};
};

/** Rectified-flow Euler sampler. Own outside the stack on embedded targets. */
class PicoFaceSampler {
public:
    using StepObserver = bool (*)(PicoFaceModel&, const float*, unsigned);

    /** Conditions are [5][8][128], schedule is [8], both owned by the caller. */
    bool Generate(PicoFaceModel& model,
                  const GenerationConfig& config,
                  const float* conditions,
                  const float* schedule,
                  uint8_t* rgb,
                  StepObserver observe = nullptr);

private:
    std::array<float, kLatentElements> m_latent{};
    std::array<float, kLatentElements> m_velocity{};
    std::array<float, kLatentElements> m_unconditional{};
};
} // namespace arm::app::pico_face
