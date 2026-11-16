/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/pico_face/PicoFaceProcessing.hpp"
#include "mlek/use_case/pico_face/PicoFaceSampler.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace arm::app::pico_face;

static void Check(bool value)
{
    if (!value) {
        std::puts("FAIL");
        std::exit(1);
    }
}

/** Analytic constant velocity checks Euler sign, schedule, guidance and reset. */
class ConstantModel : public PicoFaceModel {
public:
    unsigned calls{0};
    unsigned decodes{0};
    float finalLatent{0};
    bool DitStep(float*, const float* condition, float* velocity) override
    {
        ++calls;
        for (size_t j = 0; j < kLatentElements; ++j) {
            velocity[j] = condition[0];
        }
        return true;
    }
    const float* Decode(float* latent) override
    {
        ++decodes;
        finalLatent = latent[0];
        return m_image.data();
    }

private:
    std::array<float, kImageElements> m_image{};
};

int main()
{
    const float chw[] = {-2.0f, 1.0f, 0.0f, -1.0f, 2.0f, 0.0f};
    uint8_t rgb[6]{};
    ConvertToRgb(chw, rgb, 2);
    const uint8_t expected[] = {0, 128, 255, 255, 0, 128};
    for (size_t i = 0; i < 6; ++i) {
        Check(rgb[i] == expected[i]);
    }
    Check(ImageCrc32(reinterpret_cast<const uint8_t*>("123456789"), 9) == 0xcbf43926U);
    std::array<float, 5 * 8 * 128> conditions{};
    for (size_t k = 0; k < 8; ++k) {
        conditions[(1 * 8 + k) * 128] = 2.0f;
        conditions[(4 * 8 + k) * 128] = 1.0f;
    }
    const float schedule[] = {1, .875f, .75f, .625f, .5f, .375f, .25f, .125f};
    std::array<uint8_t, kImageElements> image{};
    PicoFaceSampler sampler;
    ConstantModel model;
    GenerationConfig config;
    NoiseGenerator noise(config.seed);
    const float initial = noise.Gaussian();
    Check(sampler.Generate(model, config, conditions.data(), schedule, image.data()));
    Check(model.calls == 8 && model.decodes == 1);
    Check(std::fabs(model.finalLatent - (initial - 5.0f)) < 1e-5f);
    config.steps    = 1;
    config.guidance = 0;
    model.calls     = 0;
    Check(sampler.Generate(model, config, conditions.data(), schedule, image.data()));
    Check(model.calls == 1 && model.decodes == 2);
    Check(std::fabs(model.finalLatent - (initial - 2.0f)) < 1e-5f);
    config.steps = 3;
    Check(!sampler.Generate(model, config, conditions.data(), schedule, image.data()));
    Check(model.calls == 1);
    std::puts("PASS");
}
