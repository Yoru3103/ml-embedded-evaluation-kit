/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ConformerVsiVad.hpp"
#include <cassert>
using namespace arm::app::conformer_vsi;
static_assert(kPostRollSamples == 16000);
int main()
{
    int16_t pcm[320];
    for (auto& sample : pcm)
        sample = 399;
    assert(!HasSpeechEnergy(pcm, 320));
    for (auto& sample : pcm)
        sample = 400;
    assert(HasSpeechEnergy(pcm, 320));
}
