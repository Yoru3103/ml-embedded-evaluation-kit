/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "FvpAudioInput.hpp"
#include "RTE_Components.h"
#include "arm_vsi.h"
#include <cstring>
#include <limits>

namespace arm::app::fvp_audio {
namespace {
    /* A private cache-line-aligned DMA buffer, never overlapping application state. */
    alignas(32) int16_t s_pcm[kBlockSamples];
    constexpr uint32_t kMagic = 0x41554431;
} // namespace

bool Open()
{
    ARM_VSI0->Timer.Control = 0;
    ARM_VSI0->DMA.Control   = 0;
    ARM_VSI0->IRQ.Enable    = 0;
    ARM_VSI0->IRQ.Clear     = 1;
    return ARM_VSI0->Regs[1] == kMagic;
}

bool Read(Block& block)
{
    /* Host callback waits for one captured block in wall-clock time. No simulated
     * receive deadline or periodic timer is used to pace a live microphone. */
    ARM_VSI0->Regs[0] = 1;
    block.firstSample = static_cast<uint64_t>(ARM_VSI0->Regs[3]) << 32;
    block.firstSample |= ARM_VSI0->Regs[2];
    block.samples        = ARM_VSI0->Regs[4];
    block.flags          = ARM_VSI0->Regs[5];
    block.backlogSamples = ARM_VSI0->Regs[6];
    if (block.samples > kBlockSamples ||
        (block.flags & ~(EOS | DISCONTINUITY | AUTO_UTTERANCES | WAITING)) ||
        block.firstSample > std::numeric_limits<uint64_t>::max() - block.samples ||
        (block.samples == 0 && !(block.flags & (EOS | WAITING))) ||
        ((block.flags & WAITING) && (block.samples != 0 || block.flags != WAITING))) {
        return false;
    }
    if (block.samples == 0) {
        return true;
    }
    SCB_CleanInvalidateDCache_by_Addr(s_pcm, sizeof(s_pcm));
    ARM_VSI0->DMA.Control    = 0;
    ARM_VSI0->DMA.Address    = reinterpret_cast<uintptr_t>(s_pcm);
    ARM_VSI0->DMA.BlockSize  = sizeof(s_pcm);
    ARM_VSI0->DMA.BlockNum   = 1;
    ARM_VSI0->DMA.Control    = ARM_VSI_DMA_Enable_Msk;
    ARM_VSI0->Regs[7]        = 0;
    ARM_VSI0->Timer.Interval = 1;
    ARM_VSI0->Timer.Control  = ARM_VSI_Timer_Run_Msk | ARM_VSI_Timer_Trig_DMA_Msk;
    while (ARM_VSI0->Regs[7] == 0) {
        __NOP();
    }
    __DSB();
    SCB_InvalidateDCache_by_Addr(s_pcm, sizeof(s_pcm));
    std::memcpy(block.pcm, s_pcm, block.samples * sizeof(int16_t));
    ARM_VSI0->Timer.Control = 0;
    ARM_VSI0->DMA.Control   = 0;
    return true;
}

void Close()
{
    ARM_VSI0->Timer.Control = 0;
    ARM_VSI0->DMA.Control   = 0;
    ARM_VSI0->Regs[0]       = 0;
}

void HoldResult()
{
    for (;;) {
        __WFI();
    }
}
} // namespace arm::app::fvp_audio
