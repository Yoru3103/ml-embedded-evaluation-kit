/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_FVP_AUDIO_INPUT_HPP
#define MLEK_FVP_AUDIO_INPUT_HPP

#include <cstddef>
#include <cstdint>

namespace arm::app::fvp_audio {
constexpr size_t kBlockSamples     = 320;
constexpr uint32_t EOS             = 1;
constexpr uint32_t DISCONTINUITY   = 2;
constexpr uint32_t ERROR           = 4;
constexpr uint32_t AUTO_UTTERANCES = 8;
constexpr uint32_t WAITING         = 16;

struct Block {
    uint64_t firstSample{0};
    uint32_t samples{0};
    uint32_t flags{0};
    uint32_t backlogSamples{0};
    int16_t pcm[kBlockSamples]{};
};

/** Initialise the application-owned VSI0 audio input, without starting capture. */
bool Open();
/** Pull a host-paced block. EOS is a successful read; bridge errors return false. */
bool Read(Block& block);
/** Stop the VSI DMA peripheral. */
void Close();
/** Keep the last LCD result alive after EOS until the host stops FVP. */
void HoldResult();
/** Host monotonic milliseconds, modulo 2^32; independent of simulation speed. */
uint32_t HostTimeMs();
} // namespace arm::app::fvp_audio
#endif
