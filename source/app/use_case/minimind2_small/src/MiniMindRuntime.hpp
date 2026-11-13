/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "mlek/fwk/executorch/EtModel.hpp"

#include <cstddef>
#include <cstdint>

namespace arm::app::minimind2_small {
struct Timing {
    uint64_t npu      = 0;
    uint64_t invoke   = 0;
    uint64_t endToEnd = 0;
    size_t calls      = 0;
};

uint64_t CpuCycles();
double Milliseconds(uint64_t cycles, uint32_t frequency);
void PrintTiming(const char* phase, size_t position, const Timing& timing);
bool Infer(fwk::et::EtModel& model,
           int32_t token,
           size_t position,
           int32_t& nextToken,
           Timing& total,
           const char* phase);
} // namespace arm::app::minimind2_small
