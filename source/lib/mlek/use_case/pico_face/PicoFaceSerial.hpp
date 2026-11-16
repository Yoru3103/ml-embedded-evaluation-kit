/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "mlek/use_case/pico_face/PicoFaceSampler.hpp"
#include <array>

namespace arm::app::pico_face {
constexpr size_t kImageHeaderSize = 56;
/** PFIMAGE1, then LE version/dimensions/length/seed/steps/class/guidance/CRC/time. */
std::array<uint8_t, kImageHeaderSize>
BuildImageHeader(const GenerationConfig& config, uint32_t crc, uint64_t generationUs);
} // namespace arm::app::pico_face
