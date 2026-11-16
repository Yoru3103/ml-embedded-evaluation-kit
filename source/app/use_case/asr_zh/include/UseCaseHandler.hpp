/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ASR_ZH_USE_CASE_HANDLER_HPP
#define ASR_ZH_USE_CASE_HANDLER_HPP
#include "mlek/fwk/executorch/EtModel.hpp"
namespace arm::app::asr_zh {
#if (defined(ASR_ZH_INPUT_VSI) || defined(ASR_ZH_INPUT_UART))
void CompleteDemoClip(bool success);
bool OpenDemoInput();
const int16_t* ReadDemoClip(uint32_t& count);
bool CloseDemoInput();
void ShowDemoStatus(const char* text);
#endif
bool ProcessAudio(fwk::et::EtModel& model);
} // namespace arm::app::asr_zh
#endif
