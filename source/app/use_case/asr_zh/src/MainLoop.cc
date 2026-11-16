/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "BufAttributes.hpp"
#if defined(ASR_ZH_INPUT_VSI)
#    include "FvpAudioInput.hpp"
#endif
#include "UseCaseHandler.hpp"
#include "mlek/log/log_macros.h"
#if ASR_ZH_PROFILE
#    include "AsrZhOpProfiler.hpp"
#endif

namespace arm::app::asr_zh {
extern uint8_t* GetModelPointer();
extern size_t GetModelLen();
#if ASR_ZH_METHOD_POOL_IN_DRAM
static uint8_t s_methodPool[ACTIVATION_BUF_SZ]
    __attribute__((aligned(16), section("activation_buf_dram")));
#else
static uint8_t s_methodPool[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
#endif
} // namespace arm::app::asr_zh

void MainLoop()
{
    using namespace arm::app;
    fwk::et::EtModel model;
    fwk::et::EtBackendData backend;
    backend.m_logTensorDetails = false;
#if ASR_ZH_PROFILE
    constexpr unsigned int conversions = (ASR_ZH_INT8_STATE || ASR_ZH_INT16_STATE) ? 1U : 74U;
    asr_zh::AsrZhOpProfiler profiler{asr_zh::ReadCycles, conversions};
    backend.m_eventTracer = &profiler;
    asr_zh::g_profiler    = &profiler;
#endif
    fwk::iface::MemoryRegion pool{asr_zh::s_methodPool, sizeof(asr_zh::s_methodPool)};
    fwk::iface::MemoryRegion program{asr_zh::GetModelPointer(), asr_zh::GetModelLen()};
    const bool success = model.Init(pool, program, &backend) && asr_zh::ProcessAudio(model);
    model.LogMemoryUsage();
    info("ASR_ZH completed success=%d\n", success);
#if defined(ASR_ZH_INPUT_VSI)
    if (success) {
        printf("ASR_ZH_VSI_EOS success=1\n");
        fvp_audio::HoldResult();
    }
#endif
#if ASR_ZH_PROFILE
    asr_zh::g_profiler = nullptr;
#endif
}
