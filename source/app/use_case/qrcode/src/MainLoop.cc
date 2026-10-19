/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "BufAttributes.hpp"
#include "Labels.hpp"
#include "OutputDequantizationProfiler.hpp"
#include "UseCaseCommonUtils.hpp"
#include "UseCaseHandler.hpp"

#if defined(MLEK_FWK_TFLM)
#    include "mlek/fwk/tflm/YoloV8Model.hpp"
using QrCodeModel = arm::app::fwk::tflm::YoloV8Model;
#elif defined(MLEK_FWK_EXECUTORCH)
#    include "mlek/fwk/executorch/EtModel.hpp"
using QrCodeModel = arm::app::fwk::et::EtModel;
#else
#    error "A supported ML framework must be selected for QR-code detection."
#endif

#include "mlek/log/log_macros.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace arm::app {

#if defined(MLEK_FWK_EXECUTORCH) && QRCODE_ET_METHOD_MEMORY_IN_DDR && \
    !defined(MLEK_DISABLE_SECTION_ATTRIBUTES)
static uint8_t activationBuf[ACTIVATION_BUF_SZ] MAKE_ATTRIBUTE(section("activation_buf_dram"));
#else
static uint8_t activationBuf[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
#endif

namespace qrcode {
    extern uint8_t* GetModelPointer();
    extern size_t GetModelLen();
} /* namespace qrcode */

} /* namespace arm::app */

void MainLoop()
{
    arm::app::OutputDequantizationProfiler outputProfiler{arm::app::qrcode::GetModelPointer()};
    QrCodeModel model;
    arm::app::fwk::iface::MemoryRegion modelMem{arm::app::qrcode::GetModelPointer(),
                                                arm::app::qrcode::GetModelLen()};
    arm::app::fwk::iface::MemoryRegion computeMem{arm::app::activationBuf,
                                                  sizeof(arm::app::activationBuf)};
#if defined(MLEK_FWK_TFLM)
    arm::app::fwk::tflm::TflmBackendData backendData;
#    if defined(CPU_PROFILE_ENABLED)
    backendData.m_pProfiler = &outputProfiler;
#    endif
#else
    arm::app::fwk::et::EtBackendData backendData;
#    if defined(CPU_PROFILE_ENABLED)
    backendData.m_eventTracer = &outputProfiler;
#    endif
#endif
    if (!model.Init(computeMem, modelMem, &backendData)) {
        printf_err("Failed to initialise QR-code model\n");
        return;
    }

    arm::app::ApplicationContext context;
    arm::app::Profiler profiler{"qrcode"};
    context.Set<arm::app::Profiler&>("profiler", profiler);
    context.Set<arm::app::fwk::iface::Model&>("model", model);
    context.Set<arm::app::OutputDequantizationProfiler&>("outputProfiler", outputProfiler);

    std::vector<std::string> labels;
    GetLabelsVector(labels);
    context.Set<const std::vector<std::string>&>("labels", labels);

    const bool success = arm::app::QrCodeHandler(context);
    info("Main loop terminated %s.\n", success ? "successfully" : "with failure");
}
