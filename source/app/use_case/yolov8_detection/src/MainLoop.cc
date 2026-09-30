/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "BufAttributes.hpp"
#include "Labels.hpp"
#include "UseCaseCommonUtils.hpp"
#include "UseCaseHandler.hpp"

#if defined(MLEK_FWK_TFLM)
#    include "mlek/fwk/tflm/YoloV8Model.hpp"
using YoloV8Model = arm::app::fwk::tflm::YoloV8Model;
#elif defined(MLEK_FWK_EXECUTORCH)
#    include "mlek/fwk/executorch/EtModel.hpp"
using YoloV8Model = arm::app::fwk::et::EtModel;
#else
#    error "A supported ML framework must be selected for YOLOv8."
#endif

#include "mlek/log/log_macros.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace arm::app {

#if defined(MLEK_FWK_EXECUTORCH) && YOLOV8_ET_METHOD_MEMORY_IN_DDR && \
    !defined(MLEK_DISABLE_SECTION_ATTRIBUTES)
/* Method memory includes CPU tensors and input storage. The NPU scratch buffer
 * is allocated separately from the ExecuTorch temporary memory pool. */
static uint8_t activationBuf[ACTIVATION_BUF_SZ] MAKE_ATTRIBUTE(section("activation_buf_dram"));
#else
static uint8_t activationBuf[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
#endif

namespace yolov8_detection {
    extern uint8_t* GetModelPointer();
    extern size_t GetModelLen();
} /* namespace yolov8_detection */

} /* namespace arm::app */

void MainLoop()
{
    YoloV8Model model;

    arm::app::fwk::iface::MemoryRegion modelMem{arm::app::yolov8_detection::GetModelPointer(),
                                                arm::app::yolov8_detection::GetModelLen()};
    arm::app::fwk::iface::MemoryRegion computeMem{arm::app::activationBuf,
                                                  sizeof(arm::app::activationBuf)};

    if (!model.Init(computeMem, modelMem)) {
        printf_err("Failed to initialise YOLOv8 model\n");
        return;
    }

    arm::app::ApplicationContext caseContext;
    arm::app::Profiler profiler{"yolov8_detection"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::fwk::iface::Model&>("model", model);

    std::vector<std::string> labels;
    GetLabelsVector(labels);
    caseContext.Set<const std::vector<std::string>&>("labels", labels);

    const bool executionSuccessful = arm::app::YoloV8DetectionHandler(caseContext);
    info("Main loop terminated %s.\n", executionSuccessful ? "successfully" : "with failure");
}
