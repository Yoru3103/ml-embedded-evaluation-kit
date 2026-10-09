/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "BufAttributes.hpp"
#include "Labels.hpp"
#include "UseCaseCommonUtils.hpp"
#include "UseCaseHandler.hpp"
#include "mlek/log/log_macros.h"

#if defined(MLEK_FWK_TFLM)
#    include "mlek/fwk/tflm/YoloV8Model.hpp"
#elif defined(MLEK_FWK_EXECUTORCH)
#    include "mlek/fwk/executorch/EtModel.hpp"
#else
#    error "Unsupported ML framework for YOLOv8 pose"
#endif

namespace arm::app {

static uint8_t activationBuffer[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

namespace yolov8_pose {
    extern uint8_t* GetModelPointer();
    extern size_t GetModelLen();
} /* namespace yolov8_pose */

} /* namespace arm::app */

#if defined(MLEK_FWK_TFLM)
using YoloV8PoseModel = arm::app::fwk::tflm::YoloV8Model;
#elif defined(MLEK_FWK_EXECUTORCH)
using YoloV8PoseModel = arm::app::fwk::et::EtModel;
#endif

void MainLoop()
{
    YoloV8PoseModel model;

    arm::app::fwk::iface::MemoryRegion modelMemory{arm::app::yolov8_pose::GetModelPointer(),
                                                   arm::app::yolov8_pose::GetModelLen()};
    arm::app::fwk::iface::MemoryRegion computeMemory{arm::app::activationBuffer,
                                                     sizeof(arm::app::activationBuffer)};

    if (!model.Init(computeMemory, modelMemory)) {
        printf_err("Failed to initialise YOLOv8 pose model\n");
        return;
    }

#if defined(PERFORMANCE_DETAILED_INFO_ENABLED)
    model.LogInterpreterInfo();
#endif

    arm::app::ApplicationContext context;
    arm::app::Profiler profiler{"yolov8_pose"};
    context.Set<arm::app::Profiler&>("profiler", profiler);
    context.Set<arm::app::fwk::iface::Model&>("model", model);

    std::vector<std::string> labels;
    GetLabelsVector(labels);
    context.Set<const std::vector<std::string>&>("labels", labels);

    const bool successful = arm::app::YoloV8PoseHandler(context);
    info("YOLOv8 pose use case terminated %s\n", successful ? "successfully" : "with failure");
}
