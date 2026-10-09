/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_POSE_POST_PROCESSING_HPP
#define YOLOV8_POSE_POST_PROCESSING_HPP

#include "mlek/common/BaseProcessing.hpp"
#include "mlek/fwk/iface/Tensor.hpp"
#include "mlek/use_case/yolov8_pose/PoseResult.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace arm::app {

struct YoloV8PosePostProcessParams {
    uint32_t imageWidth{0};
    uint32_t imageHeight{0};
    uint32_t numClasses{1};
    uint32_t maxDetections{20};
    float scoreThreshold{0.25F};
    float nmsThreshold{0.45F};
    bool coordinatesNormalized{true};
};

/** Validate the packed or split FP32 output tensor contract. */
bool ValidateYoloV8PoseOutputTensors(
    const std::vector<std::shared_ptr<fwk::iface::TensorIface>>& outputTensors, size_t numClasses);

/** Decodes either a combined TFLite tensor or four split ExecuTorch tensors. */
class YoloV8PosePostProcess : public BasePostProcess {
public:
    YoloV8PosePostProcess(const std::shared_ptr<fwk::iface::TensorIface>& outputTensor,
                          std::vector<yolov8_pose::PoseResult>& results,
                          const YoloV8PosePostProcessParams& params);

    YoloV8PosePostProcess(
        const std::vector<std::shared_ptr<fwk::iface::TensorIface>>& outputTensors,
        std::vector<yolov8_pose::PoseResult>& results,
        const YoloV8PosePostProcessParams& params);

    bool DoPostProcess() override;

private:
    std::vector<std::shared_ptr<fwk::iface::TensorIface>> m_outputTensors;
    std::vector<yolov8_pose::PoseResult>& m_results;
    YoloV8PosePostProcessParams m_params;
};

} /* namespace arm::app */

#endif /* YOLOV8_POSE_POST_PROCESSING_HPP */
