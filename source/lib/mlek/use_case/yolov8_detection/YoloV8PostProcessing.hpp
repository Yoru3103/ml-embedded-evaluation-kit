/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_POST_PROCESSING_HPP
#define YOLOV8_POST_PROCESSING_HPP

#include "mlek/common/BaseProcessing.hpp"
#include "mlek/fwk/iface/Tensor.hpp"
#include "mlek/use_case/yolov8_detection/DetectionResult.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace arm::app {

struct YoloV8PostProcessParams {
    uint32_t imageWidth{0};
    uint32_t imageHeight{0};
    uint32_t numClasses{80};
    uint32_t maxDetections{20};
    float scoreThreshold{0.25F};
    float nmsThreshold{0.45F};
    bool angleIsLogits{true}; /**< Raw angles map to (sigmoid(value) - 0.25) * pi. */
};

class YoloV8PostProcess : public BasePostProcess {
public:
    YoloV8PostProcess(const std::shared_ptr<fwk::iface::TensorIface>& outputTensor,
                      std::vector<yolov8_detection::DetectionResult>& results,
                      const YoloV8PostProcessParams& params);

    bool DoPostProcess() override;

private:
    std::shared_ptr<fwk::iface::TensorIface> m_outputTensor;
    std::vector<yolov8_detection::DetectionResult>& m_results;
    YoloV8PostProcessParams m_params;
};

/**
 * @brief Post-processes the four-output YOLOv8 DFL model.
 *
 * The model has three feature maps containing four DFL distributions followed by
 * class logits, optionally plus one concatenated angle output. The DFL bin count
 * is derived from the feature-map channels and the configured class count.
 */
class YoloV8NewPostProcess : public BasePostProcess {
public:
    YoloV8NewPostProcess(const std::vector<std::shared_ptr<fwk::iface::TensorIface>>& outputTensors,
                         std::vector<yolov8_detection::DetectionResult>& results,
                         const YoloV8PostProcessParams& params);

    bool DoPostProcess() override;

private:
    std::vector<std::shared_ptr<fwk::iface::TensorIface>> m_outputTensors;
    std::vector<yolov8_detection::DetectionResult>& m_results;
    YoloV8PostProcessParams m_params;
};

} /* namespace arm::app */

#endif /* YOLOV8_POST_PROCESSING_HPP */
