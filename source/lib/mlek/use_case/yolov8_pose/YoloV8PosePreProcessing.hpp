/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_POSE_PRE_PROCESSING_HPP
#define YOLOV8_POSE_PRE_PROCESSING_HPP

#include "mlek/common/BaseProcessing.hpp"
#include "mlek/fwk/iface/Tensor.hpp"

#include <memory>

namespace arm::app {

/** Converts interleaved uint8 RGB images to a YOLOv8 pose input tensor. */
class YoloV8PosePreProcess : public BasePreProcess {
public:
    explicit YoloV8PosePreProcess(
        const std::shared_ptr<fwk::iface::TensorIface>& inputTensor,
        fwk::iface::TensorLayout inputLayout = fwk::iface::TensorLayout::INVALID);

    bool DoPreProcess(const void* input, size_t inputSize) override;

private:
    std::shared_ptr<fwk::iface::TensorIface> m_inputTensor;
    fwk::iface::TensorLayout m_inputLayout;
};

} /* namespace arm::app */

#endif /* YOLOV8_POSE_PRE_PROCESSING_HPP */
