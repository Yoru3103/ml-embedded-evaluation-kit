/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/yolov8_pose/YoloV8PosePreProcessing.hpp"

#include "mlek/log/log_macros.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace arm::app {
namespace {

    struct InputDescription {
        fwk::iface::TensorLayout layout{fwk::iface::TensorLayout::INVALID};
        size_t height{0};
        size_t width{0};
    };

    bool GetInputDescription(const std::vector<size_t>& shape, InputDescription& description)
    {
        if (shape.size() != 4 || shape[0] != 1) {
            return false;
        }

        if (shape[1] == 3 && shape[3] != 3) {
            description.height = shape[2];
            description.width  = shape[3];
            return true;
        }

        if (shape[3] == 3 && shape[1] != 3) {
            description.height = shape[1];
            description.width  = shape[2];
            return true;
        }

        return false;
    }

    int8_t QuantizeInput(const uint8_t value, const fwk::iface::QuantParams& quantParams)
    {
        const float realValue     = static_cast<float>(value) / 255.0F;
        const long quantizedValue = std::lround(realValue / quantParams.scale) + quantParams.offset;

        return static_cast<int8_t>(
            std::clamp(quantizedValue, static_cast<long>(-128), static_cast<long>(127)));
    }

    size_t GetDestinationIndex(const size_t pixel,
                               const size_t channel,
                               const size_t planeSize,
                               const fwk::iface::TensorLayout layout)
    {
        return layout == fwk::iface::TensorLayout::NCHW ? channel * planeSize + pixel
                                                        : pixel * 3 + channel;
    }

} /* anonymous namespace */

YoloV8PosePreProcess::YoloV8PosePreProcess(
    const std::shared_ptr<fwk::iface::TensorIface>& inputTensor,
    fwk::iface::TensorLayout inputLayout) : m_inputTensor{inputTensor}, m_inputLayout{inputLayout}
{}

bool YoloV8PosePreProcess::DoPreProcess(const void* input, size_t inputSize)
{
    if (input == nullptr || m_inputTensor == nullptr) {
        printf_err("Invalid YOLOv8 pose input\n");
        return false;
    }

    InputDescription description;
    if (!GetInputDescription(m_inputTensor->Shape(), description)) {
        printf_err("Expected YOLOv8 pose input shape [1, 3, H, W] or [1, H, W, 3]\n");
        return false;
    }

    description.layout = m_inputLayout == fwk::iface::TensorLayout::INVALID
                             ? m_inputTensor->Layout()
                             : m_inputLayout;
    if (description.layout == fwk::iface::TensorLayout::INVALID) {
        const auto shape = m_inputTensor->Shape();
        description.layout =
            shape[1] == 3 ? fwk::iface::TensorLayout::NCHW : fwk::iface::TensorLayout::NHWC;
    }

    const size_t planeSize     = description.height * description.width;
    const size_t requiredBytes = planeSize * 3;
    if (inputSize < requiredBytes) {
        printf_err("YOLOv8 pose input image is too small\n");
        return false;
    }

    const auto tensorType  = m_inputTensor->Type();
    const auto quantParams = m_inputTensor->GetQuantParams();
    if (tensorType == fwk::iface::TensorType::INT8 && quantParams.scale <= 0.0F) {
        printf_err("YOLOv8 pose INT8 input has invalid quantization parameters\n");
        return false;
    }

    if (tensorType != fwk::iface::TensorType::FP32 && tensorType != fwk::iface::TensorType::INT8) {
        printf_err("YOLOv8 pose input tensor must be float32 or int8\n");
        return false;
    }

    const auto* source = static_cast<const uint8_t*>(input);
    if (tensorType == fwk::iface::TensorType::FP32) {
        auto* destination = m_inputTensor->GetData<float>();
        for (size_t pixel = 0; pixel < planeSize; ++pixel) {
            for (size_t channel = 0; channel < 3; ++channel) {
                const size_t destinationIndex =
                    GetDestinationIndex(pixel, channel, planeSize, description.layout);
                destination[destinationIndex] =
                    static_cast<float>(source[pixel * 3 + channel]) / 255.0F;
            }
        }
    } else {
        std::array<int8_t, 256> quantizationTable{};
        for (size_t value = 0; value < quantizationTable.size(); ++value) {
            quantizationTable[value] = QuantizeInput(static_cast<uint8_t>(value), quantParams);
        }

        auto* destination = m_inputTensor->GetData<int8_t>();
        for (size_t pixel = 0; pixel < planeSize; ++pixel) {
            for (size_t channel = 0; channel < 3; ++channel) {
                const size_t sourceIndex = pixel * 3 + channel;
                const size_t destinationIndex =
                    GetDestinationIndex(pixel, channel, planeSize, description.layout);
                destination[destinationIndex] = quantizationTable[source[sourceIndex]];
            }
        }
    }

    return true;
}

} /* namespace arm::app */
