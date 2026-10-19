/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/yolov8_detection/YoloV8PreProcessing.hpp"
#include "mlek/log/log_macros.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace arm::app {
namespace {

    struct InputDescription {
        fwk::iface::TensorLayout layout{fwk::iface::TensorLayout::INVALID};
        size_t height{0};
        size_t width{0};
        size_t channels{0};
    };

    bool GetInputDescription(const std::vector<size_t>& shape, InputDescription& description)
    {
        if (shape.size() != 4 || shape[0] != 1 || shape[1] == 0 || shape[2] == 0 || shape[3] == 0) {
            return false;
        }

        /* Infer RGB or grayscale channel placement when no layout annotation is available. */
        if ((shape[1] == 3 && shape[3] != 3) || (shape[1] == 1 && shape[3] != 1 && shape[3] != 3)) {
            description.layout   = fwk::iface::TensorLayout::NCHW;
            description.height   = shape[2];
            description.width    = shape[3];
            description.channels = shape[1];
            return true;
        }

        if ((shape[3] == 3 && shape[1] != 3) || (shape[3] == 1 && shape[1] != 1 && shape[1] != 3)) {
            description.layout   = fwk::iface::TensorLayout::NHWC;
            description.height   = shape[1];
            description.width    = shape[2];
            description.channels = shape[3];
            return true;
        }

        return false;
    }

    long QuantizeInput(const uint8_t value, const fwk::iface::QuantParams& quantParams)
    {
        const float realValue = static_cast<float>(value) / 255.0F;
        return std::lround(realValue / quantParams.scale) + quantParams.offset;
    }

} /* anonymous namespace */

YoloV8PreProcess::YoloV8PreProcess(const std::shared_ptr<fwk::iface::TensorIface>& inputTensor,
                                   bool useTensorLayout,
                                   bool inputIsGrayscale) :
    m_inputTensor{inputTensor}, m_useTensorLayout{useTensorLayout},
    m_inputIsGrayscale{inputIsGrayscale}
{}

bool YoloV8PreProcess::DoPreProcess(const void* input, size_t inputSize)
{
    if (input == nullptr || m_inputTensor == nullptr) {
        printf_err("Invalid YOLOv8 input\n");
        return false;
    }

    InputDescription description;
    if (!GetInputDescription(m_inputTensor->Shape(), description)) {
        printf_err("Expected YOLOv8 RGB or grayscale NCHW/NHWC input\n");
        return false;
    }

    /* ExecuTorch keeps NCHW logical sizes for channels-last tensors. */
    const auto tensorLayout = m_inputTensor->Layout();
    if (m_useTensorLayout && tensorLayout != fwk::iface::TensorLayout::INVALID) {
        description.layout = tensorLayout;
    }

    const size_t planeSize     = description.height * description.width;
    const size_t requiredBytes = planeSize * 3;
    if (inputSize < requiredBytes) {
        printf_err("Input image is too small\n");
        return false;
    }

    const auto* source     = static_cast<const uint8_t*>(input);
    const auto tensorType  = m_inputTensor->Type();
    const auto quantParams = m_inputTensor->GetQuantParams();

    const bool isQuantized =
        tensorType == fwk::iface::TensorType::INT8 || tensorType == fwk::iface::TensorType::UINT8;
    if (isQuantized && (!std::isfinite(quantParams.scale) || quantParams.scale <= 0.0F)) {
        printf_err("YOLOv8 quantized input has invalid quantization parameters\n");
        return false;
    }

    if (tensorType != fwk::iface::TensorType::FP32 && !isQuantized) {
        printf_err("YOLOv8 input tensor must be float32, int8 or uint8\n");
        return false;
    }

    const bool directInt8 = tensorType == fwk::iface::TensorType::INT8 &&
                            quantParams.scale == 1.0F / 255.0F && quantParams.offset == -128;
    if (m_inputIsGrayscale && description.channels == 1 && directInt8) {
        auto* destination = m_inputTensor->GetData<int8_t>();
        if (destination == nullptr) {
            return false;
        }
        // A single channel has identical physical indexing in NCHW and NHWC.
        // The caller declares R == G == B; retain RGB storage for the LCD.
        for (size_t pixel = 0; pixel < planeSize; ++pixel) {
            destination[pixel] = static_cast<int8_t>(static_cast<int>(source[pixel * 3]) - 128);
        }
        return true;
    }
    if (isQuantized && !directInt8 &&
        (!m_lookupReady || m_type != tensorType || m_quant.scale != quantParams.scale ||
         m_quant.offset != quantParams.offset)) {
        const long minimum = tensorType == fwk::iface::TensorType::INT8 ? -128 : 0;
        const long maximum = tensorType == fwk::iface::TensorType::INT8 ? 127 : 255;
        for (size_t value = 0; value < m_lookup.size(); ++value) {
            m_lookup[value] = static_cast<uint8_t>(
                std::clamp(
                    QuantizeInput(static_cast<uint8_t>(value), quantParams), minimum, maximum) -
                minimum);
        }
        m_quant       = quantParams;
        m_type        = tensorType;
        m_lookupReady = true;
    }
    auto* int8Input =
        tensorType == fwk::iface::TensorType::INT8 ? m_inputTensor->GetData<int8_t>() : nullptr;

    for (size_t pixel = 0; pixel < planeSize; ++pixel) {
        /* Keep RGB storage for the LCD; match Pillow RGB-to-L rounding for grayscale. */
        for (size_t channel = 0; channel < description.channels; ++channel) {
            const uint8_t sourceValue =
                description.channels == 1 && !m_inputIsGrayscale
                    ? static_cast<uint8_t>((19595U * source[pixel * 3] +
                                            38470U * source[pixel * 3 + 1] +
                                            7471U * source[pixel * 3 + 2] + 32768U) >>
                                           16)
                    : source[pixel * 3 + channel];
            const size_t destinationIndex = description.layout == fwk::iface::TensorLayout::NCHW
                                                ? channel * planeSize + pixel
                                                : pixel * description.channels + channel;

            if (tensorType == fwk::iface::TensorType::FP32) {
                m_inputTensor->GetData<float>()[destinationIndex] =
                    static_cast<float>(sourceValue) / 255.0F;
            } else if (tensorType == fwk::iface::TensorType::INT8) {
                int8Input[destinationIndex] =
                    static_cast<int8_t>(directInt8 ? static_cast<int>(sourceValue) - 128
                                                   : static_cast<int>(m_lookup[sourceValue]) - 128);
            } else {
                m_inputTensor->GetData<uint8_t>()[destinationIndex] = m_lookup[sourceValue];
            }
        }
    }

    return true;
}
} /* namespace arm::app */
