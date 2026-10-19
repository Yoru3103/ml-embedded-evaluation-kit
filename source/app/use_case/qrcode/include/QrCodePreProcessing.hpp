/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef QRCODE_PRE_PROCESSING_HPP
#define QRCODE_PRE_PROCESSING_HPP

#include "mlek/fwk/iface/Tensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>

namespace arm::app {

/** Map grayscale BMP bytes to the model's NHWC 256x256 input tensor. */
class QrCodePreProcess {
public:
    explicit QrCodePreProcess(const std::shared_ptr<fwk::iface::TensorIface>& tensor) :
        m_tensor{tensor}
    {}

    bool DoPreProcess(const uint8_t* image, size_t imageBytes)
    {
        if (m_tensor == nullptr || image == nullptr || imageBytes != 256 * 256 ||
            m_tensor->Shape() != std::vector<size_t>{1, 256, 256, 1} ||
            m_tensor->GetData() == nullptr) {
            return false;
        }
        const auto type = m_tensor->Type();
        if (type == fwk::iface::TensorType::FP32) {
            auto* destination = m_tensor->GetData<float>();
            for (size_t pixel = 0; pixel < imageBytes; ++pixel) {
                destination[pixel] = static_cast<float>(image[pixel]) / 255.0F;
            }
            return true;
        }
        if (type != fwk::iface::TensorType::INT8 && type != fwk::iface::TensorType::UINT8) {
            return false;
        }
        const auto quant = m_tensor->GetQuantParams();
        if (!std::isfinite(quant.scale) || quant.scale <= 0.0F) {
            return false;
        }
        if (!m_ready || m_type != type || m_quant.scale != quant.scale ||
            m_quant.offset != quant.offset) {
            const long minimum = type == fwk::iface::TensorType::INT8 ? -128 : 0;
            const long maximum = type == fwk::iface::TensorType::INT8 ? 127 : 255;
            for (size_t pixel = 0; pixel < m_lookup.size(); ++pixel) {
                // Keep the original float operation order and lround tie handling.
                const long value =
                    std::lround((static_cast<float>(pixel) / 255.0F) / quant.scale) + quant.offset;
                m_lookup[pixel] =
                    static_cast<uint8_t>(std::clamp(value, minimum, maximum) - minimum);
            }
            m_quant = quant;
            m_type  = type;
            m_ready = true;
        }
        if (type == fwk::iface::TensorType::INT8) {
            auto* destination = m_tensor->GetData<int8_t>();
            for (size_t pixel = 0; pixel < imageBytes; ++pixel) {
                destination[pixel] =
                    static_cast<int8_t>(static_cast<int>(m_lookup[image[pixel]]) - 128);
            }
        } else {
            auto* destination = m_tensor->GetData<uint8_t>();
            for (size_t pixel = 0; pixel < imageBytes; ++pixel) {
                destination[pixel] = m_lookup[image[pixel]];
            }
        }
        return true;
    }

private:
    std::shared_ptr<fwk::iface::TensorIface> m_tensor;
    std::array<uint8_t, 256> m_lookup{};
    fwk::iface::QuantParams m_quant;
    fwk::iface::TensorType m_type{fwk::iface::TensorType::INVALID};
    bool m_ready{false};
};

} /* namespace arm::app */

#endif /* QRCODE_PRE_PROCESSING_HPP */
