/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_PRE_PROCESSING_HPP
#define YOLOV8_PRE_PROCESSING_HPP

#include "mlek/common/BaseProcessing.hpp"
#include "mlek/fwk/iface/Tensor.hpp"

#include <array>
#include <memory>

namespace arm::app {

class YoloV8PreProcess : public BasePreProcess {
public:
    /** Use explicit memory layouts for ExecuTorch, shape inference for TFLite.
     * inputIsGrayscale declares that the RGB sample bytes have equal R/G/B components. */
    explicit YoloV8PreProcess(const std::shared_ptr<fwk::iface::TensorIface>& inputTensor,
                              bool useTensorLayout  = false,
                              bool inputIsGrayscale = false);

    bool DoPreProcess(const void* input, size_t inputSize) override;

private:
    std::shared_ptr<fwk::iface::TensorIface> m_inputTensor;
    bool m_useTensorLayout;
    bool m_inputIsGrayscale;
    std::array<uint8_t, 256> m_lookup{};
    fwk::iface::QuantParams m_quant{};
    fwk::iface::TensorType m_type{fwk::iface::TensorType::INVALID};
    bool m_lookupReady{false};
};

} /* namespace arm::app */

#endif /* YOLOV8_PRE_PROCESSING_HPP */
