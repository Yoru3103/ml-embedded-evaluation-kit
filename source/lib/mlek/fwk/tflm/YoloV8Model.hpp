/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_MODEL_HPP
#define YOLOV8_MODEL_HPP

#include "mlek/fwk/tflm/MicroMutableAllOpsResolver.hpp"
#include "mlek/fwk/tflm/TflmModel.hpp"

namespace arm::app::fwk::tflm {

class YoloV8Model : public TflmModel {
protected:
    /** @brief   Gets the reference to op resolver interface class. */
    const tflite::MicroOpResolver& GetOpResolver() override;

    /** @brief   Adds operations to the op resolver instance. */
    bool EnlistOperations() override;

private:
    /* YOLOv8 exports can contain different builtin operators before Vela lowers them. */
    static constexpr int ms_maxOpCnt = kNumberOperators;

    /* A mutable op resolver instance. */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace arm::app::fwk::tflm */

#endif /* YOLOV8_MODEL_HPP */
