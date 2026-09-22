/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/fwk/tflm/YoloV8Model.hpp"

namespace arm::app::fwk::tflm {

const tflite::MicroOpResolver& YoloV8Model::GetOpResolver()
{
    return m_opResolver;
}

bool YoloV8Model::EnlistOperations()
{
    /* Cover both a plain TFLite export and a Vela-lowered Ethos-U model. */
    m_opResolver = CreateAllOpsResolver();
    return true;
}

} /* namespace arm::app::fwk::tflm */
