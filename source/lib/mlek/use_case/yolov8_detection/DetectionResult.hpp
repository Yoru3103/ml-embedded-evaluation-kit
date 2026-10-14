/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_DETECTION_RESULT_HPP
#define YOLOV8_DETECTION_RESULT_HPP

namespace arm::app::yolov8_detection {

struct DetectionResult {
    float score{0.0F};
    int classId{-1};
    /* Geometry in model pixels; angle zero describes an ordinary rectangle. */
    float centerX{0.0F};
    float centerY{0.0F};
    float width{0.0F};
    float height{0.0F};
    float angleRadians{0.0F};
};

} /* namespace arm::app::yolov8_detection */

#endif /* YOLOV8_DETECTION_RESULT_HPP */
