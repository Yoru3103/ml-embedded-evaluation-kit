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
    int x{0};
    int y{0};
    int width{0};
    int height{0};
    float centerX{0.0F};
    float centerY{0.0F};
    float boxWidth{0.0F};
    float boxHeight{0.0F};
    float angle{0.0F}; /* Radians; x/y/width/height remain the clipped display envelope. */
    bool rotated{false};
};

} /* namespace arm::app::yolov8_detection */

#endif /* YOLOV8_DETECTION_RESULT_HPP */
