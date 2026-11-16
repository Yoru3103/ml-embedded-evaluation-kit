/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_DETECTION_DRAWING_HPP
#define YOLOV8_DETECTION_DRAWING_HPP

#include "mlek/use_case/yolov8_detection/DetectionResult.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace arm::app {

/** Shared RGB565 colour for a frame-local result index. */
uint16_t GetDetectionColor(size_t index);

/** Draw coloured edges and nonoverlapping numbered tags where space permits. */
void DrawDetectionBoxes(const std::vector<yolov8_detection::DetectionResult>& results,
                        uint32_t startX,
                        uint32_t startY,
                        uint32_t imageWidth,
                        uint32_t imageHeight,
                        uint32_t downscale);

} /* namespace arm::app */

#endif /* YOLOV8_DETECTION_DRAWING_HPP */
