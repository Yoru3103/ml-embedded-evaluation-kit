/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "DetectionDrawing.hpp"

#include "hal_display.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace arm::app {
namespace {

    using Point = std::array<float, 2>;

    void DrawLine(const Point& first, const Point& second, const std::array<float, 4>& bounds)
    {
        const float dx = second[0] - first[0];
        const float dy = second[1] - first[1];
        const std::array<float, 4> directions{-dx, dx, -dy, dy};
        const std::array<float, 4> distances{
            first[0] - bounds[0], bounds[2] - first[0], first[1] - bounds[1], bounds[3] - first[1]};
        float enter = 0.0F;
        float leave = 1.0F;

        /* Clip the segment before rasterization; do not clamp individual corners. */
        for (size_t side = 0; side < directions.size(); ++side) {
            if (directions[side] == 0.0F) {
                if (distances[side] < 0.0F) {
                    return;
                }
            } else {
                const float ratio = distances[side] / directions[side];
                if (directions[side] < 0.0F) {
                    enter = std::max(enter, ratio);
                } else {
                    leave = std::min(leave, ratio);
                }
                if (enter > leave) {
                    return;
                }
            }
        }

        int x            = std::clamp(static_cast<int>(std::lround(first[0] + enter * dx)),
                                      static_cast<int>(bounds[0]),
                                      static_cast<int>(bounds[2]));
        int y            = std::clamp(static_cast<int>(std::lround(first[1] + enter * dy)),
                                      static_cast<int>(bounds[1]),
                                      static_cast<int>(bounds[3]));
        const int endX   = std::clamp(static_cast<int>(std::lround(first[0] + leave * dx)),
                                      static_cast<int>(bounds[0]),
                                      static_cast<int>(bounds[2]));
        const int endY   = std::clamp(static_cast<int>(std::lround(first[1] + leave * dy)),
                                      static_cast<int>(bounds[1]),
                                      static_cast<int>(bounds[3]));
        const int width  = std::abs(endX - x);
        const int height = -std::abs(endY - y);
        const int stepX  = x < endX ? 1 : -1;
        const int stepY  = y < endY ? 1 : -1;
        int error        = width + height;

        for (;;) {
            /* The HAL publishes each pixel to the HDLCD with its cache maintenance. */
            hal_display_show_box(static_cast<uint32_t>(x),
                                 static_cast<uint32_t>(y),
                                 1,
                                 1,
                                 0x07e0); /* Green RGB565. */
            if (x == endX && y == endY) {
                break;
            }
            const int twiceError = 2 * error;
            if (twiceError >= height) {
                error += height;
                x += stepX;
            }
            if (twiceError <= width) {
                error += width;
                y += stepY;
            }
        }
    }

} /* anonymous namespace */

void DrawDetectionBoxes(const std::vector<yolov8_detection::DetectionResult>& results,
                        uint32_t startX,
                        uint32_t startY,
                        uint32_t imageWidth,
                        uint32_t imageHeight,
                        uint32_t downscale)
{
    const uint32_t screenWidth  = hal_display_get_width();
    const uint32_t screenHeight = hal_display_get_height();
    if (downscale == 0 || startX >= screenWidth || startY >= screenHeight) {
        return;
    }
    const uint32_t width  = std::min(imageWidth / downscale, screenWidth - startX);
    const uint32_t height = std::min(imageHeight / downscale, screenHeight - startY);
    if (width == 0 || height == 0) {
        return;
    }
    const std::array<float, 4> bounds{static_cast<float>(startX),
                                      static_cast<float>(startY),
                                      static_cast<float>(startX + width - 1),
                                      static_cast<float>(startY + height - 1)};

    for (const auto& result : results) {
        if (!std::isfinite(result.centerX) || !std::isfinite(result.centerY) ||
            !std::isfinite(result.width) || !std::isfinite(result.height) ||
            !std::isfinite(result.angleRadians) || result.width <= 0.0F || result.height <= 0.0F) {
            continue;
        }
        const float cosine = std::cos(result.angleRadians);
        const float sine   = std::sin(result.angleRadians);
        std::array<Point, 4> corners{{{-result.width * 0.5F, -result.height * 0.5F},
                                      {result.width * 0.5F, -result.height * 0.5F},
                                      {result.width * 0.5F, result.height * 0.5F},
                                      {-result.width * 0.5F, result.height * 0.5F}}};
        for (auto& corner : corners) {
            const Point offset = corner;
            corner = {startX + (result.centerX + offset[0] * cosine - offset[1] * sine) / downscale,
                      startY +
                          (result.centerY + offset[0] * sine + offset[1] * cosine) / downscale};
        }
        for (size_t edge = 0; edge < corners.size(); ++edge) {
            DrawLine(corners[edge], corners[(edge + 1) % corners.size()], bounds);
        }
    }
}

} /* namespace arm::app */
