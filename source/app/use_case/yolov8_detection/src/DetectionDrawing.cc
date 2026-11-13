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
#include <cstdio>
#include <limits>

namespace arm::app {
namespace {

    using Point = std::array<float, 2>;

    void DrawLine(const Point& first,
                  const Point& second,
                  const std::array<float, 4>& bounds,
                  uint16_t color)
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
            hal_display_show_box(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 1, 1, color);
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

    struct NumberTag {
        uint32_t x;
        uint32_t y;
        uint32_t width;
        Point anchor;
        size_t index;
    };

    constexpr uint32_t kTagHeight = 13;

    bool Overlaps(const NumberTag& first, const NumberTag& second)
    {
        return first.x < second.x + second.width + 2 && second.x < first.x + first.width + 2 &&
               first.y < second.y + kTagHeight + 2 && second.y < first.y + kTagHeight + 2;
    }

    void DrawNumberTag(const NumberTag& tag)
    {
        // Pixel glyphs avoid the HAL text renderer's 9x15 cell snapping.
        constexpr std::array<uint16_t, 10> kDigits{
            0x7b6f, 0x2492, 0x73e7, 0x73cf, 0x5bc9, 0x79cf, 0x79ef, 0x7249, 0x7bef, 0x7bcf};
        char number[24]{};
        const int length     = std::snprintf(number, sizeof(number), "%zu", tag.index);
        const uint16_t color = GetDetectionColor(tag.index);
        hal_display_show_box(tag.x, tag.y, tag.width, kTagHeight, 0);
        hal_display_show_box(tag.x, tag.y, tag.width, 1, color);
        hal_display_show_box(tag.x, tag.y + kTagHeight - 1, tag.width, 1, color);
        hal_display_show_box(tag.x, tag.y, 1, kTagHeight, color);
        hal_display_show_box(tag.x + tag.width - 1, tag.y, 1, kTagHeight, color);
        for (int digit = 0; digit < length; ++digit) {
            const uint16_t glyph = kDigits[number[digit] - '0'];
            for (uint32_t row = 0; row < 5; ++row) {
                for (uint32_t column = 0; column < 3; ++column) {
                    if ((glyph & (1U << (14 - row * 3 - column))) != 0) {
                        hal_display_show_box(
                            tag.x + 2 + digit * 8 + column * 2, tag.y + 2 + row * 2, 2, 2, color);
                    }
                }
            }
        }
    }

} /* anonymous namespace */

uint16_t GetDetectionColor(size_t index)
{
    constexpr std::array<uint16_t, 10> kColors{
        0x07e0, 0xffe0, 0x07ff, 0xf81f, 0xfd20, 0xffff, 0xf800, 0x867f, 0xafec, 0xfbb7};
    return kColors[index % kColors.size()];
}

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

    std::vector<NumberTag> tags;
    tags.reserve(results.size());
    for (size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
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
            DrawLine(corners[edge],
                     corners[(edge + 1) % corners.size()],
                     bounds,
                     GetDetectionColor(index));
        }
        char number[24]{};
        const int digits        = std::snprintf(number, sizeof(number), "%zu", index);
        const uint32_t tagWidth = static_cast<uint32_t>(digits) * 8 + 2;
        if (tagWidth > width || kTagHeight > height) {
            continue;
        }
        const Point anchor{std::clamp(corners[0][0], bounds[0], bounds[2]),
                           std::clamp(corners[0][1], bounds[1], bounds[3])};
        NumberTag best{startX, startY, tagWidth, anchor, index};
        float bestDistance = std::numeric_limits<float>::infinity();
        // Search the visible image for the closest free label position.
        for (uint32_t y = startY; y <= startY + height - kTagHeight; y += 2) {
            for (uint32_t x = startX; x <= startX + width - tagWidth; x += 2) {
                NumberTag candidate{x, y, tagWidth, anchor, index};
                if (std::any_of(tags.begin(), tags.end(), [&](const NumberTag& placed) {
                        return Overlaps(candidate, placed);
                    })) {
                    continue;
                }
                const float dx       = static_cast<float>(x) - anchor[0];
                const float dy       = static_cast<float>(y) - anchor[1];
                const float distance = dx * dx + dy * dy;
                if (distance < bestDistance) {
                    best         = candidate;
                    bestDistance = distance;
                }
            }
        }
        if (std::isfinite(bestDistance)) {
            tags.push_back(best);
        }
    }
    // Draw every leader before the tags so no later box obscures a number.
    for (const auto& tag : tags) {
        DrawLine(tag.anchor,
                 Point{static_cast<float>(tag.x + tag.width / 2),
                       static_cast<float>(tag.y + kTagHeight / 2)},
                 bounds,
                 GetDetectionColor(tag.index));
    }
    for (const auto& tag : tags) {
        DrawNumberTag(tag);
    }
}

} /* namespace arm::app */
