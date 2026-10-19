/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "DetectionDrawing.hpp"

extern "C" {
void reset_frame_buffer(void);
bool check_pixels(const char* expected);
bool test_detection_drawing(void);
}

namespace {

bool CheckBox(const arm::app::yolov8_detection::DetectionResult& box,
              const char* pixels,
              uint32_t downscale = 1,
              uint32_t startX    = 0,
              uint32_t startY    = 0)
{
    reset_frame_buffer();
    arm::app::DrawDetectionBoxes({box}, startX, startY, 8 * downscale, 6 * downscale, downscale);
    return check_pixels(pixels);
}

} /* anonymous namespace */

bool test_detection_drawing(void)
{
    constexpr float kPi   = 3.14159265358979323846F;
    const char* rectangle = "........"
                            "........"
                            ".ggggg.."
                            ".g...g.."
                            ".ggggg.."
                            "........";
    if (!CheckBox({1, 0, 3, 3, 4, 2, 0}, rectangle) ||
        !CheckBox({1, 0, 6, 6, 8, 4, 0}, rectangle, 2)) {
        return false;
    }
    if (!CheckBox({1, 0, 3, 3, 4, 2, kPi * 0.5F},
                  "........"
                  "..ggg..."
                  "..g.g..."
                  "..g.g..."
                  "..g.g..."
                  "..ggg...")) {
        return false;
    }
    for (float angle : {-kPi * 0.25F, kPi * 0.25F}) {
        if (!CheckBox({1, 0, 4, 3, 2.828427125F, 2.828427125F, angle},
                      "........"
                      "....g..."
                      "...g.g.."
                      "..g...g."
                      "...g.g.."
                      "....g...")) {
            return false;
        }
    }
    /* Offscreen edges are clipped as segments; no artificial vertical edge at x=0. */
    if (!CheckBox({1, 0, -1, 2, 4, 2, 0},
                  "........"
                  "gg......"
                  ".g......"
                  "gg......"
                  "........"
                  "........") ||
        !CheckBox({1, 0, -1, 2, 4, 2, 0},
                  "........"
                  "........"
                  "..gg...."
                  "...g...."
                  "..gg...."
                  "........",
                  1,
                  2,
                  1) ||
        !CheckBox({1, 0, -5, 3, 2, 2, 0},
                  "........"
                  "........"
                  "........"
                  "........"
                  "........"
                  "........")) {
        return false;
    }
    return true;
}
