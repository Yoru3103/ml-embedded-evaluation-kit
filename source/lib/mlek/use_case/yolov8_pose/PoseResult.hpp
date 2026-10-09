/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_POSE_RESULT_HPP
#define YOLOV8_POSE_RESULT_HPP

#include <array>
#include <cstddef>

namespace arm::app::yolov8_pose {

constexpr size_t kNumKeypoints = 21;

struct Keypoint {
    float x{0.0F};
    float y{0.0F};
    float score{0.0F};
};

struct PoseResult {
    float score{0.0F};
    int classId{-1};
    int x{0};
    int y{0};
    int width{0};
    int height{0};
    std::array<Keypoint, kNumKeypoints> keypoints{};
};

} /* namespace arm::app::yolov8_pose */

#endif /* YOLOV8_POSE_RESULT_HPP */
