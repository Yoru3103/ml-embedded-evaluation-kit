/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/yolov8_pose/YoloV8PosePostProcessing.hpp"

#include "mlek/log/log_macros.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace arm::app {
namespace {

    struct PoseOutputView {
        const float* combined{nullptr};
        const float* boxes{nullptr};
        const float* scores{nullptr};
        const float* keypointCoordinates{nullptr};
        const float* keypointScores{nullptr};
        size_t boxCount{0};
        size_t numClasses{0};

        float Box(size_t component, size_t box) const
        {
            const float* data = combined != nullptr ? combined : boxes;
            return data[component * boxCount + box];
        }

        float Score(size_t classIndex, size_t box) const
        {
            if (combined != nullptr) {
                return combined[(4 + classIndex) * boxCount + box];
            }
            return scores[classIndex * boxCount + box];
        }

        float KeypointCoordinate(size_t keypoint, size_t axis, size_t box) const
        {
            if (combined != nullptr) {
                const size_t attribute = 4 + numClasses + keypoint * 3 + axis;
                return combined[attribute * boxCount + box];
            }
            return keypointCoordinates[(keypoint * 2 + axis) * boxCount + box];
        }

        float KeypointScore(size_t keypoint, size_t box) const
        {
            if (combined != nullptr) {
                const size_t attribute = 4 + numClasses + keypoint * 3 + 2;
                return combined[attribute * boxCount + box];
            }
            return keypointScores[keypoint * boxCount + box];
        }
    };

    bool HasShape(const std::shared_ptr<fwk::iface::TensorIface>& tensor,
                  size_t attributes,
                  size_t boxCount)
    {
        if (tensor == nullptr || tensor->Type() != fwk::iface::TensorType::FP32 ||
            attributes == 0 || boxCount == 0) {
            return false;
        }
        const auto shape = tensor->Shape();
        return shape.size() == 3 && shape[0] == 1 && shape[1] == attributes &&
               shape[2] == boxCount && tensor->Bytes() == attributes * boxCount * sizeof(float);
    }

    bool MakeOutputView(const std::vector<std::shared_ptr<fwk::iface::TensorIface>>& tensors,
                        size_t numClasses,
                        PoseOutputView& view)
    {
        if (numClasses == 0) {
            return false;
        }
        if (tensors.size() == 1 && tensors[0] != nullptr) {
            const auto shape        = tensors[0]->Shape();
            const size_t attributes = 4 + numClasses + yolov8_pose::kNumKeypoints * 3;
            if (!HasShape(tensors[0], attributes, shape.size() == 3 ? shape[2] : 0)) {
                return false;
            }
            view.combined   = tensors[0]->GetData<float>();
            view.boxCount   = shape[2];
            view.numClasses = numClasses;
            return true;
        }

        if (tensors.size() != 4 || tensors[0] == nullptr) {
            return false;
        }
        const auto boxShape = tensors[0]->Shape();
        if (boxShape.size() != 3 || boxShape[0] != 1 || boxShape[1] != 4) {
            return false;
        }
        const size_t boxCount = boxShape[2];
        if (!HasShape(tensors[0], 4, boxCount) || !HasShape(tensors[1], numClasses, boxCount) ||
            !HasShape(tensors[2], yolov8_pose::kNumKeypoints * 2, boxCount) ||
            !HasShape(tensors[3], yolov8_pose::kNumKeypoints, boxCount)) {
            return false;
        }

        view.boxes               = tensors[0]->GetData<float>();
        view.scores              = tensors[1]->GetData<float>();
        view.keypointCoordinates = tensors[2]->GetData<float>();
        view.keypointScores      = tensors[3]->GetData<float>();
        view.boxCount            = boxCount;
        view.numClasses          = numClasses;
        return true;
    }

    float CalculateIoU(const yolov8_pose::PoseResult& first, const yolov8_pose::PoseResult& second)
    {
        const int intersectionLeft   = std::max(first.x, second.x);
        const int intersectionTop    = std::max(first.y, second.y);
        const int intersectionRight  = std::min(first.x + first.width, second.x + second.width);
        const int intersectionBottom = std::min(first.y + first.height, second.y + second.height);
        const int intersectionWidth  = std::max(0, intersectionRight - intersectionLeft);
        const int intersectionHeight = std::max(0, intersectionBottom - intersectionTop);
        const float intersectionArea = static_cast<float>(intersectionWidth * intersectionHeight);
        const float firstArea        = static_cast<float>(first.width * first.height);
        const float secondArea       = static_cast<float>(second.width * second.height);
        const float unionArea        = firstArea + secondArea - intersectionArea;

        return unionArea > 0.0F ? intersectionArea / unionArea : 0.0F;
    }

    float ToImageCoordinate(float value, uint32_t dimension, bool normalized)
    {
        return normalized ? value * static_cast<float>(dimension) : value;
    }

    void RetainCandidate(std::vector<yolov8_pose::PoseResult>& candidates,
                         yolov8_pose::PoseResult&& candidate,
                         size_t candidateLimit)
    {
        if (candidates.size() < candidateLimit) {
            candidates.push_back(std::move(candidate));
            return;
        }

        const auto lowest = std::min_element(
            candidates.begin(), candidates.end(), [](const auto& first, const auto& second) {
                return first.score < second.score;
            });
        if (lowest != candidates.end() && candidate.score > lowest->score) {
            *lowest = std::move(candidate);
        }
    }

    void ApplyClassAwareNms(std::vector<yolov8_pose::PoseResult>& candidates,
                            std::vector<yolov8_pose::PoseResult>& results,
                            const YoloV8PosePostProcessParams& params)
    {
        std::sort(candidates.begin(), candidates.end(), [](const auto& first, const auto& second) {
            return first.score > second.score;
        });

        for (const auto& candidate : candidates) {
            const bool suppressed = std::any_of(
                results.begin(), results.end(), [&candidate, &params](const auto& selected) {
                    return candidate.classId == selected.classId &&
                           CalculateIoU(candidate, selected) > params.nmsThreshold;
                });
            if (!suppressed) {
                results.push_back(candidate);
                if (results.size() >= params.maxDetections) {
                    break;
                }
            }
        }
    }

} /* anonymous namespace */

bool ValidateYoloV8PoseOutputTensors(
    const std::vector<std::shared_ptr<fwk::iface::TensorIface>>& outputTensors, size_t numClasses)
{
    PoseOutputView output;
    return MakeOutputView(outputTensors, numClasses, output);
}

YoloV8PosePostProcess::YoloV8PosePostProcess(
    const std::shared_ptr<fwk::iface::TensorIface>& outputTensor,
    std::vector<yolov8_pose::PoseResult>& results,
    const YoloV8PosePostProcessParams& params) :
    m_outputTensors{outputTensor}, m_results{results}, m_params{params}
{}

YoloV8PosePostProcess::YoloV8PosePostProcess(
    const std::vector<std::shared_ptr<fwk::iface::TensorIface>>& outputTensors,
    std::vector<yolov8_pose::PoseResult>& results,
    const YoloV8PosePostProcessParams& params) :
    m_outputTensors{outputTensors}, m_results{results}, m_params{params}
{}

bool YoloV8PosePostProcess::DoPostProcess()
{
    m_results.clear();

    if (m_params.imageWidth == 0 || m_params.imageHeight == 0 || m_params.numClasses == 0 ||
        m_params.maxDetections == 0) {
        printf_err("Invalid YOLOv8 pose post-processing parameters\n");
        return false;
    }

    PoseOutputView output;
    if (!MakeOutputView(m_outputTensors, m_params.numClasses, output)) {
        printf_err("Unexpected YOLOv8 pose output tensor contract\n");
        return false;
    }

    const size_t boxCount       = output.boxCount;
    const size_t candidateLimit = std::max<size_t>(m_params.maxDetections * 4, 1);
    std::vector<yolov8_pose::PoseResult> candidates;
    candidates.reserve(candidateLimit);

    for (size_t box = 0; box < boxCount; ++box) {
        float bestScore = 0.0F;
        int bestClass   = -1;
        for (size_t classIndex = 0; classIndex < m_params.numClasses; ++classIndex) {
            const float score = output.Score(classIndex, box);
            if (score > bestScore) {
                bestScore = score;
                bestClass = static_cast<int>(classIndex);
            }
        }

        if (bestClass < 0 || bestScore < m_params.scoreThreshold) {
            continue;
        }

        const float centerX = ToImageCoordinate(
            output.Box(0, box), m_params.imageWidth, m_params.coordinatesNormalized);
        const float centerY = ToImageCoordinate(
            output.Box(1, box), m_params.imageHeight, m_params.coordinatesNormalized);
        const float boxWidth = ToImageCoordinate(
            output.Box(2, box), m_params.imageWidth, m_params.coordinatesNormalized);
        const float boxHeight = ToImageCoordinate(
            output.Box(3, box), m_params.imageHeight, m_params.coordinatesNormalized);

        const int left   = std::clamp(static_cast<int>(std::lround(centerX - boxWidth * 0.5F)),
                                      0,
                                      static_cast<int>(m_params.imageWidth) - 1);
        const int top    = std::clamp(static_cast<int>(std::lround(centerY - boxHeight * 0.5F)),
                                      0,
                                      static_cast<int>(m_params.imageHeight) - 1);
        const int right  = std::clamp(static_cast<int>(std::lround(centerX + boxWidth * 0.5F)),
                                      0,
                                      static_cast<int>(m_params.imageWidth));
        const int bottom = std::clamp(static_cast<int>(std::lround(centerY + boxHeight * 0.5F)),
                                      0,
                                      static_cast<int>(m_params.imageHeight));
        if (right <= left || bottom <= top) {
            continue;
        }

        yolov8_pose::PoseResult candidate;
        candidate.score   = bestScore;
        candidate.classId = bestClass;
        candidate.x       = left;
        candidate.y       = top;
        candidate.width   = right - left;
        candidate.height  = bottom - top;

        for (size_t keypoint = 0; keypoint < yolov8_pose::kNumKeypoints; ++keypoint) {
            candidate.keypoints[keypoint] = {
                std::clamp(ToImageCoordinate(output.KeypointCoordinate(keypoint, 0, box),
                                             m_params.imageWidth,
                                             m_params.coordinatesNormalized),
                           0.0F,
                           static_cast<float>(m_params.imageWidth - 1)),
                std::clamp(ToImageCoordinate(output.KeypointCoordinate(keypoint, 1, box),
                                             m_params.imageHeight,
                                             m_params.coordinatesNormalized),
                           0.0F,
                           static_cast<float>(m_params.imageHeight - 1)),
                output.KeypointScore(keypoint, box)};
        }

        RetainCandidate(candidates, std::move(candidate), candidateLimit);
    }

    ApplyClassAwareNms(candidates, m_results, m_params);
    return true;
}

} /* namespace arm::app */
