/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp"

#include "mlek/log/log_macros.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace arm::app {

namespace {

float CalculateDetectionIoU(
    const yolov8_detection::DetectionResult& first,
    const yolov8_detection::DetectionResult& second)
{
    const int intersectionLeft = std::max(first.x, second.x);
    const int intersectionTop = std::max(first.y, second.y);
    const int intersectionRight =
        std::min(first.x + first.width, second.x + second.width);
    const int intersectionBottom =
        std::min(first.y + first.height, second.y + second.height);

    const int intersectionWidth =
        std::max(0, intersectionRight - intersectionLeft);
    const int intersectionHeight =
        std::max(0, intersectionBottom - intersectionTop);

    const float intersectionArea =
        static_cast<float>(intersectionWidth * intersectionHeight);
    const float firstArea =
        static_cast<float>(first.width * first.height);
    const float secondArea =
        static_cast<float>(second.width * second.height);
    const float unionArea = firstArea + secondArea - intersectionArea;

    return unionArea > 0.0F ? intersectionArea / unionArea : 0.0F;
}

void ApplyClassAwareNms(
    std::vector<yolov8_detection::DetectionResult>& candidates,
    std::vector<yolov8_detection::DetectionResult>& results,
    const YoloV8PostProcessParams& params)
{
    if (params.maxDetections == 0) {
        return;
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const auto& first, const auto& second) {
            return first.score > second.score;
        });

    for (const auto& candidate : candidates) {
        bool suppressed = false;

        for (const auto& selected : results) {
            if (candidate.classId != selected.classId) {
                continue;
            }

            if (CalculateDetectionIoU(candidate, selected) > params.nmsThreshold) {
                suppressed = true;
                break;
            }
        }

        if (!suppressed) {
            results.push_back(candidate);

            if (results.size() >= params.maxDetections) {
                break;
            }
        }
    }
}

float Dequantize(const int8_t value, const fwk::iface::QuantParams& quantParams)
{
    return (static_cast<float>(value) - static_cast<float>(quantParams.offset)) *
           quantParams.scale;
}

float Sigmoid(const float value)
{
    return 1.0F / (1.0F + std::exp(-value));
}

float ToProbability(const float value)
{
    /* Ultralytics exports probabilities, while some quantized heads expose logits. */
    if (value >= -0.01F && value <= 1.01F) {
        return std::clamp(value, 0.0F, 1.0F);
    }
    return Sigmoid(value);
}

bool ReadTensorValue(
    const std::shared_ptr<fwk::iface::TensorIface>& tensor,
    size_t index,
    float& value)
{
    if (tensor == nullptr || index >= tensor->GetNumElements()) {
        return false;
    }

    switch (tensor->Type()) {
    case fwk::iface::TensorType::FP32:
        value = tensor->GetData<float>()[index];
        return true;
    case fwk::iface::TensorType::INT8: {
        const auto quantParams = tensor->GetQuantParams();
        if (quantParams.scale <= 0.0F) {
            return false;
        }
        value = Dequantize(tensor->GetData<int8_t>()[index], quantParams);
        return true;
    }
    case fwk::iface::TensorType::UINT8: {
        const auto quantParams = tensor->GetQuantParams();
        if (quantParams.scale <= 0.0F) {
            return false;
        }
        value = (static_cast<float>(tensor->GetData<uint8_t>()[index]) -
                 static_cast<float>(quantParams.offset)) * quantParams.scale;
        return true;
    }
    default:
        return false;
    }
}

enum class SingleOutputLayout { AttributesFirst, BoxesFirst };

struct SingleOutputDescription {
    size_t attributes{0};
    size_t boxCount{0};
    SingleOutputLayout layout{SingleOutputLayout::AttributesFirst};
};

bool DescribeSingleOutput(
    const std::vector<size_t>& shape,
    uint32_t numClasses,
    SingleOutputDescription& description)
{
    size_t firstDimension = 0;
    size_t secondDimension = 0;

    if (shape.size() == 3 && shape[0] == 1) {
        firstDimension = shape[1];
        secondDimension = shape[2];
    } else if (shape.size() == 2) {
        firstDimension = shape[0];
        secondDimension = shape[1];
    } else {
        return false;
    }

    const size_t classOnlyAttributes = 4 + numClasses;
    const size_t objectnessAttributes = 5 + numClasses;
    if (firstDimension == classOnlyAttributes || firstDimension == objectnessAttributes) {
        description.attributes = firstDimension;
        description.boxCount = secondDimension;
        description.layout = SingleOutputLayout::AttributesFirst;
    } else if (secondDimension == classOnlyAttributes ||
               secondDimension == objectnessAttributes) {
        description.attributes = secondDimension;
        description.boxCount = firstDimension;
        description.layout = SingleOutputLayout::BoxesFirst;
    } else {
        return false;
    }

    return description.boxCount > 0;
}

size_t GetSingleOutputIndex(
    const SingleOutputDescription& description,
    size_t box,
    size_t attribute)
{
    if (description.layout == SingleOutputLayout::AttributesFirst) {
        return attribute * description.boxCount + box;
    }
    return box * description.attributes + attribute;
}

bool GetDetectionBox(
    const float centerX,
    const float centerY,
    const float boxWidth,
    const float boxHeight,
    const YoloV8PostProcessParams& params,
    const bool normalized,
    yolov8_detection::DetectionResult& result)
{
    if (!std::isfinite(centerX) || !std::isfinite(centerY) ||
        !std::isfinite(boxWidth) || !std::isfinite(boxHeight)) {
        return false;
    }

    const float scaledCenterX =
        normalized ? centerX * static_cast<float>(params.imageWidth) : centerX;
    const float scaledCenterY =
        normalized ? centerY * static_cast<float>(params.imageHeight) : centerY;
    const float scaledWidth =
        normalized ? boxWidth * static_cast<float>(params.imageWidth) : boxWidth;
    const float scaledHeight =
        normalized ? boxHeight * static_cast<float>(params.imageHeight) : boxHeight;

    const int left = static_cast<int>(scaledCenterX - scaledWidth * 0.5F);
    const int top = static_cast<int>(scaledCenterY - scaledHeight * 0.5F);
    const int right = static_cast<int>(scaledCenterX + scaledWidth * 0.5F);
    const int bottom = static_cast<int>(scaledCenterY + scaledHeight * 0.5F);

    result.x = std::clamp(left, 0, static_cast<int>(params.imageWidth) - 1);
    result.y = std::clamp(top, 0, static_cast<int>(params.imageHeight) - 1);
    const int clippedRight = std::clamp(right, 0, static_cast<int>(params.imageWidth));
    const int clippedBottom = std::clamp(bottom, 0, static_cast<int>(params.imageHeight));
    result.width = clippedRight - result.x;
    result.height = clippedBottom - result.y;

    return result.width > 0 && result.height > 0;
}

} /* anonymous namespace */

YoloV8PostProcess::YoloV8PostProcess(
    const std::shared_ptr<fwk::iface::TensorIface>& outputTensor,
    std::vector<yolov8_detection::DetectionResult>& results,
    const YoloV8PostProcessParams& params) :
    m_outputTensor{outputTensor},
    m_results{results},
    m_params{params}
{}

bool YoloV8PostProcess::DoPostProcess()
{
    m_results.clear();

    if (m_outputTensor == nullptr) {
        printf_err("YOLOv8 output tensor is unavailable\n");
        return false;
    }

    SingleOutputDescription description;
    if (!DescribeSingleOutput(m_outputTensor->Shape(), m_params.numClasses, description)) {
        printf_err(
            "Expected YOLOv8 output shape [1, 4 + classes, boxes] or "
            "[1, boxes, 4 + classes]\n");
        return false;
    }

    const bool hasObjectness = description.attributes == 5 + m_params.numClasses;
    float maximumCoordinate = 0.0F;
    for (size_t box = 0; box < description.boxCount; ++box) {
        for (size_t coordinate = 0; coordinate < 4; ++coordinate) {
            float value = 0.0F;
            if (!ReadTensorValue(
                    m_outputTensor,
                    GetSingleOutputIndex(description, box, coordinate),
                    value)) {
                printf_err("YOLOv8 output type or quantization is unsupported\n");
                return false;
            }
            maximumCoordinate = std::max(maximumCoordinate, std::fabs(value));
        }
    }

    /* Normalized exports stay close to [0, 1]; pixel-space exports are much larger. */
    const bool normalizedCoordinates = maximumCoordinate <= 2.0F;
    std::vector<yolov8_detection::DetectionResult> candidates;
    candidates.reserve(std::min(
        description.boxCount,
        static_cast<size_t>(m_params.maxDetections) * 4U));

    for (size_t box = 0; box < description.boxCount; ++box) {
        float bestScore = 0.0F;
        int bestClass = -1;

        float objectness = 1.0F;
        if (hasObjectness) {
            float objectnessValue = 0.0F;
            if (!ReadTensorValue(
                    m_outputTensor,
                    GetSingleOutputIndex(description, box, 4),
                    objectnessValue)) {
                printf_err("YOLOv8 objectness output is unavailable\n");
                return false;
            }
            objectness = ToProbability(objectnessValue);
        }

        for (size_t classIndex = 0; classIndex < m_params.numClasses; ++classIndex) {
            float classValue = 0.0F;
            if (!ReadTensorValue(
                    m_outputTensor,
                    GetSingleOutputIndex(
                        description, box, 4 + classIndex + (hasObjectness ? 1 : 0)),
                    classValue)) {
                printf_err("YOLOv8 class output is unavailable\n");
                return false;
            }
            const float score = objectness * ToProbability(classValue);

            if (score > bestScore) {
                bestScore = score;
                bestClass = static_cast<int>(classIndex);
            }
        }

        if (bestScore < m_params.scoreThreshold) {
            continue;
        }

        float centerX = 0.0F;
        float centerY = 0.0F;
        float boxWidth = 0.0F;
        float boxHeight = 0.0F;
        if (!ReadTensorValue(
                m_outputTensor, GetSingleOutputIndex(description, box, 0), centerX) ||
            !ReadTensorValue(
                m_outputTensor, GetSingleOutputIndex(description, box, 1), centerY) ||
            !ReadTensorValue(
                m_outputTensor, GetSingleOutputIndex(description, box, 2), boxWidth) ||
            !ReadTensorValue(
                m_outputTensor, GetSingleOutputIndex(description, box, 3), boxHeight)) {
            printf_err("YOLOv8 box output is unavailable\n");
            continue;
        }

        yolov8_detection::DetectionResult result{bestScore, bestClass};
        if (GetDetectionBox(
                centerX,
                centerY,
                boxWidth,
                boxHeight,
                m_params,
                normalizedCoordinates,
                result)) {
            candidates.push_back(result);
        }
    }

    /* Class-aware NMS: different classes do not suppress each other. */
    ApplyClassAwareNms(candidates, m_results, m_params);

    return true;
}

YoloV8NewPostProcess::YoloV8NewPostProcess(
    const std::vector<std::shared_ptr<fwk::iface::TensorIface>>& outputTensors,
    std::vector<yolov8_detection::DetectionResult>& results,
    const YoloV8PostProcessParams& params) :
    m_outputTensors{outputTensors},
    m_results{results},
    m_params{params}
{}

bool YoloV8NewPostProcess::DoPostProcess()
{
    struct FeatureMap {
        std::shared_ptr<fwk::iface::TensorIface> tensor;
        size_t height{0};
        size_t width{0};
        size_t channels{0};
        size_t stride{0};
        size_t objectnessOffset{0};
        bool channelsFirst{false};
    };

    m_results.clear();

    if (m_outputTensors.size() != 3 && m_outputTensors.size() != 4) {
        printf_err("New YOLOv8 model must have three or four output tensors\n");
        return false;
    }

    std::shared_ptr<fwk::iface::TensorIface> objectnessTensor;
    std::vector<FeatureMap> featureMaps;

    for (const auto& tensor : m_outputTensors) {
        if (tensor == nullptr ||
            (tensor->Type() != fwk::iface::TensorType::FP32 &&
             tensor->Type() != fwk::iface::TensorType::INT8 &&
             tensor->Type() != fwk::iface::TensorType::UINT8)) {
            printf_err("New YOLOv8 outputs must be float32, int8 or uint8 tensors\n");
            return false;
        }

        const auto shape = tensor->Shape();
        const bool isObjectnessShape =
            (shape.size() == 1) ||
            (shape.size() == 2 && shape[0] == 1) ||
            (shape.size() == 3 && shape[0] == 1 && (shape[1] == 1 || shape[2] == 1));
        if (isObjectnessShape) {
            if (objectnessTensor != nullptr) {
                printf_err("New YOLOv8 model has multiple objectness outputs\n");
                return false;
            }
            objectnessTensor = tensor;
            continue;
        }

        if (shape.size() != 4 || shape[0] != 1 ||
            shape[1] == 0 || shape[2] == 0 || shape[3] == 0) {
            printf_err("Invalid new YOLOv8 feature map shape\n");
            return false;
        }

        const size_t expectedChannels = ms_dflValues + m_params.numClasses;
        if (shape[3] == expectedChannels ||
            (shape[3] >= ms_dflValues && shape[1] < ms_dflValues)) {
            featureMaps.push_back({tensor, shape[1], shape[2], shape[3], 0, 0, false});
        } else if (shape[1] == expectedChannels ||
                   (shape[1] >= ms_dflValues && shape[3] < ms_dflValues)) {
            featureMaps.push_back({tensor, shape[2], shape[3], shape[1], 0, 0, true});
        } else {
            printf_err("New YOLOv8 feature map has no DFL channel dimension\n");
            return false;
        }
    }

    if (featureMaps.size() != 3) {
        printf_err("New YOLOv8 model must have three feature maps\n");
        return false;
    }

    std::sort(
        featureMaps.begin(),
        featureMaps.end(),
        [](const FeatureMap& first, const FeatureMap& second) {
            return first.width > second.width;
        });

    const size_t objectnessElements =
        objectnessTensor == nullptr ? 0 : objectnessTensor->GetNumElements();
    size_t objectnessOffset = 0;
    size_t numClasses = 0;

    for (auto& featureMap : featureMaps) {
        if (m_params.imageWidth % featureMap.width != 0 ||
            m_params.imageHeight % featureMap.height != 0 ||
            featureMap.channels < ms_dflValues) {
            printf_err("Invalid new YOLOv8 feature map dimensions\n");
            return false;
        }

        featureMap.stride = m_params.imageWidth / featureMap.width;
        if (featureMap.stride == 0 ||
            m_params.imageHeight / featureMap.height != featureMap.stride) {
            printf_err("New YOLOv8 feature map stride is invalid\n");
            return false;
        }

        const size_t featureMapClasses = featureMap.channels - ms_dflValues;
        if (numClasses == 0) {
            numClasses = featureMapClasses;
        } else if (numClasses != featureMapClasses) {
            printf_err("New YOLOv8 feature maps have different class counts\n");
            return false;
        }

        featureMap.objectnessOffset = objectnessOffset;
        objectnessOffset += featureMap.height * featureMap.width;
    }

    if (m_params.numClasses != 0 && m_params.numClasses != numClasses) {
        printf_err("New YOLOv8 output class count does not match configuration\n");
        return false;
    }

    if (objectnessTensor != nullptr && objectnessOffset != objectnessElements) {
        printf_err("New YOLOv8 objectness output has an unexpected size\n");
        return false;
    }

    std::vector<yolov8_detection::DetectionResult> candidates;
    candidates.reserve(objectnessElements);

    for (const auto& featureMap : featureMaps) {
        for (size_t y = 0; y < featureMap.height; ++y) {
            for (size_t x = 0; x < featureMap.width; ++x) {
                const size_t cell = y * featureMap.width + x;
                float objectnessScore = 1.0F;
                if (objectnessTensor != nullptr) {
                    float objectnessValue = 0.0F;
                    if (!ReadTensorValue(
                            objectnessTensor,
                            featureMap.objectnessOffset + cell,
                            objectnessValue)) {
                        printf_err("New YOLOv8 objectness output is unsupported\n");
                        return false;
                    }
                    objectnessScore = Sigmoid(objectnessValue);
                }

                std::array<float, ms_dflBins> dflValues{};
                std::array<float, 4> distances{};

                for (size_t side = 0; side < 4; ++side) {
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (size_t bin = 0; bin < ms_dflBins; ++bin) {
                        const size_t channel = side * ms_dflBins + bin;
                        const size_t index =
                            featureMap.channelsFirst
                                ? channel * featureMap.height * featureMap.width + cell
                                : cell * featureMap.channels + channel;
                        if (!ReadTensorValue(featureMap.tensor, index, dflValues[bin])) {
                            printf_err("New YOLOv8 feature map output is unsupported\n");
                            return false;
                        }
                        maximum = std::max(maximum, dflValues[bin]);
                    }

                    float sum = 0.0F;
                    float weightedSum = 0.0F;
                    for (size_t bin = 0; bin < ms_dflBins; ++bin) {
                        const float probability = std::exp(dflValues[bin] - maximum);
                        sum += probability;
                        weightedSum += static_cast<float>(bin) * probability;
                    }
                    distances[side] = sum > 0.0F ? weightedSum / sum : 0.0F;
                }

                float bestScore = 0.0F;
                int bestClass = -1;
                for (size_t classIndex = 0; classIndex < numClasses; ++classIndex) {
                    const size_t channel = ms_dflValues + classIndex;
                    const size_t index =
                        featureMap.channelsFirst
                            ? channel * featureMap.height * featureMap.width + cell
                            : cell * featureMap.channels + channel;
                    float classValue = 0.0F;
                    if (!ReadTensorValue(featureMap.tensor, index, classValue)) {
                        printf_err("New YOLOv8 class output is unsupported\n");
                        return false;
                    }
                    const float classScore = Sigmoid(classValue);
                    const float score = objectnessScore * classScore;
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = static_cast<int>(classIndex);
                    }
                }

                if (bestClass < 0 || bestScore < m_params.scoreThreshold) {
                    continue;
                }

                const float centerX = (static_cast<float>(x) + 0.5F) * featureMap.stride;
                const float centerY = (static_cast<float>(y) + 0.5F) * featureMap.stride;
                const float left = centerX - distances[0] * featureMap.stride;
                const float top = centerY - distances[1] * featureMap.stride;
                const float right = centerX + distances[2] * featureMap.stride;
                const float bottom = centerY + distances[3] * featureMap.stride;

                const int clippedLeft = std::clamp(
                    static_cast<int>(left), 0, static_cast<int>(m_params.imageWidth) - 1);
                const int clippedTop = std::clamp(
                    static_cast<int>(top), 0, static_cast<int>(m_params.imageHeight) - 1);
                const int clippedRight = std::clamp(
                    static_cast<int>(right), 0, static_cast<int>(m_params.imageWidth));
                const int clippedBottom = std::clamp(
                    static_cast<int>(bottom), 0, static_cast<int>(m_params.imageHeight));

                if (clippedRight <= clippedLeft || clippedBottom <= clippedTop) {
                    continue;
                }

                candidates.push_back({
                    bestScore,
                    bestClass,
                    clippedLeft,
                    clippedTop,
                    clippedRight - clippedLeft,
                    clippedBottom - clippedTop
                });
            }
        }
    }

    ApplyClassAwareNms(candidates, m_results, m_params);
    return true;
}

} /* namespace arm::app */
