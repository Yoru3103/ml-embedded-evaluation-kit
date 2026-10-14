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

struct TensorView {
    const void* data{nullptr};
    size_t numElements{0};
    fwk::iface::TensorType type{fwk::iface::TensorType::INVALID};
    fwk::iface::QuantParams quantParams{};
};

bool MakeTensorView(const std::shared_ptr<fwk::iface::TensorIface>& tensor, TensorView& view)
{
    if (tensor == nullptr) {
        return false;
    }

    view.type = tensor->Type();
    switch (view.type) {
    case fwk::iface::TensorType::FP32:
        break;
    case fwk::iface::TensorType::INT8:
    case fwk::iface::TensorType::UINT8:
        view.quantParams = tensor->GetQuantParams();
        if (view.quantParams.scale <= 0.0F) {
            return false;
        }
        break;
    default:
        return false;
    }

    /* Cache metadata once per pass; no virtual tensor calls in the pixel loops. */
    view.numElements = tensor->GetNumElements();
    view.data        = tensor->GetData();
    return view.data != nullptr;
}

bool ReadTensorValue(const TensorView& view, size_t index, float& value)
{
    if (index >= view.numElements) {
        return false;
    }

    switch (view.type) {
    case fwk::iface::TensorType::FP32:
        value = static_cast<const float*>(view.data)[index];
        return true;
    case fwk::iface::TensorType::INT8:
        value = Dequantize(static_cast<const int8_t*>(view.data)[index], view.quantParams);
        return true;
    case fwk::iface::TensorType::UINT8:
        value = (static_cast<float>(static_cast<const uint8_t*>(view.data)[index]) -
                 static_cast<float>(view.quantParams.offset)) *
                view.quantParams.scale;
        return true;
    default:
        return false;
    }
}

struct ProbabilityGate {
    int minimumCode{0};
    bool enabled{false};
};

ProbabilityGate MakeProbabilityGate(const TensorView& view, float threshold)
{
    ProbabilityGate gate;
    if (!(threshold > 0.0F) || !std::isfinite(view.quantParams.scale)) {
        return gate;
    }

    int lower = 0;
    int upper = 0;
    if (view.type == fwk::iface::TensorType::INT8) {
        lower = std::numeric_limits<int8_t>::min();
        upper = std::numeric_limits<int8_t>::max() + 1;
    } else if (view.type == fwk::iface::TensorType::UINT8) {
        lower = std::numeric_limits<uint8_t>::min();
        upper = std::numeric_limits<uint8_t>::max() + 1;
    } else {
        return gate;
    }

    /* Use the same float arithmetic as scoring, including equality at the threshold.
     * The exclusive upper bound also represents "no quantized value can pass". */
    while (lower < upper) {
        const int middle = lower + (upper - lower) / 2;
        const float value =
            (static_cast<float>(middle) - static_cast<float>(view.quantParams.offset)) *
            view.quantParams.scale;
        if (Sigmoid(value) < threshold) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    gate.minimumCode = lower;
    gate.enabled     = true;
    return gate;
}

bool ReadGatedTensorValue(const TensorView& view,
                          size_t index,
                          const ProbabilityGate& gate,
                          float& value,
                          bool& passesGate)
{
    if (index >= view.numElements) {
        return false;
    }
    passesGate = true;
    if (gate.enabled) {
        const int code = view.type == fwk::iface::TensorType::INT8
                             ? static_cast<const int8_t*>(view.data)[index]
                             : static_cast<const uint8_t*>(view.data)[index];
        passesGate     = code >= gate.minimumCode;
        if (!passesGate) {
            return true;
        }
    }
    return ReadTensorValue(view, index, value);
}

template <typename T>
size_t FindMaximumClass(const T* pData, size_t classStride, size_t numClasses)
{
    T maximum        = std::numeric_limits<T>::lowest();
    size_t bestClass = 0;
    for (size_t classIndex = 0; classIndex < numClasses; ++classIndex) {
        const T value = pData[classIndex * classStride];
        if (value > maximum) {
            maximum   = value;
            bestClass = classIndex;
        }
    }
    return bestClass;
}

bool ReadMaximumClass(const TensorView& view,
                      size_t firstIndex,
                      size_t classStride,
                      size_t numClasses,
                      const ProbabilityGate& gate,
                      float& value,
                      int& bestClass,
                      bool& passesGate)
{
    if (numClasses == 0 || firstIndex >= view.numElements || classStride == 0 ||
        numClasses - 1 > (view.numElements - 1 - firstIndex) / classStride) {
        return false;
    }

    size_t classIndex = 0;
    switch (view.type) {
    case fwk::iface::TensorType::INT8:
        classIndex = FindMaximumClass(
            static_cast<const int8_t*>(view.data) + firstIndex, classStride, numClasses);
        break;
    case fwk::iface::TensorType::UINT8:
        classIndex = FindMaximumClass(
            static_cast<const uint8_t*>(view.data) + firstIndex, classStride, numClasses);
        break;
    case fwk::iface::TensorType::FP32:
        classIndex = FindMaximumClass(
            static_cast<const float*>(view.data) + firstIndex, classStride, numClasses);
        break;
    default:
        return false;
    }

    /* Positive per-tensor scale preserves logit order. Equal logits keep the first class. */
    bestClass = static_cast<int>(classIndex);
    return ReadGatedTensorValue(
        view, firstIndex + classIndex * classStride, gate, value, passesGate);
}

enum class SingleOutputLayout { AttributesFirst, BoxesFirst };

struct SingleOutputDescription {
    size_t attributes{0};
    size_t boxCount{0};
    SingleOutputLayout layout{SingleOutputLayout::AttributesFirst};
};

bool DescribeSingleOutput(const std::vector<size_t>& shape,
                          uint32_t numClasses,
                          SingleOutputDescription& description)
{
    size_t firstDimension  = 0;
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
        printf_err("Expected YOLOv8 output shape [1, 4 + classes, boxes] or "
                   "[1, boxes, 4 + classes]\n");
        return false;
    }

    const bool hasObjectness = description.attributes == 5 + m_params.numClasses;
    TensorView outputView;
    if (!MakeTensorView(m_outputTensor, outputView)) {
        printf_err("YOLOv8 output type or quantization is unsupported\n");
        return false;
    }
    float maximumCoordinate = 0.0F;
    for (size_t box = 0; box < description.boxCount; ++box) {
        for (size_t coordinate = 0; coordinate < 4; ++coordinate) {
            float value = 0.0F;
            if (!ReadTensorValue(
                    outputView, GetSingleOutputIndex(description, box, coordinate), value)) {
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
                    outputView,
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
                    outputView,
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
                outputView, GetSingleOutputIndex(description, box, 0), centerX) ||
            !ReadTensorValue(
                outputView, GetSingleOutputIndex(description, box, 1), centerY) ||
            !ReadTensorValue(
                outputView, GetSingleOutputIndex(description, box, 2), boxWidth) ||
            !ReadTensorValue(
                outputView, GetSingleOutputIndex(description, box, 3), boxHeight)) {
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
    const YoloV8PostProcessParams& params,
    bool useTensorLayout) :
    m_outputTensors{outputTensors},
    m_useTensorLayout{useTensorLayout},
    m_results{results},
    m_params{params}
{}

bool YoloV8NewPostProcess::DoPostProcess()
{
    const size_t dflBins = m_params.dflBins;
    const size_t dflValues = 4 * dflBins;
    if ((dflBins != 8 && dflBins != 16) || dflBins > ms_maxDflBins ||
        (m_params.orientedBoxes &&
         (m_outputTensors.size() != 4 || m_params.maxDetections > 1))) {
        printf_err("DFL requires 8/16 bins; OBB requires four outputs and Top1\n");
        return false;
    }
    struct FeatureMap {
        TensorView view;
        size_t height{0};
        size_t width{0};
        size_t channels{0};
        size_t stride{0};
        size_t objectnessOffset{0};
        bool channelsFirst{false};
        ProbabilityGate classGate;
    };

    m_results.clear();

    if (m_outputTensors.size() != 3 && m_outputTensors.size() != 4) {
        printf_err("New YOLOv8 model must have three or four output tensors\n");
        return false;
    }

    std::shared_ptr<fwk::iface::TensorIface> objectnessTensor;
    TensorView objectnessView;
    ProbabilityGate objectnessGate;
    std::vector<FeatureMap> featureMaps;

    for (const auto& tensor : m_outputTensors) {
        TensorView view;
        if (!MakeTensorView(tensor, view)) {
            printf_err("New YOLOv8 output data, type or quantization is unsupported\n");
            return false;
        }

        const auto shape = tensor->Shape();
        const bool isObjectnessShape =
            (shape.size() == 1) || (shape.size() == 2 && shape[0] == 1) ||
            (shape.size() == 3 && shape[0] == 1 && (shape[1] == 1 || shape[2] == 1));
        if (isObjectnessShape) {
            if (objectnessTensor != nullptr) {
                printf_err("New YOLOv8 model has multiple objectness outputs\n");
                return false;
            }
            objectnessTensor = tensor;
            objectnessView   = view;
            if (!m_params.orientedBoxes) {
                objectnessGate = MakeProbabilityGate(view, m_params.scoreThreshold);
            }
            continue;
        }

        if (shape.size() != 4 || shape[0] != 1 || shape[1] == 0 || shape[2] == 0 || shape[3] == 0) {
            printf_err("Invalid new YOLOv8 feature map shape\n");
            return false;
        }

        const size_t expectedChannels = dflValues + m_params.numClasses;
        if (shape[3] == expectedChannels || (shape[3] >= dflValues && shape[1] < dflValues)) {
            featureMaps.push_back({view,
                                   shape[1],
                                   shape[2],
                                   shape[3],
                                   0,
                                   0,
                                   false,
                                   MakeProbabilityGate(view, m_params.scoreThreshold)});
        } else if (shape[1] == expectedChannels ||
                   (shape[1] >= dflValues && shape[3] < dflValues)) {
            const bool channelsFirst =
                !m_useTensorLayout || tensor->Layout() != fwk::iface::TensorLayout::NHWC;
            featureMaps.push_back({view,
                                   shape[2],
                                   shape[3],
                                   shape[1],
                                   0,
                                   0,
                                   channelsFirst,
                                   MakeProbabilityGate(view, m_params.scoreThreshold)});
        } else {
            printf_err("New YOLOv8 feature map has no DFL channel dimension\n");
            return false;
        }
    }

    if (featureMaps.size() != 3) {
        printf_err("New YOLOv8 model must have three feature maps\n");
        return false;
    }

    std::sort(featureMaps.begin(),
              featureMaps.end(),
              [](const FeatureMap& first, const FeatureMap& second) {
                  return first.width > second.width;
              });

    const size_t objectnessElements = objectnessTensor == nullptr ? 0 : objectnessView.numElements;
    size_t objectnessOffset         = 0;
    size_t numClasses               = 0;

    for (auto& featureMap : featureMaps) {
        if (m_params.imageWidth % featureMap.width != 0 ||
            m_params.imageHeight % featureMap.height != 0 || featureMap.channels < dflValues ||
            featureMap.view.numElements !=
                featureMap.height * featureMap.width * featureMap.channels) {
            printf_err("Invalid new YOLOv8 feature map dimensions\n");
            return false;
        }

        featureMap.stride = m_params.imageWidth / featureMap.width;
        if (featureMap.stride == 0 ||
            m_params.imageHeight / featureMap.height != featureMap.stride) {
            printf_err("New YOLOv8 feature map stride is invalid\n");
            return false;
        }

        const size_t featureMapClasses = featureMap.channels - dflValues;
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

    struct ScoredCell {
        const FeatureMap* featureMap;
        size_t cell;
        float score;
        int classId;
    };

    if (m_params.maxDetections == 0) {
        return true;
    }

    std::vector<ScoredCell> scoredCells;
    scoredCells.reserve(
        std::min(objectnessOffset, static_cast<size_t>(m_params.maxDetections) * 4U));

    for (const auto& featureMap : featureMaps) {
        for (size_t y = 0; y < featureMap.height; ++y) {
            for (size_t x = 0; x < featureMap.width; ++x) {
                const size_t cell = y * featureMap.width + x;
                const size_t classStride =
                    featureMap.channelsFirst ? featureMap.height * featureMap.width : 1;
                const size_t firstClassIndex = featureMap.channelsFirst
                                                    ? dflValues * classStride + cell
                                                    : cell * featureMap.channels + dflValues;
                float classValue             = 0.0F;
                int bestClass                = -1;
                bool classPassesGate         = false;
                if (!ReadMaximumClass(featureMap.view,
                                      firstClassIndex,
                                      classStride,
                                      numClasses,
                                      featureMap.classGate,
                                      classValue,
                                      bestClass,
                                      classPassesGate)) {
                    printf_err("New YOLOv8 class output is unsupported\n");
                    return false;
                }
                /* The class gate is valid for both scoring modes. Angle is never
                 * a confidence bound and is only read when decoding an OBB. */
                if (!classPassesGate) {
                    continue;
                }

                float objectnessScore = 1.0F;
                if (objectnessTensor != nullptr && !m_params.orientedBoxes) {
                    float objectnessValue     = 0.0F;
                    bool objectnessPassesGate = false;
                    if (!ReadGatedTensorValue(objectnessView,
                                              featureMap.objectnessOffset + cell,
                                              objectnessGate,
                                              objectnessValue,
                                              objectnessPassesGate)) {
                        printf_err("New YOLOv8 objectness output is unsupported\n");
                        return false;
                    }
                    if (!objectnessPassesGate) {
                        continue;
                    }
                    objectnessScore = Sigmoid(objectnessValue);
                }
                const float bestScore = objectnessScore * Sigmoid(classValue);

                if (!(bestScore > 0.0F) || bestScore < m_params.scoreThreshold) {
                    continue;
                }

                scoredCells.push_back({&featureMap, cell, bestScore, bestClass});
            }
        }
    }

    const auto decodeBox = [&](const ScoredCell& candidate,
                               yolov8_detection::DetectionResult& result) {
        const auto& featureMap = *candidate.featureMap;
        const size_t cell      = candidate.cell;
        const size_t x         = cell % featureMap.width;
        const size_t y         = cell / featureMap.width;

        /* Decode boxes only after their confidence passes the threshold. */
        std::array<float, ms_maxDflBins> logits{};
        std::array<float, 4> distances{};

        for (size_t side = 0; side < 4; ++side) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t bin = 0; bin < dflBins; ++bin) {
                const size_t channel = side * dflBins + bin;
                const size_t index   = featureMap.channelsFirst
                                           ? channel * featureMap.height * featureMap.width + cell
                                           : cell * featureMap.channels + channel;
                if (!ReadTensorValue(featureMap.view, index, logits[bin])) {
                    printf_err("New YOLOv8 feature map output is unsupported\n");
                    return false;
                }
                maximum = std::max(maximum, logits[bin]);
            }

            float sum         = 0.0F;
            float weightedSum = 0.0F;
            for (size_t bin = 0; bin < dflBins; ++bin) {
                const float probability = std::exp(logits[bin] - maximum);
                sum += probability;
                weightedSum += static_cast<float>(bin) * probability;
            }
            distances[side] = sum > 0.0F ? weightedSum / sum : 0.0F;
        }

        const float centerX = (static_cast<float>(x) + 0.5F) * featureMap.stride;
        const float centerY = (static_cast<float>(y) + 0.5F) * featureMap.stride;
        float left    = centerX - distances[0] * featureMap.stride;
        float top     = centerY - distances[1] * featureMap.stride;
        float right   = centerX + distances[2] * featureMap.stride;
        float bottom  = centerY + distances[3] * featureMap.stride;
        float angle = 0.0F;
        float boxCenterX = centerX;
        float boxCenterY = centerY;
        const float boxWidth = (distances[0] + distances[2]) * featureMap.stride;
        const float boxHeight = (distances[1] + distances[3]) * featureMap.stride;
        if (m_params.orientedBoxes) {
            float angleLogit = 0.0F;
            if (!ReadTensorValue(objectnessView, featureMap.objectnessOffset + cell, angleLogit) ||
                !std::isfinite(angleLogit)) {
                printf_err("OBB angle output is invalid\n");
                return false;
            }
            constexpr float pi = 3.14159265358979323846F;
            angle = (Sigmoid(angleLogit) - 0.25F) * pi;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const float offsetX = (distances[2] - distances[0]) * 0.5F * featureMap.stride;
            const float offsetY = (distances[3] - distances[1]) * 0.5F * featureMap.stride;
            boxCenterX += offsetX * cosine - offsetY * sine;
            boxCenterY += offsetX * sine + offsetY * cosine;
            const float envelopeWidth = boxWidth * std::fabs(cosine) + boxHeight * std::fabs(sine);
            const float envelopeHeight = boxWidth * std::fabs(sine) + boxHeight * std::fabs(cosine);
            left = boxCenterX - envelopeWidth * 0.5F;
            right = boxCenterX + envelopeWidth * 0.5F;
            top = boxCenterY - envelopeHeight * 0.5F;
            bottom = boxCenterY + envelopeHeight * 0.5F;
        }

        const int clippedLeft =
            std::clamp(static_cast<int>(left), 0, static_cast<int>(m_params.imageWidth) - 1);
        const int clippedTop =
            std::clamp(static_cast<int>(top), 0, static_cast<int>(m_params.imageHeight) - 1);
        const int clippedRight =
            std::clamp(static_cast<int>(right), 0, static_cast<int>(m_params.imageWidth));
        const int clippedBottom =
            std::clamp(static_cast<int>(bottom), 0, static_cast<int>(m_params.imageHeight));

        result = {candidate.score,
                  candidate.classId,
                  clippedLeft,
                  clippedTop,
                  clippedRight - clippedLeft,
                  clippedBottom - clippedTop};
        if (m_params.orientedBoxes) {
            result.centerX = boxCenterX;
            result.centerY = boxCenterY;
            result.boxWidth = boxWidth;
            result.boxHeight = boxHeight;
            result.angle = angle;
            result.rotated = true;
        }
        return true;
    };

    if (m_params.maxDetections == 1) {
        /* Select globally before DFL; normally decode just one box and skip NMS. */
        while (!scoredCells.empty()) {
            const auto best =
                std::max_element(scoredCells.begin(),
                                 scoredCells.end(),
                                 [](const ScoredCell& first, const ScoredCell& second) {
                                     return first.score < second.score;
                                 });
            yolov8_detection::DetectionResult result;
            if (!decodeBox(*best, result)) {
                return false;
            }
            if (result.width > 0 && result.height > 0) {
                m_results.push_back(result);
                return true;
            }
            /* A clipped zero-area box must not hide the next valid candidate. */
            scoredCells.erase(best);
        }
        return true;
    }

    std::vector<yolov8_detection::DetectionResult> candidates;
    candidates.reserve(scoredCells.size());
    for (const auto& candidate : scoredCells) {
        yolov8_detection::DetectionResult result;
        if (!decodeBox(candidate, result)) {
            return false;
        }
        if (result.width > 0 && result.height > 0) {
            candidates.push_back(result);
        }
    }

    ApplyClassAwareNms(candidates, m_results, m_params);
    return true;
}

} /* namespace arm::app */
