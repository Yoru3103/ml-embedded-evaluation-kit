/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mlek/use_case/asr/ConformerProcessing.hpp"

#include <algorithm>
#include <array>
#include <cstring>

#include "mlek/log/log_macros.h"

namespace arm::app {
template <class T>
ConformerPreProcess<T>::ConformerPreProcess(
    const std::shared_ptr<fwk::iface::TensorIface>& inputTensorMelSpec,
    const std::shared_ptr<fwk::iface::TensorIface>& inputTensorChunkSize,
    uint32_t melSpecWindowSize,
    uint32_t melSpecHopSize,
    uint32_t chunkSize) :
    m_inputTensorMelSpec{inputTensorMelSpec}, m_inputTensorChunkSize{inputTensorChunkSize},
    m_melSpecWindowSize{melSpecWindowSize}, m_melSpecHopSize{melSpecHopSize},
    m_chunkSize{chunkSize}, m_melSpectrogram{audio::ConformerMelSpectrogram(melSpecWindowSize)}
{
    this->m_slidingWindow =
        audio::SlidingWindow<const T>(this->m_melSpecWindowSize, this->m_melSpecHopSize);
}

template <class T>
bool ConformerPreProcess<T>::DoPreProcess(const void* input, size_t inputSize)
{
    if (!this->m_inputTensorMelSpec || !this->m_inputTensorChunkSize ||
        this->m_inputTensorMelSpec->Type() != fwk::iface::TensorType::FP32 ||
        this->m_inputTensorChunkSize->Type() != fwk::iface::TensorType::INT32 ||
        this->m_inputTensorChunkSize->GetNumElements() != 1 ||
        !this->m_inputTensorMelSpec->GetData() || !this->m_inputTensorChunkSize->GetData()) {
        printf_err("Invalid Conformer input tensor types or storage\n");
        return false;
    }
    const auto shape = this->m_inputTensorMelSpec->Shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != this->m_chunkSize ||
        shape[2] != audio::ConformerMelSpectrogram::ms_defaultNumFbankBins) {
        printf_err("Conformer input shape does not match preprocessing settings\n");
        return false;
    }

    // Generate full mel spectrogram with padding
    auto elementsWritten =
        this->GenerateMelSpectrogram(input,
                                     inputSize,
                                     static_cast<float*>(this->m_inputTensorMelSpec->GetData()),
                                     this->m_inputTensorMelSpec->GetNumElements());
    if (elementsWritten == 0) {
        return false;
    }
    auto paddingNeeded = this->m_inputTensorMelSpec->GetNumElements() - elementsWritten;
    std::fill_n(static_cast<float*>(this->m_inputTensorMelSpec->GetData()) + elementsWritten,
                paddingNeeded,
                -20.f);

    // Populate second input tensor with audio chunk size value
    auto chunkSizeTensorData = static_cast<int32_t*>(this->m_inputTensorChunkSize->GetData());
    *chunkSizeTensorData     = static_cast<int32_t>(this->m_chunkSize);

    return true;
}

template <class T>
size_t ConformerPreProcess<T>::GenerateMelSpectrogram(const void* input,
                                                      size_t inputSize,
                                                      float* output,
                                                      size_t outputSize)
{
    constexpr size_t numMels = audio::ConformerMelSpectrogram::ms_defaultNumFbankBins;
    if (!input || !output || this->m_melSpecWindowSize == 0 || this->m_melSpecHopSize == 0 ||
        inputSize < this->m_melSpecWindowSize) {
        printf_err("Invalid Conformer audio or spectrogram buffer\n");
        return 0;
    }
    const size_t numFrames = 1 + (inputSize - this->m_melSpecWindowSize) / this->m_melSpecHopSize;
    if (numFrames > outputSize / numMels) {
        printf_err("Conformer audio exceeds spectrogram capacity\n");
        return 0;
    }
    this->m_melSpectrogram.Init();
    this->m_slidingWindow.Reset(static_cast<const T*>(input), inputSize);
    auto writeFrom         = output;
    size_t elementsWritten = 0;
    while (this->m_slidingWindow.HasNext()) {
        const T* windowStart = this->m_slidingWindow.Next();
        const T* windowEnd   = windowStart + this->m_melSpecWindowSize;
        auto window          = std::vector(windowStart, windowEnd);
        auto melSpecVector   = this->m_melSpectrogram.ComputeMelSpec(window);
        if (melSpecVector.size() != numMels ||
            melSpecVector.size() > outputSize - elementsWritten) {
            printf_err("Invalid Conformer mel frame size\n");
            return 0;
        }
        memcpy(writeFrom, melSpecVector.data(), melSpecVector.size() * sizeof(float));
        writeFrom += melSpecVector.size();
        elementsWritten += melSpecVector.size();
    }
    return elementsWritten;
}

ConformerPostProcess::ConformerPostProcess(
    const std::shared_ptr<fwk::iface::TensorIface>& outputTensorLogits,
    const std::shared_ptr<fwk::iface::TensorIface>& outputTensorChunkSize,
    const std::vector<std::string>& labels,
    std::string& decodedResult) :
    m_outputTensorLogits{outputTensorLogits}, m_outputTensorChunkSize{outputTensorChunkSize},
    m_labels{labels}, m_decodedResult{decodedResult}
{}

uint32_t ConformerPostProcess::ArgMax1D(const float* logits, const size_t logitsSize)
{
    uint32_t argMax        = 0;
    float highestValueSeen = -FLT_MAX;
    for (size_t i = 0; i < logitsSize; i++) {
        auto logit = logits[i];
        if (logit > highestValueSeen) {
            highestValueSeen = logit;
            argMax           = i;
        }
    }
    return argMax;
}

std::vector<uint32_t> ConformerPostProcess::ArgMax2D(const float* logits,
                                                     const size_t logitsSize,
                                                     const std::array<size_t, 2>& logitsShape)
{
    size_t shapeProduct =
        std::accumulate(logitsShape.begin(), logitsShape.end(), 1, std::multiplies<size_t>());
    if (shapeProduct != logitsSize) {
        printf_err(
            "Provided shape of logits data does not match the specified size of the logits data\n");
        return {};
    }
    auto result = std::vector<uint32_t>(logitsShape[0]);
    for (size_t i = 0, j = 0; i < logitsSize; i += logitsShape[1], j++) {
        result[j] = ArgMax1D(logits + i, logitsShape[1]);
    }
    return result;
}

void ConformerPostProcess::Decode(const float* logits,
                                  const size_t logitsSize,
                                  const std::array<size_t, 2>& logitsShape,
                                  const std::vector<std::string>& labels,
                                  std::string& decodedResult)
{
    auto tokenIds = ArgMax2D(logits, logitsSize, logitsShape);
    std::vector<uint32_t> pieces;
    int32_t previous = -1;
    for (auto tokenId : tokenIds) {
        if (tokenId == 0) {
            previous = -1;
        } else if (static_cast<int32_t>(tokenId) != previous) {
            pieces.emplace_back(tokenId - 1);
            previous = static_cast<int32_t>(tokenId);
        }
    }

    auto decoded = std::vector<std::string>(pieces.size());
    std::transform(
        pieces.begin(), pieces.end(), decoded.begin(), [&](uint32_t i) { return labels[i]; });
    auto resultSize = std::accumulate(
        decoded.begin(), decoded.end(), 0, [](std::size_t acc, const std::string& s) {
            return acc + s.size();
        });

    decodedResult = "";
    decodedResult.reserve(resultSize);
    for (auto& piece : decoded) {
        decodedResult.append(piece);
    }

    size_t i              = 0;
    std::string toReplace = "▁";
    while ((i = decodedResult.find(toReplace, i)) != std::string::npos) {
        auto replacement = i == 0 ? "" : " ";
        decodedResult.replace(i, toReplace.size(), replacement);
        i++;
    }
}

bool ConformerPostProcess::DoPostProcess()
{
    this->m_decodedResult        = "";
    auto outputTensorLogitsShape = this->m_outputTensorLogits->Shape();
    auto nonUnitLengthDimensions = std::vector<size_t>();
    for (auto dimension : outputTensorLogitsShape) {
        if (dimension != 1) {
            nonUnitLengthDimensions.emplace_back(dimension);
        }
    }
    if (nonUnitLengthDimensions.size() != 2) {
        printf_err("Cannot decode output logits, tensor shape is not 2D\n");
        return false;
    }
    std::array<size_t, 2> shape2d;
    std::copy(nonUnitLengthDimensions.begin(), nonUnitLengthDimensions.end(), shape2d.data());
    Decode(static_cast<float*>(this->m_outputTensorLogits->GetData()),
           this->m_outputTensorLogits->GetNumElements(),
           shape2d,
           this->m_labels,
           this->m_decodedResult);
    return true;
}

template class ConformerPreProcess<float>;
template class ConformerPreProcess<int16_t>;
} // namespace arm::app
