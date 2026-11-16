/*
 * SPDX-FileCopyrightText: Copyright 2021-2022, 2024-2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
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
#include "UseCaseHandler.hpp"
#include "mlek/common/AudioSlidingWindow.hpp"
#include "mlek/common/ImageUtils.hpp"
#include "mlek/use_case/noise_reduction/RNNoiseFeatureProcessor.hpp"
#include "mlek/fwk/tflm/RNNoiseModel.hpp"
#include "mlek/use_case/noise_reduction/RNNoiseProcessing.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "mlek/log/log_macros.h"

#include <cstring>
#include <limits>
#include <memory>

#ifdef NOISE_REDUCTION_SAVE_WAV
#include <cstdio>
#endif

namespace arm {
namespace app {

#ifdef NOISE_REDUCTION_SAVE_WAV
    static bool WriteWavHeader(std::FILE* file, uint32_t pcmBytes)
    {
        // RIFF PCM, mono, 48 kHz, 16 bits. Sizes are stored in little-endian order.
        uint8_t header[] = {
            'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E',
            'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 1, 0,
            0x80, 0xbb, 0, 0, 0, 0x77, 1, 0, 2, 0, 16, 0,
            'd', 'a', 't', 'a', 0, 0, 0, 0
        };
        for (size_t i = 0; i < sizeof(uint32_t); ++i) {
            header[4 + i]  = static_cast<uint8_t>((pcmBytes + 36) >> (8 * i));
            header[40 + i] = static_cast<uint8_t>(pcmBytes >> (8 * i));
        }
        return std::fwrite(header, 1, sizeof(header), file) == sizeof(header);
    }
#endif

    /* Noise reduction inference handler. */
    bool NoiseReductionHandler(ApplicationContext& ctx)
    {
        constexpr uint32_t dataPsnTxtInfStartX = 20;
        constexpr uint32_t dataPsnTxtInfStartY = 40;

        /* Variables used for memory dumping. */
        size_t memDumpMaxLen            = 0;
        uint8_t* memDumpBaseAddr        = nullptr;
        size_t undefMemDumpBytesWritten = 0;
        size_t* pMemDumpBytesWritten    = &undefMemDumpBytesWritten;
        if (ctx.Has("MEM_DUMP_LEN") && ctx.Has("MEM_DUMP_BASE_ADDR") &&
            ctx.Has("MEM_DUMP_BYTE_WRITTEN")) {
            memDumpMaxLen        = ctx.Get<size_t>("MEM_DUMP_LEN");
            memDumpBaseAddr      = ctx.Get<uint8_t*>("MEM_DUMP_BASE_ADDR");
            pMemDumpBytesWritten = ctx.Get<size_t*>("MEM_DUMP_BYTE_WRITTEN");
        }
        std::reference_wrapper<size_t> memDumpBytesWritten = std::ref(*pMemDumpBytesWritten);
        auto& profiler                                     = ctx.Get<Profiler&>("profiler");

        /* Get model reference. */
        auto& model = ctx.Get<fwk::tflm::RNNoiseModel&>("model");
        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

        /* Populate Pre-Processing related parameters. */
        auto audioFrameLen      = ctx.Get<uint32_t>("frameLength");
        auto audioFrameStride   = ctx.Get<uint32_t>("frameStride");
        auto nrNumInputFeatures = ctx.Get<uint32_t>("numInputFeatures");

        auto inputTensor = model.GetInputTensor(0);
        if (nrNumInputFeatures != inputTensor->Bytes()) {
            printf_err("Input features size must be equal to input tensor size."
                       " Feature size = %" PRIu32 ", Tensor size = %zu.\n",
                       nrNumInputFeatures,
                       inputTensor->Bytes());
            return false;
        }

        if (memDumpMaxLen > 0 && !memDumpBaseAddr) {
            return false;
        }

        auto outputTensor = model.GetOutputTensor(model.m_indexForModelOutput);

        hal_audio_init();
        if (!hal_audio_configure(HAL_AUDIO_MODE_SINGLE_BURST,
                                 HAL_AUDIO_FORMAT_48KHZ_MONO_16BIT)) {
            printf_err("Failed to configure audio\n");
            return false;
        }

        uint32_t loopCount = 0;

        while (true) {
#ifdef INTERACTIVE_MODE
            AwaitUserInput(); // Wait for user input before moving forward.
#endif /* INTERACTIVE_MODE */

            uint32_t nElements = 0;
            hal_audio_start();
            auto audioData = hal_audio_get_captured_frame(&nElements);
            if (!nElements || !audioData) {
                debug("End of stream\n");
                break;
            }

            hal_display_clear(COLOR_BLACK);

            /* Creating a sliding window through the audio. */
            auto audioDataSlider =
                audio::SlidingWindow<const int16_t>(audioData,
                                                    nElements,
                                                    audioFrameLen,
                                                    audioFrameStride);
            if (!audioDataSlider.HasNext()) {
                printf_err("Audio clip is shorter than one inference frame.\n");
                return false;
            }

            const size_t frameCount = audioDataSlider.TotalStrides() + 1;
            const uint64_t pcmBytes = uint64_t{frameCount} * audioFrameLen * sizeof(int16_t);
            const std::string audioFileName = "loop" + std::to_string(loopCount++);

            const size_t dumpHeaderSize = 2 * sizeof(int32_t) + audioFileName.size();
            size_t dumpWriteOffset = memDumpBytesWritten;
            if (memDumpMaxLen > 0) {
                const size_t overhead = dumpHeaderSize + sizeof(int32_t);
                if (pcmBytes > std::numeric_limits<int32_t>::max() ||
                    memDumpBytesWritten > memDumpMaxLen ||
                    overhead + pcmBytes > memDumpMaxLen - memDumpBytesWritten) {
                    printf_err("Denoised audio exceeds the configured memory dump size.\n");
                    return false;
                }

                /* Leave the previous end marker in place until this clip is complete. */
                dumpWriteOffset += dumpHeaderSize;
            }

#ifdef NOISE_REDUCTION_SAVE_WAV
            if (pcmBytes > std::numeric_limits<uint32_t>::max() - 36) {
                printf_err("Noise reduction output exceeds the RIFF WAV size limit.\n");
                return false;
            }
            const std::string wavName = "denoised_" + audioFileName + ".wav";
            std::unique_ptr<std::FILE, decltype(&std::fclose)> wav(
                std::fopen(wavName.c_str(), "wb"), &std::fclose);
            if (!wav || !WriteWavHeader(wav.get(), static_cast<uint32_t>(pcmBytes))) {
                printf_err("Failed to open or write WAV header for %s.\n", wavName.c_str());
                return false;
            }
#endif

            /* Set up pre and post-processing. */
            std::shared_ptr<rnn::RNNoiseFeatureProcessor> featureProcessor =
                std::make_shared<rnn::RNNoiseFeatureProcessor>();
            std::shared_ptr<rnn::FrameFeatures> frameFeatures =
                std::make_shared<rnn::FrameFeatures>();

            RNNoisePreProcess preProcess =
                RNNoisePreProcess(inputTensor, featureProcessor, frameFeatures);

            std::vector<int16_t> denoisedAudioFrame(audioFrameLen);
            RNNoisePostProcess postProcess = RNNoisePostProcess(
                outputTensor, denoisedAudioFrame, featureProcessor, frameFeatures);

            bool resetGRU = true;

            while (audioDataSlider.HasNext()) {
                const int16_t* inferenceWindow = audioDataSlider.Next();

                if (!preProcess.DoPreProcess(inferenceWindow, audioFrameLen)) {
                    printf_err("Pre-processing failed.");
                    return false;
                }

                /* Reset or copy over GRU states first to avoid TFLu memory overlap issues. */
                if (resetGRU) {
                    model.ResetGruState();
                } else {
                    /* Copying gru state outputs to gru state inputs.
                     * Call ResetGruState in between the sequence of inferences on unrelated input
                     * data. */
                    model.CopyGruStates();
                }

                /* Strings for presentation/logging. */
                std::string str_inf{"Running inference... "};

                /* Display message on the LCD - inference running. */
                hal_display_show_text(str_inf.c_str(),
                                     str_inf.size(),
                                     dataPsnTxtInfStartX,
                                     dataPsnTxtInfStartY,
                                     false);

                info("Inference %zu/%zu\n",
                     audioDataSlider.Index() + 1,
                     frameCount);

                /* Run inference over this feature sliding window. */
                if (!RunInference(model, profiler)) {
                    printf_err("Inference failed.");
                    return false;
                }
                resetGRU = false;

                /* Carry out post-processing. */
                if (!postProcess.DoPostProcess()) {
                    printf_err("Post-processing failed.");
                    return false;
                }

                /* Erase. */
                str_inf = std::string(str_inf.size(), ' ');
                hal_display_show_text(str_inf.c_str(),
                                     str_inf.size(),
                                     dataPsnTxtInfStartX,
                                     dataPsnTxtInfStartY,
                                     false);

#ifdef NOISE_REDUCTION_SAVE_WAV
                const size_t samplesWritten = std::fwrite(
                    denoisedAudioFrame.data(), sizeof(int16_t), denoisedAudioFrame.size(), wav.get());
                if (samplesWritten != denoisedAudioFrame.size()) {
                    printf_err("Failed to write PCM data to %s.\n", wavName.c_str());
                    return false;
                }
#endif

                if (memDumpMaxLen > 0) {
                    /* Dump final post processed output to memory. */
                    dumpWriteOffset += DumpOutputDenoisedAudioFrame(
                        denoisedAudioFrame, memDumpBaseAddr + dumpWriteOffset,
                        memDumpMaxLen - dumpWriteOffset);
                }
            }

#ifdef NOISE_REDUCTION_SAVE_WAV
            if (std::fclose(wav.release()) != 0) {
                printf_err("Failed to close %s.\n", wavName.c_str());
                return false;
            }
            info("Wrote %s\n", wavName.c_str());
#endif

            if (memDumpMaxLen > 0) {
                DumpDenoisedAudioHeader(audioFileName.c_str(), pcmBytes / sizeof(int16_t),
                                        memDumpBaseAddr + memDumpBytesWritten,
                                        memDumpMaxLen - memDumpBytesWritten);
                memDumpBytesWritten = dumpWriteOffset;
                /* The next clip overwrites this marker if there is room for it. */
                const size_t footerSize = DumpDenoisedAudioFooter(
                    memDumpBaseAddr + memDumpBytesWritten, memDumpMaxLen - memDumpBytesWritten);
                info("Output memory dump of %zu bytes written at address %p\n",
                     memDumpBytesWritten + footerSize, memDumpBaseAddr);
            }

            info("All inferences for audio clip complete.\n");
            profiler.PrintProfilingResult();

            std::string clearString{' '};
            hal_display_show_text(clearString.c_str(),
                                 clearString.size(),
                                 dataPsnTxtInfStartX,
                                 dataPsnTxtInfStartY,
                                 false);

            std::string completeMsg{"Inference complete!"};

            /* Display message on the LCD - inference complete. */
            hal_display_show_text(completeMsg.c_str(),
                                 completeMsg.size(),
                                 dataPsnTxtInfStartX,
                                 dataPsnTxtInfStartY,
                                 false);
        }

        return true;
    }

    size_t DumpDenoisedAudioHeader(const char* filename,
                                   size_t dumpSize,
                                   uint8_t* memAddress,
                                   size_t memSize)
    {

        const size_t filenameSize = std::strlen(filename);
        const size_t headerSize = 2 * sizeof(int32_t) + filenameSize;
        if (!memAddress || memSize < headerSize ||
            filenameSize > std::numeric_limits<int32_t>::max() ||
            dumpSize > std::numeric_limits<int32_t>::max() / sizeof(int16_t)) {
            return 0;
        }

        const auto filenameLength = static_cast<int32_t>(filenameSize);
        const auto dumpSizeBytes = static_cast<int32_t>(dumpSize * sizeof(int16_t));
        std::memcpy(memAddress, &filenameLength, sizeof(filenameLength));
        std::memcpy(memAddress + sizeof(filenameLength), filename, filenameSize);
        std::memcpy(memAddress + sizeof(filenameLength) + filenameSize,
                    &dumpSizeBytes, sizeof(dumpSizeBytes));
        return headerSize;
    }

    size_t DumpDenoisedAudioFooter(uint8_t* memAddress, size_t memSize)
    {
        if ((memAddress == nullptr) || (memSize < 4)) {
            return 0;
        }
        const int32_t eofMarker = -1;
        std::memcpy(memAddress, &eofMarker, sizeof(int32_t));

        return sizeof(int32_t);
    }

    size_t DumpOutputDenoisedAudioFrame(const std::vector<int16_t>& audioFrame,
                                        uint8_t* memAddress,
                                        size_t memSize)
    {
        if (memAddress == nullptr) {
            return 0;
        }

        size_t numByteToBeWritten = audioFrame.size() * sizeof(int16_t);
        if (numByteToBeWritten > memSize) {
            printf_err("Overflow error: Writing %zu of %zu bytes to memory @ 0x%p.\n",
                       memSize,
                       numByteToBeWritten,
                       memAddress);
            numByteToBeWritten = memSize;
        }

        std::memcpy(memAddress, audioFrame.data(), numByteToBeWritten);
        info("Copied %zu bytes to %p\n", numByteToBeWritten, memAddress);

        return numByteToBeWritten;
    }

    size_t
    DumpOutputTensorsToMemory(fwk::iface::Model& model, uint8_t* memAddress, const size_t memSize)
    {
        const size_t numOutputs = model.GetNumOutputs();
        size_t numBytesWritten  = 0;
        uint8_t* ptr            = memAddress;

        /* Iterate over all output tensors. */
        for (size_t i = 0; i < numOutputs; ++i) {
            auto tensor         = model.GetOutputTensor(i);
            const auto* tData   = tensor->GetData<uint8_t>();
            const size_t nBytes = tensor->Bytes();
#if VERIFY_TEST_OUTPUT
            DumpTensor(tensor);
#endif /* VERIFY_TEST_OUTPUT */
            /* Ensure that we don't overflow the allowed limit. */
            if (numBytesWritten + nBytes <= memSize) {
                if (nBytes > 0) {
                    std::memcpy(ptr, tData, nBytes);

                    info("Copied %zu bytes for tensor %zu to 0x%p\n", nBytes, i, ptr);

                    numBytesWritten += nBytes;
                    ptr += nBytes;
                }
            } else {
                printf_err("Error writing tensor %zu to memory @ 0x%p\n", i, memAddress);
                break;
            }
        }

        info("%zu bytes written to memory @ 0x%p\n", numBytesWritten, memAddress);

        return numBytesWritten;
    }
} /* namespace app */
} /* namespace arm */
