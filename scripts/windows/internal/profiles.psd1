# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

@{
    Build = @{
        Firmware = 'internal/build-firmware.ps1'
        Vision = 'internal/build-vision-fvp-demo.ps1'
        VisionBenchmark = 'internal/build-vision-benchmark.ps1'
        AudioDemo = 'internal/build-fvp-audio-demo.ps1'
        FpgaFileDemo = 'internal/build-fpga-file-demo.ps1'
        HandPose = 'internal/build-hand-pose-fvp-demo.ps1'
        QrCode = 'internal/build-qrcode-fvp-demo.ps1'
        NoiseReduction = 'internal/build-noise-reduction-fvp-demo.ps1'
        KwsZh = 'internal/build-kws-zh.ps1'
        AsrZh = 'internal/build-asr-zh.ps1'
        AsrStageProfile = 'internal/build-asr-stage-profile.ps1'
        ConformerStageProfile = 'internal/build-conformer-stage-profile.ps1'
        ConformerUart = 'internal/build-conformer-uart-demo.ps1'
        Key0Runtime = 'internal/prepare-key0-runtime.ps1'
        PicoFace = 'internal/pico-face.ps1'
        TinyStoriesFvp = '../../source/app/use_case/tinystories/tools/build_fvp_demo.ps1'
        TinyStoriesFpga = '../../source/app/use_case/tinystories/tools/build_fpga_demo.ps1'
        TinyStoriesTest = '../../source/app/use_case/tinystories/tools/build_test.ps1'
        MiniMind2Test = '../../source/app/use_case/minimind2_small/tools/build_test.ps1'
    }
    Run = @{
        Firmware = 'internal/run-firmware-fvp.ps1'
        Vision = 'internal/run-vision-fvp-demo.ps1'
        VisionBenchmark = 'internal/run-vision-benchmark-fvp.ps1'
        AudioDemo = 'internal/run-fvp-audio-demo.ps1'
        ConformerUart = 'internal/run-conformer-uart-fvp.ps1'
        HandPose = 'internal/run-hand-pose-fvp-demo.ps1'
        QrCode = 'internal/run-qrcode-fvp-demo.ps1'
        NoiseReduction = 'internal/run-noise-reduction-fvp-demo.ps1'
        NoiseReductionDataset = 'internal/run-noise-reduction-dataset-fvp.ps1'
        PicoFace = 'internal/pico-face.ps1'
        TinyStories = '../../source/app/use_case/tinystories/tools/run_fvp_demo.ps1'
        TinyStoriesFloor1e3 = '../../source/app/use_case/tinystories/tools/run_fvp_demo_floor1e3.ps1'
        MiniMind2Test = '../../source/app/use_case/minimind2_small/tools/run_fvp_test.ps1'
    }
}
