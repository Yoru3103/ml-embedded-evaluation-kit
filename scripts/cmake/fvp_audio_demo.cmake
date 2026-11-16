# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)

function(mlek_add_fvp_audio_demo use_case)
    if (NOT TARGET_PLATFORM STREQUAL "mps4" OR NOT TARGET_SUBSYSTEM STREQUAL "sse-320"
        OR FPGA_PLATFORM_SSE_320 OR SEMIHOSTING_ENABLED OR NOT MPS4_DISPLAY_ENABLED)
        message(FATAL_ERROR "VSI audio demo requires SSE-320 FVP, UART stdout and LCD enabled")
    endif()
    if (NOT TARGET fvp_audio_input)
        add_library(fvp_audio_input STATIC ${MLEK_ROOT}/source/app/audio_demo/FvpAudioInput.cc)
        target_include_directories(fvp_audio_input PUBLIC
            ${MLEK_ROOT}/source/app/audio_demo/include
            ${MLEK_DEPENDENCY_ROOT_DIR}/avh/interface/include)
        target_link_libraries(fvp_audio_input PUBLIC hal)
    endif()
    set(${use_case}_LINK_LIBS ${${use_case}_LINK_LIBS} fvp_audio_input PARENT_SCOPE)
endfunction()
