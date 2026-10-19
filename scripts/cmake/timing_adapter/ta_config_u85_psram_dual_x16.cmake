# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Pair with scripts/vela/u85_psram_dual_x16.ini. Target NPU MHz is a modelling
# reference, not a command to change the physical FPGA or FVP clock.
set(PSRAM_TARGET_NPU_MHZ "800" CACHE STRING "PSRAM profile target NPU clock in MHz")
set_property(CACHE PSRAM_TARGET_NPU_MHZ PROPERTY STRINGS "400" "800")

if (NOT DEFINED TA_SRAM_BASE OR NOT DEFINED TA_EXT_BASE)
    message(FATAL_ERROR "TA_SRAM_BASE and TA_EXT_BASE must be defined by the platform.")
endif ()
if (DEFINED ETHOS_U_NPU_ID AND NOT ETHOS_U_NPU_ID STREQUAL "U85")
    message(FATAL_ERROR "The dual-x16 PSRAM profile requires an Ethos-U85.")
endif ()
if (DEFINED ETHOS_U_NPU_CONFIG_ID AND NOT ETHOS_U_NPU_CONFIG_ID STREQUAL "Z1024")
    message(FATAL_ERROR "The dual-x16 PSRAM profile assumes the Z1024 AXI configuration.")
endif ()
if (NOT ETHOS_U_NPU_MEMORY_MODE STREQUAL "Dedicated_Sram")
    message(FATAL_ERROR "The dual-x16 PSRAM profile requires Dedicated_Sram memory mode.")
endif ()

if (PSRAM_TARGET_NPU_MHZ STREQUAL "800")
    set(EXT_RLATENCY "80")
    set(EXT_WLATENCY "80")
    set(EXT_BWCAP "384")
elseif (PSRAM_TARGET_NPU_MHZ STREQUAL "400")
    set(EXT_RLATENCY "40")
    set(EXT_WLATENCY "40")
    set(EXT_BWCAP "768")
else ()
    message(FATAL_ERROR "PSRAM_TARGET_NPU_MHZ must be 400 or 800.")
endif ()

# Retain the existing reference SRAM timing in NPU cycles.
set(SRAM_MAXR "8")
set(SRAM_MAXW "8")
set(SRAM_MAXRW "0")
set(SRAM_RLATENCY "32")
set(SRAM_WLATENCY "32")
set(SRAM_PULSE_ON "3999")
set(SRAM_PULSE_OFF "1")
set(SRAM_BWCAP "4000")

# Per 128-bit EXT AXI: cap/(4095+1) equals Vela Dram_clock_scale.
# At the target frequency, each of the two AXIs models 1.2 GB/s.
# Caps are multiples of a 256-byte burst (16 beats), reducing cap rounding.
# MAXRW=0 adds no combined restriction beyond the separate MAXR and MAXW caps.
set(EXT_MAXR "2")
set(EXT_MAXW "2")
set(EXT_MAXRW "0")
set(EXT_PULSE_ON "4095")
set(EXT_PULSE_OFF "1")

foreach (TA_PREFIX IN ITEMS SRAM EXT)
    set(${TA_PREFIX}_PERFCTRL "0")
    set(${TA_PREFIX}_PERFCNT "0")
    set(${TA_PREFIX}_MODE "1")
    set(${TA_PREFIX}_HISTBIN "0")
    set(${TA_PREFIX}_HISTCNT "0")
endforeach ()

message(STATUS "PSRAM profile: target NPU ${PSRAM_TARGET_NPU_MHZ} MHz, "
    "2.4 GB/s total model bandwidth, 100 ns response latency")
