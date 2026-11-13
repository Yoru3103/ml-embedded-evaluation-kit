# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

set(${use_case}_ML_FRAMEWORK "${ML_FRAMEWORK_EXECUTORCH}")
if (NOT ${ML_FRAMEWORK} IN_LIST ${use_case}_ML_FRAMEWORK)
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif()

if (NOT TARGET_PLATFORM STREQUAL "mps4" OR
    NOT TARGET_SUBSYSTEM STREQUAL "sse-320" OR
    NOT ETHOS_U_NPU_ENABLED OR
    NOT ETHOS_U_NPU_ID STREQUAL "U85" OR
    NOT ETHOS_U_NPU_CONFIG_ID STREQUAL "Z1024" OR
    NOT ETHOS_U_NPU_MEMORY_MODE STREQUAL "Dedicated_Sram")
    message(FATAL_ERROR
        "minimind2_small requires MPS4 SSE-320 U85 Z1024 Dedicated_Sram")
endif()

# The PTE was compiled for a 512 KiB dedicated SRAM cache.
if (ETHOS_U_NPU_CACHE_SIZE LESS 524288)
    message(FATAL_ERROR
        "minimind2_small requires ETHOS_U_NPU_CACHE_SIZE >= 524288 (512 KiB)")
endif()

USER_OPTION(${use_case}_NPU_CLOCK_HZ
    "Configured NPU clock for cycle-to-ms conversion."
    50000000 STRING)
if (NOT ${use_case}_NPU_CLOCK_HZ MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "MiniMind2 NPU clock must be a positive integer in Hz")
endif()
list(APPEND ${use_case}_COMPILE_DEFS
    "MINIMIND_NPU_CLOCK_HZ=${${use_case}_NPU_CLOCK_HZ}")
if (NOT FPGA_PLATFORM_SSE_320)
    list(APPEND ${use_case}_COMPILE_DEFS MINIMIND_FVP)
endif()
set(${use_case}_supports_${ML_FRAMEWORK} ON)

string(CONCAT DEFAULT_MODEL_PATH
    "${MLEK_ROOT}/resources_downloaded/minimind2_small/"
    "minimind2_small_u85_1024_ctx64_w8a16_embedding8_linear.pte")
USER_OPTION(${use_case}_MODEL_PATH
    "U85 Z1024 512 KiB dedicated SRAM MiniMind2 PTE."
    "${DEFAULT_MODEL_PATH}" FILEPATH)
USER_OPTION(${use_case}_ACTIVATION_BUF_SZ
    "ExecuTorch method pool in DDR."
    0x00180000 STRING)
USER_OPTION(${use_case}_TOKENIZER_PATH
    "Exported MiniMind token byte table."
    "${MLEK_ROOT}/resources_downloaded/minimind2_small/tokenizer_bytes.bin" FILEPATH)
USER_OPTION(${use_case}_CACHE_RESET
    "Zero the persistent KV cache after load, before the first inference."
    ON BOOL)
if (NOT ${use_case}_CACHE_RESET)
    list(APPEND ${use_case}_COMPILE_DEFS MINIMIND_NO_CACHE_RESET)
endif()

if (NOT EXISTS "${${use_case}_MODEL_PATH}")
    message(FATAL_ERROR "MiniMind2 PTE not found: ${${use_case}_MODEL_PATH}")
endif()
if (NOT EXISTS "${${use_case}_TOKENIZER_PATH}")
    message(FATAL_ERROR "MiniMind tokenizer table not found: ${${use_case}_TOKENIZER_PATH}")
endif()

file(SIZE "${${use_case}_MODEL_PATH}" _minimind_pte_bytes)
file(SIZE "${${use_case}_TOKENIZER_PATH}" _minimind_tokenizer_bytes)
math(EXPR _minimind_method_pool_bytes "${${use_case}_ACTIVATION_BUF_SZ}")
math(EXPR _minimind_tmp_pool_bytes "${ML_FWK_TMP_MEM_SIZE}")
math(EXPR _minimind_static_ddr_bytes
    "${_minimind_pte_bytes} + ${_minimind_method_pool_bytes} + ${_minimind_tmp_pool_bytes} + ${_minimind_tokenizer_bytes} + 15")
math(EXPR _minimind_ddr_budget_bytes "64 * 1024 * 1024")
if (_minimind_static_ddr_bytes GREATER _minimind_ddr_budget_bytes)
    message(FATAL_ERROR
        "MiniMind2 model and static pools exceed the 64 MiB DDR budget: "
        "${_minimind_static_ddr_bytes} bytes")
endif()

math(EXPR _minimind_ddr_base "0x70000000")
math(EXPR _minimind_ddr_end "0x74000000")
math(EXPR _minimind_pte_end "${_minimind_ddr_base} + ${_minimind_pte_bytes}")
math(EXPR _minimind_method_base "(${_minimind_pte_end} + 15) / 16 * 16")
math(EXPR _minimind_method_end
    "${_minimind_method_base} + ${_minimind_method_pool_bytes}")
math(EXPR _minimind_tmp_pool_base "${ML_FWK_TMP_MEM_BASE}")
math(EXPR _minimind_tmp_pool_end
    "${_minimind_tmp_pool_base} + ${_minimind_tmp_pool_bytes}")
math(EXPR _minimind_tmp_alignment "${_minimind_tmp_pool_base} % 16")
math(EXPR _minimind_assets_end
    "${_minimind_method_end} + ${_minimind_tokenizer_bytes} + 15")
if (_minimind_assets_end GREATER _minimind_tmp_pool_base)
    message(FATAL_ERROR
        "MiniMind2 method pool and tokenizer overlap the temporary pool: "
        "assets end by ${_minimind_assets_end}, temp starts at ${_minimind_tmp_pool_base}")
endif()
if (_minimind_tmp_pool_base LESS _minimind_ddr_base OR
    _minimind_tmp_pool_end GREATER _minimind_ddr_end)
    message(FATAL_ERROR
        "MiniMind2 temporary pool [${_minimind_tmp_pool_base}, "
        "${_minimind_tmp_pool_end}) must fit in DDR [${_minimind_ddr_base}, "
        "${_minimind_ddr_end})")
endif()
if (NOT _minimind_tmp_alignment EQUAL 0)
    message(FATAL_ERROR "MiniMind2 temporary pool base must be 16-byte aligned")
endif()
message(STATUS
    "MiniMind2 DDR layout: PTE [0x70000000, ${_minimind_pte_end}), "
    "method [${_minimind_method_base}, ${_minimind_method_end}), "
    "temporary [${_minimind_tmp_pool_base}, ${_minimind_tmp_pool_end})")

# The model and method pool need the 64 MiB MPS4 DDR window.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_minimind_linker_suffix llvm)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_minimind_linker_suffix gnu)
else()
    message(FATAL_ERROR "minimind2_small requires the GNU or LLVM MPS4 linker script")
endif()
string(CONCAT _minimind_default_linker
    "${MLEK_ROOT}/scripts/cmake/platforms/mps4/sse-320/"
    "mps4-sse-320.${_minimind_linker_suffix}.ld")
file(READ "${_minimind_default_linker}" _minimind_linker_script)
set(_minimind_linker_original "${_minimind_linker_script}")
string(REPLACE
    "DDR   (rwx) : ORIGIN = 0x70000000, LENGTH = 0x02000000"
    "DDR   (rwx) : ORIGIN = 0x70000000, LENGTH = 0x04000000"
    _minimind_linker_script "${_minimind_linker_script}")
if ("${_minimind_linker_script}" STREQUAL "${_minimind_linker_original}")
    message(FATAL_ERROR "Could not expand the MPS4 DDR region for MiniMind2")
endif()
string(REPLACE "*(activation_buf_dram)"
    "*(activation_buf_dram)\n        . = ALIGN(16);\n        *(minimind_tokenizer)\n        ASSERT(. <= ${ML_FWK_TMP_MEM_BASE}, \"MiniMind tokenizer overlaps temporary pool\");"
    _minimind_linker_script "${_minimind_linker_script}")
string(FIND "${_minimind_linker_script}" "*(minimind_tokenizer)"
    _minimind_tokenizer_section)
if (_minimind_tokenizer_section LESS 0)
    message(FATAL_ERROR "Could not add the MiniMind tokenizer linker section")
endif()
set(_minimind_custom_linker
    "${CMAKE_BINARY_DIR}/mps4-sse-320-minimind2.${_minimind_linker_suffix}.ld")
file(WRITE "${_minimind_custom_linker}" "${_minimind_linker_script}")
set(${use_case}_LINKER_SCRIPT_OVERRIDE_PATH "${_minimind_custom_linker}")

# Validate the PTE with the same metadata checks as the other ExecuTorch use cases.
file(TO_CMAKE_PATH "${${use_case}_MODEL_PATH}" MINIMIND_MODEL_INC_PATH)
file(TO_CMAKE_PATH "${${use_case}_TOKENIZER_PATH}" MINIMIND_TOKENIZER_INC_PATH)
set(_minimind_pte_validation_script
    "${CMAKE_BINARY_DIR}/validate_minimind2_pte.py")
file(WRITE "${_minimind_pte_validation_script}"
    "from pathlib import Path\n"
    "from mlek_tools.validation.npu_validation import NpuValidationArgs\n"
    "from mlek_tools.validation.pte_validator import validate_pte_model\n"
    "validate_pte_model(Path(r'''${${use_case}_MODEL_PATH}'''), "
    "NpuValidationArgs(memory_mode='${ETHOS_U_NPU_MEMORY_MODE}', "
    "config='${ETHOS_U_NPU_CONFIG_ID}'))\n")
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${MLEK_GENERATOR_PYTHONPATH}"
        "${PYTHON}" "${_minimind_pte_validation_script}"
    RESULT_VARIABLE _minimind_validation_result)
if (NOT _minimind_validation_result EQUAL "0")
    message(FATAL_ERROR "MiniMind2 PTE validation failed")
endif()

configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/src/ModelData.cc.in"
    "${SRC_GEN_DIR}/MiniMindModelData.cc"
    @ONLY)
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/src/TokenizerData.cc.in"
    "${SRC_GEN_DIR}/MiniMindTokenizerData.cc"
    @ONLY)

add_library(minimind2_options INTERFACE)
USER_OPTION(minimind2_small_CONTEXT_LENGTH
    "Static MiniMind2 PTE context length in tokens (64 or 128)."
    64 STRING)
if (NOT minimind2_small_CONTEXT_LENGTH MATCHES "^(64|128)$")
    message(FATAL_ERROR "MiniMind2 context length must be 64 or 128 tokens")
endif()
target_compile_definitions(minimind2_options INTERFACE
    "MINIMIND_CONTEXT_LENGTH=${minimind2_small_CONTEXT_LENGTH}")
USER_OPTION(minimind2_small_DEMO "Enable the MiniMind UART demo." OFF BOOL)
if (minimind2_small_DEMO)
    if (NOT minimind2_small_INT16_KV OR NOT minimind2_small_PROFILE_STAGES
        OR NOT minimind2_small_CACHE_RESET)
        message(FATAL_ERROR "MiniMind demo requires INT16 KV, profiling, and cache reset")
    endif()
    target_compile_definitions(minimind2_options INTERFACE MINIMIND_DEMO)
endif()
USER_OPTION(minimind2_small_INT16_KV "Use 16-bit persistent KV cache buffers." OFF BOOL)
if (minimind2_small_INT16_KV)
    target_compile_definitions(minimind2_options INTERFACE MINIMIND_INT16_KV)
endif()
USER_OPTION(minimind2_small_PROFILE_STAGES "Measure synchronous Ethos-U driver stages." OFF BOOL)
if (minimind2_small_PROFILE_STAGES)
    if (NOT CPU_PROFILE_ENABLED)
        message(FATAL_ERROR "MiniMind stage profiling requires CPU_PROFILE_ENABLED=ON")
    endif()
    target_compile_definitions(minimind2_options INTERFACE MINIMIND_PROFILE_STAGES)
    target_link_options(minimind2_options INTERFACE "LINKER:--wrap=ethosu_invoke_v3")
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "MiniMind Q/DQ profiling requires the Arm LLVM toolchain")
    endif()
    if (minimind2_small_INT16_KV)
        target_link_options(minimind2_options INTERFACE "LINKER:--wrap=_ZN5torch8executor6native9clamp_outERN10executorch7runtime20KernelRuntimeContextERKNS3_7etensor6TensorERKNSt3__28optionalINS6_6ScalarEEESF_RS7_")
    endif()
    target_link_options(minimind2_options INTERFACE
        "LINKER:--wrap=_ZN5torch8executor6native23quantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS3_7etensor6TensorEdxxxNS6_10ScalarTypeERS7_"
        "LINKER:--wrap=_ZN5torch8executor6native25dequantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS3_7etensor6TensorEdxxxNS6_10ScalarTypeENSt3__28optionalISA_EERS7_"
        "LINKER:--wrap=_ZN8cortex_m6native23quantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS2_7etensor6TensorEdxxxNS5_10ScalarTypeERS6_"
        "LINKER:--wrap=_ZN8cortex_m6native25dequantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS2_7etensor6TensorEdxxxNS5_10ScalarTypeERS6_"
    )
endif()
target_include_directories(minimind2_options INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/src")
if (DEFINED minimind2_small_CACHE_OFFSETS)
    target_compile_definitions(minimind2_options INTERFACE
        "MINIMIND_CACHE_OFFSETS=${minimind2_small_CACHE_OFFSETS}")
endif()
list(APPEND ${use_case}_LINK_LIBS minimind2_options ml_framework_et)

if (COMMAND generate_pte_ops_lib)
    generate_pte_ops_lib(
        MODEL_PATH "${${use_case}_MODEL_PATH}"
        LIB_NAME "${use_case}_portable_ops_lib"
        SELECT_OPS_LIST "")
    if (TARGET ${use_case}_portable_ops_lib)
        list(APPEND ${use_case}_LINK_LIBS ${use_case}_portable_ops_lib)
    endif()
endif()

# Bare-metal ATFE libc++ does not implement std::random_device. These portable
# operators are absent from this PTE, so omit their otherwise unconditional
# build from ExecuTorch's generic portable kernel library.
if (TARGET portable_kernels)
    get_target_property(_minimind_portable_sources portable_kernels SOURCES)
    list(FILTER _minimind_portable_sources EXCLUDE REGEX
        "op_(native_dropout|rand|randn)\\.cpp$")
    set_property(TARGET portable_kernels PROPERTY SOURCES
        "${_minimind_portable_sources}")
endif()
