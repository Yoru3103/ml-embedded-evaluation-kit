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
    message(FATAL_ERROR "tinystories requires MPS4 SSE-320 U85 Z1024 Dedicated_Sram")
endif()

if (NOT CPU_PROFILE_ENABLED)
    message(FATAL_ERROR "TinyStories per-token timing requires CPU_PROFILE_ENABLED=ON")
endif()
USER_OPTION(${use_case}_NPU_CLOCK_HZ "Configured NPU clock for cycle-to-ms conversion."
    50000000 STRING)
if (NOT ${use_case}_NPU_CLOCK_HZ MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "TinyStories NPU clock must be a positive integer in Hz")
endif()
list(APPEND ${use_case}_COMPILE_DEFS
    "TINYSTORIES_NPU_CLOCK_HZ=${${use_case}_NPU_CLOCK_HZ}")
set(${use_case}_supports_${ML_FRAMEWORK} ON)
USER_OPTION(${use_case}_MODE "TinyStories application mode: TEST or DEMO." TEST STRING)
if (NOT ${use_case}_MODE MATCHES "^(TEST|DEMO)$")
    message(FATAL_ERROR "tinystories_MODE must be TEST or DEMO")
endif()
if (${use_case}_MODE STREQUAL "DEMO")
    list(APPEND ${use_case}_COMPILE_DEFS TINYSTORIES_DEMO)
endif()
if (NOT FPGA_PLATFORM_SSE_320)
    list(APPEND ${use_case}_COMPILE_DEFS TINYSTORIES_FVP)
endif()
string(CONCAT DEFAULT_MODEL_PATH "${MLEK_ROOT}/resources_downloaded/tinystories42m/"
    "npu_int16_kv_ctx128_1200/"
    "tinystories42m_u85_1024_ctx128_int16_kv_1200.pte")
USER_OPTION(${use_case}_MODEL_PATH "U85 Z1024 Dedicated SRAM TinyStories PTE."
    "${DEFAULT_MODEL_PATH}" FILEPATH)
USER_OPTION(${use_case}_CONTEXT "Exported KV context length; must match the PTE."
    128 INT)
list(APPEND ${use_case}_COMPILE_DEFS TINYSTORIES_CONTEXT=${${use_case}_CONTEXT})
USER_OPTION(${use_case}_LOGIT_DUMP "Print the top five logits for every inference."
    OFF BOOL)
if (${${use_case}_LOGIT_DUMP})
    list(APPEND ${use_case}_COMPILE_DEFS TINYSTORIES_LOGIT_DUMP)
endif()
USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "ExecuTorch method pool in DDR."
    0x00700000 STRING)

if (NOT EXISTS "${${use_case}_MODEL_PATH}")
    message(FATAL_ERROR "TinyStories PTE not found: ${${use_case}_MODEL_PATH}")
endif()
USER_OPTION(${use_case}_TOKENIZER_PATH "Exported complete SentencePiece token surfaces."
    "${MLEK_ROOT}/resources_downloaded/tinystories42m/tokenizer_surfaces.bin" FILEPATH)
if (NOT EXISTS "${${use_case}_TOKENIZER_PATH}")
    message(FATAL_ERROR "Export tokenizer_surfaces.bin with tinystories/tools/export_tokenizer.py")
endif()
file(SIZE "${${use_case}_TOKENIZER_PATH}" _tinystories_tokenizer_bytes)
if (${use_case}_MODE STREQUAL "DEMO")
    USER_OPTION(${use_case}_ENCODER_PATH "Exported SentencePiece BPE encoder table."
        "${MLEK_ROOT}/resources_downloaded/tinystories42m/tokenizer_encoder.bin" FILEPATH)
    if (NOT EXISTS "${${use_case}_ENCODER_PATH}")
        message(FATAL_ERROR "Export tokenizer_encoder.bin with tools/export_encoder.py")
    endif()
    file(SIZE "${${use_case}_ENCODER_PATH}" _tinystories_encoder_bytes)
    math(EXPR _tinystories_tokenizer_bytes
        "${_tinystories_tokenizer_bytes} + ${_tinystories_encoder_bytes} + 15")
    file(TO_CMAKE_PATH "${${use_case}_ENCODER_PATH}" TINYSTORIES_ENCODER_INC_PATH)
    configure_file("${CMAKE_CURRENT_LIST_DIR}/src/EncoderData.cc.in"
        "${SRC_GEN_DIR}/TinyStoriesEncoderData.cc" @ONLY)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${MLEK_GENERATOR_PYTHONPATH}"
        "${PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/tools/demo_cache_contract.py"
        "${${use_case}_MODEL_PATH}" "${INC_GEN_DIR}/TinyStoriesCacheContract.hpp"
        --context "${${use_case}_CONTEXT}"
        RESULT_VARIABLE _tinystories_demo_contract_result)
    if (NOT _tinystories_demo_contract_result EQUAL "0")
        message(FATAL_ERROR "TinyStories DEMO requires the validated INT16 KV model")
    endif()
endif()
file(TO_CMAKE_PATH "${${use_case}_TOKENIZER_PATH}" TINYSTORIES_TOKENIZER_INC_PATH)
configure_file("${CMAKE_CURRENT_LIST_DIR}/src/TokenizerData.cc.in"
    "${SRC_GEN_DIR}/TinyStoriesTokenizerData.cc" @ONLY)
add_library(tinystories_options INTERFACE)
target_include_directories(tinystories_options INTERFACE "${CMAKE_CURRENT_LIST_DIR}/src")
target_link_options(tinystories_options INTERFACE
    "LINKER:--wrap=ethosu_invoke_v3" "LINKER:--wrap=arm_ethos_io_memcpy")
list(APPEND ${use_case}_LINK_LIBS tinystories_options)
file(SIZE "${${use_case}_MODEL_PATH}" _tinystories_pte_bytes)
math(EXPR _tinystories_method_pool_bytes "${${use_case}_ACTIVATION_BUF_SZ}")
math(EXPR _tinystories_tmp_pool_bytes "${ML_FWK_TMP_MEM_SIZE}")
math(EXPR _tinystories_static_ddr_bytes
    "${_tinystories_pte_bytes} + ${_tinystories_method_pool_bytes} + ${_tinystories_tmp_pool_bytes} + ${_tinystories_tokenizer_bytes} + 15")
math(EXPR _tinystories_ddr_budget_bytes "64 * 1024 * 1024")
if (_tinystories_static_ddr_bytes GREATER _tinystories_ddr_budget_bytes)
    message(FATAL_ERROR
        "TinyStories model and static pools exceed the 64 MiB DDR budget: "
        "${_tinystories_static_ddr_bytes} bytes")
endif()
message(STATUS "TinyStories static DDR budget: ${_tinystories_static_ddr_bytes} / "
    "${_tinystories_ddr_budget_bytes} bytes")

# Keep the model, method pool, and temporary pool inside the same
# 64 MiB DDR window. The platform default temporary base (0x72000000) overlaps
# the statically linked PTE, so validate actual address placement as well as
# aggregate capacity.
if (TARGET_PLATFORM STREQUAL "mps4")
    if (MPS4_DISPLAY_ENABLED)
        message(FATAL_ERROR "TinyStories FPGA build must disable the unused MPS4 display")
    endif()

    math(EXPR _tinystories_ddr_base "0x70000000")
    math(EXPR _tinystories_ddr_end "0x74000000")
    math(EXPR _tinystories_pte_end
        "${_tinystories_ddr_base} + ${_tinystories_pte_bytes}")
    math(EXPR _tinystories_method_pool_base "(${_tinystories_pte_end} + 15) / 16 * 16")
    math(EXPR _tinystories_method_pool_end
        "${_tinystories_method_pool_base} + ${_tinystories_method_pool_bytes}")
    math(EXPR _tinystories_tmp_pool_base "${ML_FWK_TMP_MEM_BASE}")
    math(EXPR _tinystories_tmp_pool_end
        "${_tinystories_tmp_pool_base} + ${_tinystories_tmp_pool_bytes}")
    math(EXPR _tinystories_tmp_alignment "${_tinystories_tmp_pool_base} % 16")

    math(EXPR _tinystories_assets_end "${_tinystories_method_pool_end} + ${_tinystories_tokenizer_bytes} + 15")
    if (_tinystories_assets_end GREATER _tinystories_tmp_pool_base)
        message(FATAL_ERROR
            "TinyStories FPGA method pool overlaps the temporary pool: "
            "method plus tokenizer ends by ${_tinystories_assets_end}, temp starts at "
            "${_tinystories_tmp_pool_base}")
    endif()
    if (_tinystories_tmp_pool_base LESS _tinystories_ddr_base OR
        _tinystories_tmp_pool_end GREATER _tinystories_ddr_end)
        message(FATAL_ERROR
            "TinyStories FPGA temporary pool [${_tinystories_tmp_pool_base}, "
            "${_tinystories_tmp_pool_end}) must fit in DDR [${_tinystories_ddr_base}, "
            "${_tinystories_ddr_end})")
    endif()
    if (NOT _tinystories_tmp_alignment EQUAL 0)
        message(FATAL_ERROR "TinyStories FPGA temporary pool base must be 16-byte aligned")
    endif()

    math(EXPR _tinystories_pte_end_hex "${_tinystories_pte_end}" OUTPUT_FORMAT HEXADECIMAL)
    math(EXPR _tinystories_method_pool_base_hex
        "${_tinystories_method_pool_base}" OUTPUT_FORMAT HEXADECIMAL)
    math(EXPR _tinystories_method_pool_end_hex
        "${_tinystories_method_pool_end}" OUTPUT_FORMAT HEXADECIMAL)
    math(EXPR _tinystories_tmp_pool_end_hex
        "${_tinystories_tmp_pool_end}" OUTPUT_FORMAT HEXADECIMAL)
    message(STATUS
        "TinyStories FPGA DDR layout: PTE [0x70000000, ${_tinystories_pte_end_hex}), "
        "method [${_tinystories_method_pool_base_hex}, ${_tinystories_method_pool_end_hex}), "
        "temporary [${ML_FWK_TMP_MEM_BASE}, ${_tinystories_tmp_pool_end_hex})")
endif()

# Validate the PTE with the same metadata checks used by generate_model_code,
# but avoid expanding its 56 MiB binary into a several-hundred-MiB C++ array.
file(TO_CMAKE_PATH "${${use_case}_MODEL_PATH}" TINYSTORIES_MODEL_INC_PATH)
set(_tinystories_pte_validation_script
    "${CMAKE_BINARY_DIR}/validate_tinystories_pte.py")
file(WRITE "${_tinystories_pte_validation_script}"
    "from pathlib import Path\n"
    "from mlek_tools.validation.npu_validation import NpuValidationArgs\n"
    "from mlek_tools.validation.pte_validator import validate_pte_model\n"
    "validate_pte_model(Path(r'''${${use_case}_MODEL_PATH}'''), "
    "NpuValidationArgs(memory_mode='${ETHOS_U_NPU_MEMORY_MODE}', "
    "config='${ETHOS_U_NPU_CONFIG_ID}'))\n")
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${MLEK_GENERATOR_PYTHONPATH}"
        "${PYTHON}" "${_tinystories_pte_validation_script}"
    RESULT_VARIABLE _tinystories_validation_result
)
if (NOT _tinystories_validation_result EQUAL "0")
    message(FATAL_ERROR "TinyStories PTE validation failed")
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${MLEK_GENERATOR_PYTHONPATH}"
        "${PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/tools/validate_embedding_indices.py"
        "${${use_case}_MODEL_PATH}"
    RESULT_VARIABLE _tinystories_index_validation_result
)
if (NOT _tinystories_index_validation_result EQUAL "0")
    message(FATAL_ERROR "TinyStories CPU embedding index validation failed")
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${MLEK_GENERATOR_PYTHONPATH}"
        "${PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/tools/validate_delegate_io.py"
        "${${use_case}_MODEL_PATH}" --context "${${use_case}_CONTEXT}"
    RESULT_VARIABLE _tinystories_io_validation_result
)
if (NOT _tinystories_io_validation_result EQUAL "0")
    message(FATAL_ERROR "TinyStories delegate I/O validation failed")
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${MLEK_GENERATOR_PYTHONPATH}"
        "${PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/tools/profile_contract.py"
        "${${use_case}_MODEL_PATH}" "${INC_GEN_DIR}/TinyStoriesProfileContract.hpp"
    RESULT_VARIABLE _tinystories_profile_contract_result)
if (NOT _tinystories_profile_contract_result EQUAL "0")
    message(FATAL_ERROR "TinyStories profiling contract generation failed")
endif()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/src/ModelData.cc.in"
    "${SRC_GEN_DIR}/TinyStoriesModelData.cc"
    @ONLY
)

if (COMMAND generate_pte_ops_lib)
    generate_pte_ops_lib(
        MODEL_PATH "${${use_case}_MODEL_PATH}"
        LIB_NAME "${use_case}_portable_ops_lib"
        SELECT_OPS_LIST "")
    if (TARGET ${use_case}_portable_ops_lib)
        # Keep inline EventTracer hooks consistent with the traced runtime.
        target_compile_definitions(${use_case}_portable_ops_lib PRIVATE ET_EVENT_TRACER_ENABLED)
        list(APPEND ${use_case}_LINK_LIBS ${use_case}_portable_ops_lib)
    endif()
endif()

# Bare-metal ATFE libc++ does not implement std::random_device. These portable
# operators are absent from this PTE, so omit their otherwise unconditional
# build from ExecuTorch's generic portable kernel library.
if (TARGET portable_kernels)
    get_target_property(_tinystories_portable_sources portable_kernels SOURCES)
    list(FILTER _tinystories_portable_sources EXCLUDE REGEX
        "op_(native_dropout|rand|randn)\\.cpp$")
    set_property(TARGET portable_kernels PROPERTY SOURCES
        "${_tinystories_portable_sources}")
endif()

# The INT16-state candidate uses standard integer clamp operations to preserve
# the original Q(DQ(-32768)) saturation rule. Optimize only this newly added
# integer kernel; retain the existing build settings for all other operators.
USER_OPTION(${use_case}_CACHE_CLAMP_OPTIMIZED "Optimize the integer cache clamp kernel."
    OFF BOOL)
if (${${use_case}_CACHE_CLAMP_OPTIMIZED} AND TARGET portable_kernels)
    set_property(SOURCE "${EXECUTORCH_SRC_PATH}/kernels/portable/cpu/op_clamp.cpp"
        TARGET_DIRECTORY portable_kernels APPEND PROPERTY COMPILE_OPTIONS -O2)
endif()
USER_OPTION(${use_case}_CACHE_CLAMP_SPECIALIZED "Use the in-place INT16 KV clamp loop."
    OFF BOOL)
if (${${use_case}_CACHE_CLAMP_SPECIALIZED} AND TARGET portable_kernels)
    list(APPEND ${use_case}_COMPILE_DEFS TINYSTORIES_CACHE_CLAMP_SPECIALIZED)
    # Generate a use-case-only copy; leave the ExecuTorch submodule unchanged.
    set(_tinystories_clamp_source
        "${EXECUTORCH_SRC_PATH}/kernels/portable/cpu/op_clamp.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_tinystories_clamp_source}" "${CMAKE_CURRENT_LIST_DIR}/src/CacheClampFastPath.inc")
    file(READ "${_tinystories_clamp_source}" _tinystories_clamp_original)
    file(READ "${CMAKE_CURRENT_LIST_DIR}/src/CacheClampFastPath.inc" _tinystories_clamp_hook)
    set(_tinystories_clamp_anchor "Tensor& clamp_out(")
    string(FIND "${_tinystories_clamp_original}" "${_tinystories_clamp_anchor}"
        _tinystories_clamp_begin)
    if (_tinystories_clamp_begin LESS 0)
        message(FATAL_ERROR "TinyStories cannot locate the portable clamp function")
    endif()
    string(SUBSTRING "${_tinystories_clamp_original}" 0 ${_tinystories_clamp_begin}
        _tinystories_clamp_prefix)
    string(SUBSTRING "${_tinystories_clamp_original}" ${_tinystories_clamp_begin} -1
        _tinystories_clamp_body)
    set(_tinystories_clamp_signature "    Tensor& out) {\n  bool has_min")
    string(FIND "${_tinystories_clamp_body}" "${_tinystories_clamp_signature}"
        _tinystories_clamp_match)
    if (_tinystories_clamp_match LESS 0)
        message(FATAL_ERROR "TinyStories portable clamp signature has changed")
    endif()
    # Replace only the first occurrence (clamp_tensor_out must remain generic).
    string(SUBSTRING "${_tinystories_clamp_body}" 0 ${_tinystories_clamp_match}
        _tinystories_clamp_head)
    string(SUBSTRING "${_tinystories_clamp_body}" ${_tinystories_clamp_match} -1
        _tinystories_clamp_tail)
    string(LENGTH "${_tinystories_clamp_signature}" _tinystories_clamp_signature_length)
    string(SUBSTRING "${_tinystories_clamp_tail}"
        ${_tinystories_clamp_signature_length} -1 _tinystories_clamp_tail)
    # Keep this outside SRC_GEN_DIR, which is also globbed into the application;
    # the replacement must be compiled only once, by portable_kernels at -O2.
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tinystories-kernels")
    set(_tinystories_clamp_generated
        "${CMAKE_BINARY_DIR}/tinystories-kernels/TinyStoriesCacheClamp.cpp")
    file(WRITE "${_tinystories_clamp_generated}"
        "#include \"${CMAKE_CURRENT_LIST_DIR}/src/CacheClamp.hpp\"\n"
        "${_tinystories_clamp_prefix}${_tinystories_clamp_head}    Tensor& out) {\n"
        "${_tinystories_clamp_hook}\n  bool has_min${_tinystories_clamp_tail}")
    get_target_property(_tinystories_kernel_sources portable_kernels SOURCES)
    list(REMOVE_ITEM _tinystories_kernel_sources "${_tinystories_clamp_source}")
    list(APPEND _tinystories_kernel_sources "${_tinystories_clamp_generated}")
    set_property(TARGET portable_kernels PROPERTY SOURCES "${_tinystories_kernel_sources}")
    set_property(SOURCE "${_tinystories_clamp_generated}" TARGET_DIRECTORY portable_kernels
        APPEND PROPERTY COMPILE_OPTIONS -O2)
endif()
list(APPEND ${use_case}_LINK_LIBS ml_framework_et)

# The shared platform script reserves 32 MiB for the DDR image. This model,
# method pool and ExecuTorch temporary pool need a 64 MiB use-case-only region.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_tinystories_linker_suffix llvm)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_tinystories_linker_suffix gnu)
else()
    message(FATAL_ERROR "tinystories requires the GNU or LLVM MPS4 linker script")
endif()
string(CONCAT _tinystories_default_linker
    "${MLEK_ROOT}/scripts/cmake/platforms/mps4/sse-320/"
    "mps4-sse-320.${_tinystories_linker_suffix}.ld")
file(READ "${_tinystories_default_linker}" _tinystories_linker_script)
set(_tinystories_linker_original "${_tinystories_linker_script}")
string(REPLACE
    "DDR   (rwx) : ORIGIN = 0x70000000, LENGTH = 0x02000000"
    "DDR   (rwx) : ORIGIN = 0x70000000, LENGTH = 0x04000000"
    _tinystories_linker_script "${_tinystories_linker_script}")
if ("${_tinystories_linker_script}" STREQUAL "${_tinystories_linker_original}")
    message(FATAL_ERROR "Could not expand the MPS4 DDR region for TinyStories")
endif()
# Keep the PTE and method addresses stable. Place the read-only vocabulary after
# the method pool in the same DDR image, before the fixed temporary pool.
string(REPLACE "*(activation_buf_dram)"
    "*(activation_buf_dram)\n        . = ALIGN(16);\n        *(tinystories_tokenizer)\n        ASSERT(. <= ${ML_FWK_TMP_MEM_BASE}, \"TinyStories tokenizer overlaps temporary pool\");"
    _tinystories_linker_script "${_tinystories_linker_script}")
set(_tinystories_custom_linker
    "${CMAKE_BINARY_DIR}/mps4-sse-320-tinystories.${_tinystories_linker_suffix}.ld")
file(WRITE "${_tinystories_custom_linker}" "${_tinystories_linker_script}")
set(${use_case}_LINKER_SCRIPT_OVERRIDE_PATH "${_tinystories_custom_linker}")
