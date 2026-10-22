/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/fwk/executorch/EtProgram.hpp"
#include "mlek/log/log_macros.h"

namespace arm::app::fwk::et {
using namespace executorch::runtime;

struct EtProgram::MethodState {
    std::string name;
    std::vector<Span<uint8_t>> spans;
    std::unique_ptr<HierarchicalAllocator> planned;
    std::unique_ptr<MemoryManager> manager;
    std::unique_ptr<Method> method;
};

EtProgram::EtProgram(const uint8_t* data,
                     size_t size,
                     uint8_t* methodPool,
                     size_t methodSize,
                     uint8_t* tempPool,
                     size_t tempSize) :
    m_loader(std::make_unique<executorch::extension::BufferDataLoader>(data, size)),
    m_methodAllocator(methodSize, methodPool), m_tempAllocator(tempSize, tempPool)
{
    runtime_init();
    auto program = Program::load(m_loader.get());
    if (program.ok()) {
        m_program = std::make_unique<Program>(std::move(program.get()));
        info("PTE loaded: %zu bytes, %zu methods\n", size, m_program->num_methods());
    } else {
        printf_err("PTE load failed: %u\n", static_cast<unsigned>(program.error()));
    }
    info("Method pool: %p, %zu bytes; temp pool: %p, %zu bytes\n",
         methodPool,
         methodSize,
         tempPool,
         tempSize);
}

EtProgram::~EtProgram() = default;

bool EtProgram::LoadMethod(const char* name)
{
    if (!m_program) {
        return false;
    }
    for (const auto& state : m_methods) {
        if (state->name == name) {
            return true;
        }
    }
    auto meta = m_program->method_meta(name);
    if (!meta.ok()) {
        return false;
    }
    auto state          = std::make_unique<MethodState>();
    state->name         = name;
    size_t plannedBytes = 0;
    for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
        auto bytes = meta->memory_planned_buffer_size(i);
        if (!bytes.ok()) {
            return false;
        }
        const size_t bufferSize = static_cast<size_t>(bytes.get());
        auto* buffer            = static_cast<uint8_t*>(m_methodAllocator.allocate(bufferSize, 16));
        if (!buffer) {
            printf_err("%s planned allocation failed: %zu bytes\n", name, bufferSize);
            return false;
        }
        state->spans.emplace_back(buffer, bufferSize);
        plannedBytes += bufferSize;
    }
    state->planned = std::make_unique<HierarchicalAllocator>(
        Span<Span<uint8_t>>(state->spans.data(), state->spans.size()));
    state->manager =
        std::make_unique<MemoryManager>(&m_methodAllocator, state->planned.get(), &m_tempAllocator);
    auto method = m_program->load_method(name, state->manager.get());
    if (!method.ok()) {
        printf_err("%s load failed: %u\n", name, static_cast<unsigned>(method.error()));
        return false;
    }
    state->method = std::make_unique<Method>(std::move(method.get()));
    m_methods.push_back(std::move(state));
    info("Method %s: %zu inputs, %zu outputs, %zu planned bytes\n",
         name,
         meta->num_inputs(),
         meta->num_outputs(),
         plannedBytes);
    return true;
}

Result<MethodMeta> EtProgram::Metadata(const char* name)
{
    if (!m_program) {
        return Error::InvalidState;
    }
    return m_program->method_meta(name);
}

Result<EValue> EtProgram::Execute(const char* name, const std::vector<EValue>& inputs)
{
    for (const auto& state : m_methods) {
        if (state->name != name) {
            continue;
        }
        if (inputs.size() != state->method->inputs_size()) {
            return Error::InvalidArgument;
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto error = state->method->set_input(inputs[i], i);
            if (error != Error::Ok) {
                return error;
            }
        }
        auto error = state->method->execute();
        if (error != Error::Ok) {
            return error;
        }
        return state->method->get_output(0);
    }
    return Error::NotFound;
}

void EtProgram::LogMemoryUsage() const
{
    info("Method pool peak: %zu; temp pool peak: %zu bytes\n",
         m_methodAllocator.UsedSizePeak(),
         m_tempAllocator.UsedSizePeak());
}
} // namespace arm::app::fwk::et
