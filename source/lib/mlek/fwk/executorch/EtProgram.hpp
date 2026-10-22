/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlek/fwk/executorch/EtMemoryAllocator.hpp"
#include <memory>
#include <string>
#include <vector>

namespace arm::app::fwk::et {

/** A program with independently owned, persistent methods and caller-owned pools.
 * Outputs remain valid until the same method executes again or this object dies.
 * Execute is synchronous; methods share the temporary pool and must not run concurrently.
 */
class EtProgram {
public:
    EtProgram(const uint8_t* data,
              size_t size,
              uint8_t* methodPool,
              size_t methodSize,
              uint8_t* tempPool,
              size_t tempSize);
    ~EtProgram();
    EtProgram(const EtProgram&)            = delete;
    EtProgram& operator=(const EtProgram&) = delete;

    /** Load the program and cache a named method, including planned memory. */
    bool LoadMethod(const char* name);
    /** Get metadata after loading a method. */
    executorch::runtime::Result<executorch::runtime::MethodMeta> Metadata(const char* name);
    /** Set inputs and run a previously loaded method. */
    executorch::runtime::Result<executorch::runtime::EValue>
    Execute(const char* name, const std::vector<executorch::runtime::EValue>& inputs);
    /** Report pool placement and allocation high water marks. */
    void LogMemoryUsage() const;

private:
    struct MethodState;
    std::unique_ptr<executorch::extension::BufferDataLoader> m_loader;
    std::unique_ptr<executorch::runtime::Program> m_program;
    EtMemoryAllocator m_methodAllocator;
    EtMemoryAllocator m_tempAllocator;
    std::vector<std::unique_ptr<MethodState>> m_methods;
};
} // namespace arm::app::fwk::et
