/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/common/ProcessingStageScope.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::string> s_events;
#if defined(ASR_STAGE_PROFILING)
void Observe(const char* name, bool begin)
{ s_events.emplace_back(std::string(begin ? "begin:" : "end:") + name); }
#endif
} // namespace

int main()
{
#if defined(ASR_STAGE_PROFILING)
    arm::app::ProcessingStageScope::s_observer = Observe;
#endif
    {
        MLEK_PROCESSING_STAGE("outer");
        {
            MLEK_PROCESSING_STAGE("inner");
        }
    }
#if defined(ASR_STAGE_PROFILING)
    const std::vector<std::string> expected = {
        "begin:outer", "begin:inner", "end:inner", "end:outer"};
    if (s_events != expected) {
        throw std::runtime_error("Nested processing scopes must close in reverse order");
    }
    arm::app::ProcessingStageScope::s_observer = nullptr;
    {
        MLEK_PROCESSING_STAGE("unobserved");
    }
    if (s_events != expected) {
        throw std::runtime_error("No callback is allowed after observer removal");
    }
#else
    if (!s_events.empty()) {
        throw std::runtime_error("Disabled profiling must not record events");
    }
#endif
}
