/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_PROCESSING_STAGE_SCOPE_HPP
#define MLEK_PROCESSING_STAGE_SCOPE_HPP

#if defined(ASR_STAGE_PROFILING)
namespace arm::app {

/** Optional application-owned observer. Library code has no platform dependencies. */
class ProcessingStageScope {
public:
    using Observer                    = void (*)(const char*, bool);
    inline static Observer s_observer = nullptr;

    explicit ProcessingStageScope(const char* name) : m_name(name)
    {
        if (s_observer) {
            s_observer(m_name, true);
        }
    }

    ~ProcessingStageScope()
    {
        if (s_observer) {
            s_observer(m_name, false);
        }
    }

    ProcessingStageScope(const ProcessingStageScope&)            = delete;
    ProcessingStageScope& operator=(const ProcessingStageScope&) = delete;

private:
    const char* m_name;
};
} // namespace arm::app

#    define MLEK_STAGE_JOIN_IMPL(a, b) a##b
#    define MLEK_STAGE_JOIN(a, b)      MLEK_STAGE_JOIN_IMPL(a, b)
#endif

#if defined(ASR_STAGE_PROFILING)
#    define MLEK_PROCESSING_STAGE(name) \
        ::arm::app::ProcessingStageScope MLEK_STAGE_JOIN(processingStage, __LINE__)(name)
#else
#    define MLEK_PROCESSING_STAGE(name) static_cast<void>(0)
#endif

#endif // MLEK_PROCESSING_STAGE_SCOPE_HPP
