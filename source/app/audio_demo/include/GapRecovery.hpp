/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_AUDIO_GAP_RECOVERY_HPP
#define MLEK_AUDIO_GAP_RECOVERY_HPP

#include <cstddef>
#include <cstdint>

namespace arm::app::fvp_audio {
/** Group repeated overrun gaps into one recovery episode, without console I/O. */
class GapRecovery {
public:
    explicit GapRecovery(size_t stableSamples) : m_required{stableSamples} {}

    /** True only when a new episode starts. Every gap still resets model context. */
    bool Observe(bool gap, size_t samples)
    {
        const bool entered = gap && !m_active;
        if (gap) {
            m_active     = true;
            m_contiguous = 0;
        }
        if (m_active) {
            m_contiguous += samples;
            if (m_contiguous >= m_required) {
                m_active     = false;
                m_contiguous = 0;
            }
        }
        return entered;
    }

    bool Active() const
    { return m_active; }

private:
    size_t m_required;
    size_t m_contiguous{0};
    bool m_active{false};
};
} // namespace arm::app::fvp_audio
#endif
