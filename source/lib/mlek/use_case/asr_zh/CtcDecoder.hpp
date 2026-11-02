/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_ASR_ZH_CTC_DECODER_HPP
#define MLEK_ASR_ZH_CTC_DECODER_HPP
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace arm::app::asr_zh {
/** Borrowed byte-BPE lookup entry; IDs include blank/control/unknown. */
struct TokenBytes {
    const unsigned char* data;
    size_t size;
};

/** Greedy CTC with cross-chunk repeats and sentence-end UTF-8 recovery. */
class CtcDecoder {
public:
    CtcDecoder(const TokenBytes* tokens, size_t count) : m_tokens(tokens), m_count(count) {}
    void Reset();
    bool Append(const float* scores, size_t frames, size_t classes);
    bool AppendToken(size_t token);
    /** Recover maximal valid UTF-8 characters; compact Chinese display removes ASCII spaces. */
    std::string Text() const;
    const std::vector<uint32_t>& Path() const
    { return m_path; }

private:
    const TokenBytes* m_tokens;
    size_t m_count;
    int32_t m_previous{-1};
    std::string m_bytes;
    std::vector<uint32_t> m_path;
};
} // namespace arm::app::asr_zh
#endif
