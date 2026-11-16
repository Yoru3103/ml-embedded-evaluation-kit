/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "MiniMindTokenDecoder.hpp"

#include <cstring>

namespace arm::app::minimind2_small {
const uint8_t* GetTokenizerPointer();
size_t GetTokenizerLen();

namespace {
constexpr uint32_t kTableMagic = 0x4D4D544B; // "MMTK"
constexpr size_t kHeaderBytes = 8;

uint32_t ReadU32(const uint8_t* base, size_t index)
{
    uint32_t value = 0;
    std::memcpy(&value, base + index * sizeof(uint32_t), sizeof(value));
    return value;
}
} // namespace

bool DecodeTokenIds(const int32_t* ids,
                    size_t count,
                    char* output,
                    size_t capacity,
                    size_t& written)
{
    written = 0;
    if (ids == nullptr || output == nullptr || capacity == 0) {
        return false;
    }

    const uint8_t* table = GetTokenizerPointer();
    const size_t tableBytes = GetTokenizerLen();
    if (table == nullptr || tableBytes < kHeaderBytes) {
        return false;
    }
    const uint32_t magic = ReadU32(table, 0);
    const uint32_t vocabulary = ReadU32(table, 1);
    const size_t offsetsBytes = static_cast<size_t>(vocabulary + 1) * sizeof(uint32_t);
    if (magic != kTableMagic || tableBytes < kHeaderBytes + offsetsBytes) {
        return false;
    }
    const uint8_t* offsets = table + kHeaderBytes;
    const uint8_t* data = offsets + offsetsBytes;
    const size_t dataBytes = tableBytes - kHeaderBytes - offsetsBytes;

    for (size_t index = 0; index < count; ++index) {
        const int32_t id = ids[index];
        if (id < 0 || static_cast<uint32_t>(id) >= vocabulary) {
            return false;
        }
        const uint32_t begin = ReadU32(offsets, static_cast<size_t>(id));
        const uint32_t end = ReadU32(offsets, static_cast<size_t>(id) + 1);
        if (end < begin || end > dataBytes) {
            return false;
        }
        const size_t length = static_cast<size_t>(end - begin);
        if (written + length + 1 > capacity) {
            return false;
        }
        std::memcpy(output + written, data + begin, length);
        written += length;
    }
    output[written] = '\0';
    return true;
}
} // namespace arm::app::minimind2_small
