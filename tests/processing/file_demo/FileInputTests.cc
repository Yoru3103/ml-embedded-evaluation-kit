/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "UartFileInput.hpp"
#include <cassert>
#include <vector>

namespace {
std::vector<uint8_t> s_wire;
size_t s_offset = 0;

uint8_t ReadByte()
{
    assert(s_offset < s_wire.size());
    return s_wire[s_offset++];
}

void Word(uint32_t value)
{
    for (unsigned bit = 0; bit < 32; bit += 8) {
        s_wire.push_back(static_cast<uint8_t>(value >> bit));
    }
}

void Packet(const char* magic,
            uint32_t id,
            uint32_t width,
            uint32_t height,
            const std::vector<uint8_t>& bytes,
            bool corrupt = false)
{
    s_wire.insert(s_wire.end(), magic, magic + 4);
    Word(id);
    Word(width);
    Word(height);
    Word(bytes.size());
    Word(arm::app::image_demo::Crc32(bytes.data(), bytes.size()) ^ (corrupt ? 1U : 0U));
    s_wire.insert(s_wire.end(), bytes.begin(), bytes.end());
}
} // namespace

int main()
{
    using arm::app::file_demo::Receive;
    std::vector<uint8_t> image(32 * 32 * 3);
    for (size_t i = 0; i < image.size(); ++i) {
        image[i] = static_cast<uint8_t>(i); // Includes NUL, CR, LF, and 0xff.
    }
    s_wire.push_back('?');
    Packet("HPD1", 1, 32, 32, image, true);
    Packet("HPD1", 2, 32, 32, image);
    std::vector<uint8_t> received(image.size());
    assert(Receive("HPD1", 32, 32, received.data(), received.size(), false, nullptr, ReadByte) ==
           2);
    assert(received == image && s_offset == s_wire.size());

    std::vector<uint8_t> audio(1600, 13);
    Packet("KWD1", 3, 16000, 800, audio);
    received.resize(960000);
    size_t bytes = 0;
    assert(Receive("KWD1", 16000, 0, received.data(), received.size(), true, &bytes, ReadByte) ==
           3);
    assert(bytes == audio.size());
    assert(std::equal(audio.begin(), audio.end(), received.begin()));
    assert(s_offset == s_wire.size());
}
