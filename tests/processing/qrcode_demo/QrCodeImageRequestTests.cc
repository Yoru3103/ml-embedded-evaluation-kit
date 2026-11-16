/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "QrCodeImageRequest.hpp"

#include <cassert>
#include <vector>

using namespace arm::app::qrcode_demo;

static void AppendWord(std::vector<uint8_t>& bytes, uint32_t word)
{
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(word >> shift));
    }
}

static ReadStatus Read(const std::vector<uint8_t>& packet, bool previous = false)
{
    FILE* file = std::tmpfile();
    assert(file);
    assert(std::fwrite(packet.data(), 1, packet.size(), file) == packet.size());
    std::rewind(file);
    uint8_t image[16]{};
    uint32_t request  = 0;
    const auto status = ReadRequest(file, 4, 4, image, sizeof(image), previous, 42, request);
    assert(std::fclose(file) == 0);
    if (status == ReadStatus::READY) {
        assert(request == 42 || request == 43);
        assert(image[15] == 15);
    }
    return status;
}

int main()
{
    const uint8_t reference[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    assert(Crc32(reference, sizeof(reference)) == 0xCBF43926U);
    std::vector<uint8_t> image;
    for (uint8_t value = 0; value < 16; ++value) {
        image.push_back(value);
    }
    std::vector<uint8_t> packet{'Q', 'R', 'D', '1'};
    for (const uint32_t word : {42U, 4U, 4U, 16U, Crc32(image.data(), image.size())}) {
        AppendWord(packet, word);
    }
    packet.insert(packet.end(), image.begin(), image.end());
    assert(Read(packet) == ReadStatus::READY);
    assert(Read(packet, true) == ReadStatus::UNCHANGED);
    auto changed = packet;
    changed[4]   = 43;
    assert(Read(changed, true) == ReadStatus::READY);
    for (const size_t offset : {0U, 8U, 12U, 16U, 20U, 25U}) {
        auto corrupt = packet;
        corrupt[offset] ^= 1;
        assert(Read(corrupt) == ReadStatus::ERROR);
    }
    packet.pop_back();
    assert(Read(packet) == ReadStatus::ERROR);
    packet.push_back(15);
    packet.push_back(0);
    assert(Read(packet) == ReadStatus::ERROR);
    assert(Read(std::vector<uint8_t>{'Q'}) == ReadStatus::ERROR);

    // RGB uses the same validation with a distinct magic and exact channel count.
    std::vector<uint8_t> rgb(48);
    for (size_t index = 0; index < rgb.size(); ++index) {
        rgb[index] = static_cast<uint8_t>(index);
    }
    std::vector<uint8_t> hand{'H', 'P', 'D', '1'};
    for (const uint32_t word : {43U, 4U, 4U, 48U, Crc32(rgb.data(), rgb.size())}) {
        AppendWord(hand, word);
    }
    hand.insert(hand.end(), rgb.begin(), rgb.end());
    FILE* file = std::tmpfile();
    assert(file);
    assert(std::fwrite(hand.data(), 1, hand.size(), file) == hand.size());
    std::rewind(file);
    uint8_t received[48]{};
    uint32_t request = 0;
    assert(arm::app::image_demo::ReadRequest(
               file, 4, 4, {"HPD1", 3}, received, sizeof(received), false, 0, request) ==
           ReadStatus::READY);
    assert(request == 43 && std::memcmp(received, rgb.data(), rgb.size()) == 0);
    std::rewind(file);
    assert(arm::app::image_demo::ReadRequest(
               file, 4, 4, {"HPD1", 1}, received, sizeof(received), false, 0, request) ==
           ReadStatus::ERROR);
    std::rewind(file);
    assert(arm::app::image_demo::ReadRequest(
               file, 4, 4, {"QRD1", 3}, received, sizeof(received), false, 0, request) ==
           ReadStatus::ERROR);
    assert(std::fclose(file) == 0);
    std::puts("QR image request validation: PASS");
}
