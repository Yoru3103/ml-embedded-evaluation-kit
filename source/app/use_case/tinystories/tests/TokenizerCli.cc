/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#include "TokenDecoder.hpp"
#include "TokenEncoder.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {
std::vector<uint8_t> s_encoder, s_decoder;
std::vector<uint8_t> Read(const char* path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
} // namespace
namespace arm::app::tinystories {
const uint8_t* GetEncoderData()
{ return s_encoder.data(); }
size_t GetEncoderSize()
{ return s_encoder.size(); }
const uint8_t* GetTokenizerData()
{ return s_decoder.data(); }
size_t GetTokenizerSize()
{ return s_decoder.size(); }
} // namespace arm::app::tinystories

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4) {
        return 1;
    }
    const size_t capacity = argc == 4 ? std::stoul(argv[3]) : 1024;
    if (capacity == 0 || capacity > 1024) {
        return 1;
    }
    s_encoder = Read(argv[1]);
    s_decoder = Read(argv[2]);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string text;
        for (size_t index = 0; index + 1 < line.size(); index += 2) {
            text.push_back(static_cast<char>(std::stoi(line.substr(index, 2), nullptr, 16)));
        }
        int32_t tokens[1024];
        size_t count = 0, bytes = 0;
        char decoded[4096];
        if (!arm::app::tinystories::EncodePrompt(text.c_str(), tokens, capacity, count) ||
            !arm::app::tinystories::DecodeTokens(tokens, count, decoded, sizeof(decoded), bytes)) {
            std::cout << "ERR\n";
            continue;
        }
        for (size_t index = 0; index < count; ++index) {
            std::cout << (index ? "," : "") << tokens[index];
        }
        std::cout << '\t';
        const char hex[] = "0123456789abcdef";
        for (size_t index = 0; index < bytes; ++index) {
            const auto byte = static_cast<uint8_t>(decoded[index]);
            std::cout << hex[byte >> 4] << hex[byte & 15];
        }
        std::cout << '\n';
    }
}
