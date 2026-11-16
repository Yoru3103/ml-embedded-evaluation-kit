/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>
inline unsigned g_inferences = 0;
inline std::vector<std::vector<int16_t>> g_captured;
inline std::vector<std::string> g_lcd;
inline unsigned g_displayWrites = 0;
constexpr int COLOR_BLACK = 0, COLOR_WHITE = 1, COLOR_GREEN = 2;
inline void hal_display_clear(int)
{ g_lcd.clear(); }
inline void hal_display_show_box(int, int y, int, int, int)
{
    if (y >= 60) {
        // Only a new successful transcript may clear the result region.
        g_lcd.clear();
    }
}
inline void hal_display_set_text_color(int) {}
inline void hal_display_show_text(const char* s, size_t n, int, int, bool)
{
    ++g_displayWrites;
    g_lcd.emplace_back(s, n);
}
namespace arm::app {
namespace fwk::iface {
    struct Tensor {
        std::vector<size_t> shape;
        std::vector<size_t> Shape() const
        { return shape; }
        void* GetData()
        { return nullptr; }
    };
    struct Model {
        Tensor input{{1, 700, 80}}, output{{1, 174, 129}};
        bool IsInited() const
        { return true; }
        Tensor* GetInputTensor(int)
        { return &input; }
        Tensor* GetOutputTensor(int)
        { return &output; }
        bool RunInference()
        {
            ++g_inferences;
            return true;
        }
    };
} // namespace fwk::iface
struct ApplicationContext {
    fwk::iface::Model model;
    std::vector<std::string> labels = std::vector<std::string>(128, "token");
    template <class T>
    T Get(const char* name)
    {
        if constexpr (std::is_same_v<T, fwk::iface::Model&>)
            return model;
        else if constexpr (std::is_same_v<T, const std::vector<std::string>&>)
            return labels;
        else if constexpr (std::is_same_v<T, uint32_t>) {
            return std::string(name) == "melSpecWindowSize" ? 512
                   : std::string(name) == "melSpecHopSize"  ? 160
                                                            : 700;
        }
    }
};
template <class T>
struct ConformerPreProcess {
    ConformerPreProcess(fwk::iface::Tensor*, fwk::iface::Tensor*, int, int, int) {}
    bool DoPreProcess(const int16_t* audio, size_t size)
    {
        assert(size >= 512);
        g_captured.emplace_back(audio, audio + size);
        return true;
    }
};
struct ConformerPostProcess {
    std::string& text;
    ConformerPostProcess(fwk::iface::Tensor*,
                         fwk::iface::Tensor*,
                         const std::vector<std::string>&,
                         std::string& out) : text(out)
    {}
    bool DoPostProcess()
    {
        text = "decoded";
        return true;
    }
};
} // namespace arm::app
