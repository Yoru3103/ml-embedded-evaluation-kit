/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RTE_Components.h"

bool test_detection_drawing(void);

#define GUARD_BYTES 16
#define FRAME_BYTES (HDLCD_RES_WIDTH * HDLCD_RES_HEIGHT * HDLCD_BYTES_PER_PIXEL)

static uint8_t s_frame_buffer[GUARD_BYTES + FRAME_BYTES + GUARD_BYTES];
static uint8_t s_lcd_pixels[FRAME_BYTES];
struct test_scb g_test_scb = {.CCR = SCB_CCR_DC_Msk};

#define HDLCD_FRAME_BUFFER_BASE_ADDRESS ((uintptr_t)(s_frame_buffer + GUARD_BYTES))

/* Exercise the production HAL with RAM backing its fixed framebuffer address. */
#include "../../../source/hal/source/components/display/source/hdlcd_mps4/hal_display_mps4.c"

/* Model the HDLCD reading DDR only after the CPU cleans its cached writes. */
void SCB_CleanDCache_by_Addr(volatile void* addr, int32_t size)
{
    const uintptr_t start = (uintptr_t)addr;
    const uintptr_t base  = HDLCD_FRAME_BUFFER_BASE_ADDRESS;
    if (start < base || size <= 0 || start - base + (size_t)size > FRAME_BYTES) {
        abort();
    }
    memcpy(s_lcd_pixels + (start - base), (const void*)addr, (size_t)size);
}

void __DSB(void)
{
    if (!(g_test_scb.CCR & SCB_CCR_DC_Msk)) {
        memcpy(s_lcd_pixels, s_frame_buffer + GUARD_BYTES, FRAME_BYTES);
    }
}

#define CHECK(condition)                                                    \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #condition); \
            return false;                                                   \
        }                                                                   \
    } while (0)

void reset_frame_buffer(void)
{
    memset(s_frame_buffer, 0x5a, sizeof(s_frame_buffer));
    memset(s_lcd_pixels, 0x5a, sizeof(s_lcd_pixels));
}

/** Compare every framebuffer byte and both guard regions with a literal pixel map. */
bool check_pixels(const char* expected)
{
    CHECK(strlen(expected) == HDLCD_RES_WIDTH * HDLCD_RES_HEIGHT);
    for (size_t index = 0; index < GUARD_BYTES; ++index) {
        CHECK(s_frame_buffer[index] == 0x5a);
        CHECK(s_frame_buffer[GUARD_BYTES + FRAME_BYTES + index] == 0x5a);
    }

    for (size_t pixel = 0; pixel < HDLCD_RES_WIDTH * HDLCD_RES_HEIGHT; ++pixel) {
        const uint8_t* actual = s_frame_buffer + GUARD_BYTES + pixel * HDLCD_BYTES_PER_PIXEL;
        CHECK(memcmp(actual, s_lcd_pixels + pixel * HDLCD_BYTES_PER_PIXEL, HDLCD_BYTES_PER_PIXEL) ==
              0);
        if (expected[pixel] == '.') {
            for (size_t channel = 0; channel < HDLCD_BYTES_PER_PIXEL; ++channel) {
                CHECK(actual[channel] == 0x5a);
            }
        } else {
            CHECK(actual[0] == (expected[pixel] == 'r' ? 248 : 0));
            CHECK(actual[1] == (expected[pixel] == 'g' ? 252 : 0));
            CHECK(actual[2] == (expected[pixel] == 'b' ? 248 : 0));
#if HDLCD_BYTES_PER_PIXEL == 4
            CHECK(actual[3] == 0);
#endif
        }
    }
    return true;
}

static bool test_filled_rectangle(void)
{
    reset_frame_buffer();
    CHECK(hal_display_show_box(2, 1, 3, 2, 0x07e0) == 0);
    return check_pixels("........"
                        "..ggg..."
                        "..ggg..."
                        "........"
                        "........"
                        "........");
}

/** The detection handler uses four filled strips to leave the image interior visible. */
static bool test_detection_outline(void)
{
    reset_frame_buffer();
    CHECK(hal_display_show_box(2, 1, 4, 1, 0x07e0) == 0);
    CHECK(hal_display_show_box(2, 4, 4, 1, 0x07e0) == 0);
    CHECK(hal_display_show_box(2, 1, 1, 4, 0x07e0) == 0);
    CHECK(hal_display_show_box(5, 1, 1, 4, 0x07e0) == 0);
    return check_pixels("........"
                        "..gggg.."
                        "..g..g.."
                        "..g..g.."
                        "..gggg.."
                        "........");
}

static bool test_clipped_rectangle(void)
{
    reset_frame_buffer();
    CHECK(hal_display_show_box(6, 4, UINT32_MAX, UINT32_MAX, 0x07e0) == 0);
    return check_pixels("........"
                        "........"
                        "........"
                        "........"
                        "......gg"
                        "......gg");
}

static bool test_offscreen_rectangle(void)
{
    reset_frame_buffer();
    CHECK(hal_display_show_box(8, 0, 1, 1, 0x07e0) != 0);
    CHECK(hal_display_show_box(0, 6, 1, 1, 0x07e0) != 0);
    CHECK(hal_display_show_box(UINT32_MAX, UINT32_MAX, 1, 1, 0x07e0) != 0);
    return check_pixels("................................................");
}

static bool test_empty_rectangle(void)
{
    reset_frame_buffer();
    CHECK(hal_display_show_box(2, 1, 0, 3, 0x07e0) == 0);
    CHECK(hal_display_show_box(2, 1, 3, 0, 0x07e0) == 0);
    return check_pixels("................................................");
}

static bool test_rgb565_colors(void)
{
    reset_frame_buffer();
    CHECK(hal_display_show_box(0, 0, 1, 1, 0xf800) == 0);
    CHECK(hal_display_show_box(1, 0, 1, 1, 0x07e0) == 0);
    CHECK(hal_display_show_box(2, 0, 1, 1, 0x001f) == 0);
    CHECK(hal_display_show_box(7, 5, 1, 1, 0) == 0);
    return check_pixels("rgb....."
                        "........"
                        "........"
                        "........"
                        "........"
                        ".......k");
}

static bool test_fullscreen_rectangle(void)
{
    reset_frame_buffer();
    CHECK(hal_display_show_box(0, 0, 8, 6, 0x07e0) == 0);
    return check_pixels("gggggggggggggggggggggggggggggggggggggggggggggggg");
}

static bool test_configured_display_bounds(void)
{
    reset_frame_buffer();
    const hal_display_clr_format format = HDLCD_BYTES_PER_PIXEL == 3
                                              ? HAL_DISPLAY_COLOUR_FORMAT_RGB888
                                              : HAL_DISPLAY_COLOUR_FORMAT_ARGB8888;
    CHECK(hal_display_configure(4, 3, format) == 0);
    CHECK(hal_display_show_box(2, 1, UINT32_MAX, UINT32_MAX, 0x07e0) == 0);
    CHECK(hal_display_show_box(4, 1, 1, 1, 0x07e0) != 0);
    CHECK(hal_display_init() == 0);
    CHECK(check_pixels("........"
                       "..gg...."
                       "..gg...."
                       "........"
                       "........"
                       "........"));

    reset_frame_buffer();
    CHECK(hal_display_configure(16, 12, format) == 0);
    CHECK(hal_display_show_box(7, 5, UINT32_MAX, UINT32_MAX, 0x07e0) == 0);
    CHECK(hal_display_show_box(8, 5, 1, 1, 0x07e0) != 0);
    CHECK(hal_display_init() == 0);
    return check_pixels("........"
                        "........"
                        "........"
                        "........"
                        "........"
                        ".......g");
}

static bool test_uncached_frame_buffer(void)
{
    reset_frame_buffer();
    g_test_scb.CCR = 0;
    CHECK(hal_display_show_box(3, 2, 2, 2, 0xf800) == 0);
    g_test_scb.CCR = SCB_CCR_DC_Msk;
    return check_pixels("........"
                        "........"
                        "...rr..."
                        "...rr..."
                        "........"
                        "........");
}

static bool test_image_cache_publication(void)
{
    uint8_t image[8 * 6 * 3];
    memset(image, 0x33, sizeof(image));
    reset_frame_buffer();
    CHECK(hal_display_show_image(image, 8, 6, 3, 0, 0, 1) == 0);
    CHECK(memcmp(s_frame_buffer + GUARD_BYTES, s_lcd_pixels, FRAME_BYTES) == 0);
    for (size_t index = 0; index < GUARD_BYTES; ++index) {
        CHECK(s_frame_buffer[index] == 0x5a);
        CHECK(s_frame_buffer[GUARD_BYTES + FRAME_BYTES + index] == 0x5a);
    }
    return true;
}

int main(void)
{
    if (hal_display_init() != 0) {
        return 1;
    }

    bool (*const tests[])(void) = {
        test_filled_rectangle,
        test_detection_outline,
        test_clipped_rectangle,
        test_offscreen_rectangle,
        test_empty_rectangle,
        test_rgb565_colors,
        test_fullscreen_rectangle,
        test_configured_display_bounds,
        test_uncached_frame_buffer,
        test_image_cache_publication,
        test_detection_drawing,
    };

    for (size_t index = 0; index < sizeof(tests) / sizeof(tests[0]); ++index) {
        if (!tests[index]()) {
            return 1;
        }
    }
    printf("MPS4 %d-byte framebuffer: all pixel tests passed\n", HDLCD_BYTES_PER_PIXEL);
    return 0;
}
