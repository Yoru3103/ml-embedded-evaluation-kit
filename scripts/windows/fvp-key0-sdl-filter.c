/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

typedef unsigned int uint32_t;
typedef union {
    uint32_t type;
    struct {
        uint32_t type, timestamp, windowId;
    } window;
    unsigned char data[56];
    unsigned long long alignment;
} SdlEvent;

__declspec(dllimport) void* SDL_CreateWindow(const char*, int, int, int, int, uint32_t);
__declspec(dllimport) void* SDL_GetWindowFromID(uint32_t);
__declspec(dllimport) const char* SDL_GetWindowTitle(void*);
__declspec(dllimport) void SDL_SetEventFilter(int (*)(void*, SdlEvent*), void*);

static int IsLcdWindow(const char* title)
{
    const char* suffix = "HDLCD";
    if (!title) {
        return 0;
    }
    unsigned length = 0;
    while (title[length]) {
        ++length;
    }
    if (length < 5) {
        return 0;
    }
    title += length - 5;
    while (*title && *title == *suffix) {
        ++title;
        ++suffix;
    }
    return *title == *suffix;
}

static int KeepEvent(void* userdata, SdlEvent* event)
{
    (void)userdata;
    switch (event->type) {
    case 0x302: /* SDL_TEXTEDITING */
    case 0x305: /* SDL_TEXTEDITING_EXT */
        /* The FVP board has no text editor. A focus-loss composition can block its queue. */
        return 0;
    case 0x300: /* SDL_KEYDOWN */
    case 0x301: /* SDL_KEYUP */
    case 0x303: /* SDL_TEXTINPUT */
    case 0x400: /* SDL_MOUSEMOTION */
    case 0x401: /* SDL_MOUSEBUTTONDOWN */
    case 0x402: /* SDL_MOUSEBUTTONUP */
    case 0x403: /* SDL_MOUSEWHEEL */
        if (event->window.windowId) {
            void* window = SDL_GetWindowFromID(event->window.windowId);
            return window && !IsLcdWindow(SDL_GetWindowTitle(window));
        }
        return 1;
    default:
        /* Keep window management and rendering notifications for both windows. */
        return 1;
    }
}

void* CreateFvpWindow(const char* title, int x, int y, int width, int height, uint32_t flags)
{
    static int installed;
    if (!installed) {
        SDL_SetEventFilter(KeepEvent, 0);
        installed = 1;
    }
    return SDL_CreateWindow(title, x, y, width, height, flags);
}
