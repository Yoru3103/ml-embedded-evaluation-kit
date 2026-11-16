/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "hal_uart.h"
#include "uart_config.h"
#include "uart_stdout.h"

void hal_uart_write_raw(const uint8_t* data, size_t size)
{
    /* Same DATA (0x000), STATE (0x004), TX-full bit as core-platform's
     * uart_cmsdk_apb.c UartPutc. Omit its LF-to-CRLF text conversion.
     * CTRL and BAUDDIV remain owned by the existing UART driver. */
    volatile uint32_t* uart = (volatile uint32_t*)UART0_BASE;
    for (size_t i = 0; i < size; ++i) {
        while (uart[1] & 1U) {}
        uart[0] = data[i];
    }
}

uint8_t hal_uart_read_byte(void)
{ return UartGetc(); }
