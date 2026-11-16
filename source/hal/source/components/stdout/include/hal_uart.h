/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/** Write unmodified bytes through the already initialized CMSDK UART. */
void hal_uart_write_raw(const uint8_t* data, size_t size);
/** Read one command byte, blocking until input is available. */
uint8_t hal_uart_read_byte(void);
#ifdef __cplusplus
}
#endif
