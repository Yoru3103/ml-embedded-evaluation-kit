/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MPS4_DISPLAY_TEST_RTE_COMPONENTS_H
#define MPS4_DISPLAY_TEST_RTE_COMPONENTS_H

#include <stdint.h>

#define __DCACHE_PRESENT 1U
#define SCB_CCR_DC_Msk   1U

struct test_scb {
    uint32_t CCR;
};

extern struct test_scb g_test_scb;
#define SCB (&g_test_scb)

void SCB_CleanDCache_by_Addr(volatile void* addr, int32_t size);
void __DSB(void);

#endif /* MPS4_DISPLAY_TEST_RTE_COMPONENTS_H */
