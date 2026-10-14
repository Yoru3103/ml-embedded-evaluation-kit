/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MPS4_TIMER_TEST_RTE_COMPONENTS_H
#define MPS4_TIMER_TEST_RTE_COMPONENTS_H

#include <stdint.h>

/* Back the production timer's CMSIS and board registers with native memory. */
#define SMM_MPS4_H
#define SysTick_VAL_CURRENT_Msk 0x00ffffffU
#define SCB_ICSR_PENDSTSET_Msk  (1U << 26)
#define SysTick_IRQn            (-1)

struct test_systick {
    uint32_t LOAD;
    uint32_t VAL;
};
struct test_scb {
    uint32_t ICSR;
};
struct test_fpgaio {
    uint32_t CLK1HZ;
    uint32_t CLK100HZ;
    uint32_t COUNTER;
};
struct test_scc {
    uint32_t CFG_ACLK;
};

extern struct test_systick g_test_systick;
extern struct test_scb g_test_scb;
extern struct test_fpgaio g_test_fpgaio;
extern struct test_scc g_test_scc;
extern uint32_t SystemCoreClock;

#define SysTick              (&g_test_systick)
#define SCB                  (&g_test_scb)
#define MPS4_PLATFORM_FPGAIO (&g_test_fpgaio)
#define MPS4_PLATFORM_SCC    (&g_test_scc)

uint32_t __get_PRIMASK(void);
void __disable_irq(void);
void __set_PRIMASK(uint32_t value);
void __NOP(void);
void NVIC_DisableIRQ(int irq);
void NVIC_EnableIRQ(int irq);
uint32_t SysTick_Config(uint32_t ticks);

#endif /* MPS4_TIMER_TEST_RTE_COMPONENTS_H */
