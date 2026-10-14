/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "RTE_Components.h"

#include <stdbool.h>
#include <stdio.h>

struct test_systick g_test_systick;
struct test_scb g_test_scb;
struct test_fpgaio g_test_fpgaio;
struct test_scc g_test_scc;
uint32_t SystemCoreClock = 25000000U;
static uint32_t s_primask;

/* Exercise the real timer implementation without an FPGA or framework. */
#include "../../../source/hal/source/platform/mps4/source/timer_mps4.c"

#define CHECK(condition)                                                    \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #condition); \
            return false;                                                   \
        }                                                                   \
    } while (0)

uint32_t __get_PRIMASK(void)
{ return s_primask; }

void __disable_irq(void)
{ s_primask = 1U; }

void __set_PRIMASK(uint32_t value)
{ s_primask = value; }

void __NOP(void) {}

void NVIC_DisableIRQ(int irq)
{ (void)irq; }

void NVIC_EnableIRQ(int irq)
{ (void)irq; }

uint32_t SysTick_Config(uint32_t ticks)
{
    if (ticks == 0U || ticks - 1U > SysTick_VAL_CURRENT_Msk) {
        return 1U;
    }
    SysTick->LOAD = ticks - 1U;
    SysTick->VAL  = ticks - 1U;
    return 0U;
}

static bool test_clock_sources(void)
{
    MPS4_PLATFORM_SCC->CFG_ACLK = 0U;
#if defined(FPGA)
    CHECK(get_mps4_core_clock() == 50000000U);
#else
    CHECK(get_mps4_core_clock() == 25000000U);
    SystemCoreClock = 27000000U;
    CHECK(get_mps4_core_clock() == 27000000U);
    SystemCoreClock = 25000000U;
#endif
    MPS4_PLATFORM_SCC->CFG_ACLK = 12345678U;
    CHECK(get_mps4_core_clock() == 12345678U);
    return true;
}

static bool test_consistent_duration(void)
{
    MPS4_PLATFORM_SCC->CFG_ACLK = 25000000U;
    CHECK(get_tstamp_milliseconds(999U * 25000U) == 999U);
    CHECK(get_tstamp_milliseconds(1000U * 25000U) == 1000U);
    CHECK(get_tstamp_milliseconds(1001U * 25000U) == 1001U);
    CHECK(get_tstamp_milliseconds(4000U * 25000U) == 4000U);
    CHECK(get_tstamp_milliseconds(39902782U) == 1596U);
    CHECK(get_tstamp_milliseconds(UINT64_C(5000000000) * 25000U) == UINT64_C(5000000000));

    /* Exact frequency conversion also works when Hz is not divisible by 1000. */
    MPS4_PLATFORM_SCC->CFG_ACLK = 12345678U;
    CHECK(get_tstamp_milliseconds(12345678U) == 1000U);
    CHECK(get_tstamp_milliseconds(12345677U) == 999U);
    return true;
}

static bool test_systick_snapshot(void)
{
    MPS4_PLATFORM_SCC->CFG_ACLK = 25000000U;
    platform_reset_counters();
    CHECK(SysTick->LOAD == 249999U);
    SysTick_Handler();
    CHECK(cpu_cycle_count == 250000U);

    cpu_cycle_count = UINT64_C(4295000000);
    SysTick->VAL    = SysTick->LOAD - 1234U;
    SCB->ICSR       = 0U;
    s_primask       = 1U;
    CHECK(Get_SysTick_Cycle_Count() == UINT64_C(4295001234));
    CHECK(s_primask == 1U);
    SCB->ICSR = SCB_ICSR_PENDSTSET_Msk;
    s_primask = 0U;
    CHECK(Get_SysTick_Cycle_Count() == UINT64_C(4295251234));
    CHECK(s_primask == 0U);
    SCB->ICSR = 0U;
    return true;
}

static bool test_profile_counters(void)
{
    MPS4_PLATFORM_SCC->CFG_ACLK = 25000000U;
    SysTick->VAL                = SysTick->LOAD;
    SCB->ICSR                   = 0U;
    cpu_cycle_count             = 100000000U;
    /* A different board timer rate must not change the millisecond source. */
    MPS4_PLATFORM_FPGAIO->CLK100HZ = 1000U;
    pmu_counters counters;
    platform_get_counters(&counters);
    CHECK(counters.initialised && counters.num_counters == 2U);
    CHECK(counters.counters[0].value == 100000000U);
    CHECK(counters.counters[1].value == 4000U);
    return true;
}

int main(void)
{
    return test_clock_sources() && test_consistent_duration() && test_systick_snapshot() &&
                   test_profile_counters()
               ? 0
               : 1;
}
