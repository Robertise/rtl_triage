"""
Visible Testbench for FSM Controller Module
Covers TC-FSM-01 to TC-FSM-07 basic specification requirements.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer

# State codes
S_MAIN_GREEN = 0b00
S_MAIN_YELLOW = 0b01
S_SIDE_GREEN = 0b10
S_SIDE_YELLOW = 0b11

# Light encodings (3-bit one-hot: {RED, YELLOW, GREEN})
LIGHT_GREEN = 0b001
LIGHT_YELLOW = 0b010
LIGHT_RED = 0b100


def assert_signal(name: str, actual, expected):
    """Formats assertion failure with simulation timestamp, signal, expected, and actual."""
    sim_time = cocotb.utils.get_sim_time(unit="ns")
    assert actual == expected, (
        f"[{sim_time:.2f}ns] Signal '{name}' mismatch: expected {expected!r}, got {actual!r}"
    )


def assert_safety_invariant(dut):
    """Asserts that both streets are never simultaneously GREEN or YELLOW."""
    main_l = int(dut.main_signal.value)
    side_l = int(dut.side_signal.value)
    sim_time = cocotb.utils.get_sim_time(unit="ns")

    both_green = (main_l & LIGHT_GREEN) and (side_l & LIGHT_GREEN)
    assert not both_green, (
        f"[{sim_time:.2f}ns] Safety Violation! Both streets are GREEN: main={main_l:03b}, side={side_l:03b}"
    )
    both_yellow = (main_l & LIGHT_YELLOW) and (side_l & LIGHT_YELLOW)
    assert not both_yellow, (
        f"[{sim_time:.2f}ns] Safety Violation! Both streets are YELLOW: main={main_l:03b}, side={side_l:03b}"
    )


async def setup_clock_and_reset(dut):
    """Initializes clock, sets sensor to 0, asserts reset, and releases on falling edge."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.car_present.value = 0
    dut.reset_n.value = 0
    await Timer(15, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1


@cocotb.test()
async def test_fsm_reset(dut):
    """TC-FSM-01: Verify reset initializes to S_MAIN_GREEN with main GREEN and side RED."""
    await setup_clock_and_reset(dut)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_GREEN)
    assert_signal("main_signal", int(dut.main_signal.value), LIGHT_GREEN)
    assert_signal("side_signal", int(dut.side_signal.value), LIGHT_RED)
    assert_safety_invariant(dut)


@cocotb.test()
async def test_fsm_main_hold_without_car(dut):
    """TC-FSM-02: With car_present=0, FSM holds indefinitely in S_MAIN_GREEN."""
    await setup_clock_and_reset(dut)
    dut.car_present.value = 0

    for _ in range(20):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_GREEN)
        assert_signal("main_signal", int(dut.main_signal.value), LIGHT_GREEN)
        assert_signal("side_signal", int(dut.side_signal.value), LIGHT_RED)
        assert_safety_invariant(dut)


@cocotb.test()
async def test_fsm_full_traffic_cycle(dut):
    """TC-FSM-03, TC-FSM-04, TC-FSM-05, TC-FSM-07: Full state cycle with exact tick timings."""
    await setup_clock_and_reset(dut)
    dut.car_present.value = 1

    # In S_MAIN_GREEN: exactly TICKS_GREEN = 6 cycles (timer_cnt 0 -> 1 -> 2 -> 3 -> 4 -> 5)
    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_GREEN)
        assert_safety_invariant(dut)

    # Edge 6: transitions to S_MAIN_YELLOW (timer_cnt = 0, 1st cycle of yellow)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_YELLOW)
    assert_signal("main_signal", int(dut.main_signal.value), LIGHT_YELLOW)
    assert_signal("side_signal", int(dut.side_signal.value), LIGHT_RED)
    assert_safety_invariant(dut)

    # Edge 7: 2nd cycle of yellow (timer_cnt = 1)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_YELLOW)
    assert_safety_invariant(dut)

    # Edge 8: transitions to S_SIDE_GREEN (timer_cnt = 0, 1st cycle of side green)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_SIDE_GREEN)
    assert_signal("main_signal", int(dut.main_signal.value), LIGHT_RED)
    assert_signal("side_signal", int(dut.side_signal.value), LIGHT_GREEN)
    assert_safety_invariant(dut)

    # In S_SIDE_GREEN: cycles with timer_cnt 1 to 5 (5 edges)
    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("state_debug", int(dut.state_debug.value), S_SIDE_GREEN)
        assert_safety_invariant(dut)

    # Edge 14: transitions to S_SIDE_YELLOW (timer_cnt = 0, 1st cycle of side yellow)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_SIDE_YELLOW)
    assert_signal("main_signal", int(dut.main_signal.value), LIGHT_RED)
    assert_signal("side_signal", int(dut.side_signal.value), LIGHT_YELLOW)
    assert_safety_invariant(dut)

    # Edge 15: 2nd cycle of side yellow (timer_cnt = 1)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_SIDE_YELLOW)
    assert_safety_invariant(dut)

    # Edge 16: transitions back to S_MAIN_GREEN (timer_cnt = 0)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_GREEN)
    assert_signal("main_signal", int(dut.main_signal.value), LIGHT_GREEN)
    assert_signal("side_signal", int(dut.side_signal.value), LIGHT_RED)
    assert_safety_invariant(dut)


@cocotb.test()
async def test_fsm_side_early_termination(dut):
    """TC-FSM-06: S_SIDE_GREEN terminates early when car_present drops to 0."""
    await setup_clock_and_reset(dut)
    dut.car_present.value = 1

    # 6 edges in main green + 2 edges in main yellow = 8 edges
    for _ in range(8):
        await RisingEdge(dut.clk)

    await ReadOnly()
    # At 8th edge, enters S_SIDE_GREEN
    assert_signal("state_debug", int(dut.state_debug.value), S_SIDE_GREEN)

    # Run 1 cycle in S_SIDE_GREEN, then drop car_present to 0 on falling edge
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.car_present.value = 0

    # Next clock edge must transition to S_SIDE_YELLOW immediately
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_SIDE_YELLOW)
    assert_signal("side_signal", int(dut.side_signal.value), LIGHT_YELLOW)
    assert_safety_invariant(dut)
