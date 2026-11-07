"""
Visible Testbench for Counter Module
Covers TC-CNT-01 to TC-CNT-07 basic specification requirements.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer


def assert_signal(name: str, actual, expected):
    """Formats assertion failure with simulation timestamp, signal, expected, and actual."""
    sim_time = cocotb.utils.get_sim_time(unit="ns")
    assert actual == expected, (
        f"[{sim_time:.2f}ns] Signal '{name}' mismatch: expected {expected!r}, got {actual!r}"
    )


async def setup_clock_and_reset(dut):
    """Initializes clock and asserts active-low reset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.cnt_en.value = 0
    dut.limit_val.value = 9
    dut.reset_n.value = 0
    await Timer(15, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()


@cocotb.test()
async def test_counter_reset(dut):
    """TC-CNT-01: Verify async reset initializes count to 0 and clears term_reached."""
    await setup_clock_and_reset(dut)
    assert_signal("val_out", int(dut.val_out.value), 0)
    assert_signal("term_reached", int(dut.term_reached.value), 0)


@cocotb.test()
async def test_counter_basic_counting(dut):
    """TC-CNT-02, TC-CNT-04, TC-CNT-05: Sequential counting, terminal flag, and wrap-around."""
    await setup_clock_and_reset(dut)
    limit = 4
    await FallingEdge(dut.clk)
    dut.limit_val.value = limit
    dut.cnt_en.value = 1

    # Count from 0 up to limit
    for expected_val in range(1, limit + 1):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("val_out", int(dut.val_out.value), expected_val)
        expected_term = 1 if expected_val == limit else 0
        assert_signal("term_reached", int(dut.term_reached.value), expected_term)

    # Wrap-around check at next clock edge
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("val_out", int(dut.val_out.value), 0)
    assert_signal("term_reached", int(dut.term_reached.value), 0)


@cocotb.test()
async def test_counter_enable_freeze(dut):
    """TC-CNT-03: When cnt_en is 0, count value holds steady across cycles."""
    await setup_clock_and_reset(dut)
    await FallingEdge(dut.clk)
    dut.limit_val.value = 8
    dut.cnt_en.value = 1

    # Step twice to reach 2
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("val_out", int(dut.val_out.value), 2)

    # Disable counting on falling edge
    await FallingEdge(dut.clk)
    dut.cnt_en.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("val_out", int(dut.val_out.value), 2)
        assert_signal("term_reached", int(dut.term_reached.value), 0)


@cocotb.test()
async def test_counter_zero_limit(dut):
    """TC-CNT-06: Minimum corner case limit_val = 0."""
    await setup_clock_and_reset(dut)
    await FallingEdge(dut.clk)
    dut.limit_val.value = 0
    dut.cnt_en.value = 1

    for _ in range(4):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("val_out", int(dut.val_out.value), 0)
        assert_signal("term_reached", int(dut.term_reached.value), 1)


@cocotb.test()
async def test_counter_exceed_limit_recovery(dut):
    """TC-CNT-07: When val_out > limit_val and cnt_en is 1, wraps to 0 next cycle."""
    await setup_clock_and_reset(dut)
    await FallingEdge(dut.clk)
    dut.limit_val.value = 10
    dut.cnt_en.value = 1

    # Advance to 6
    for _ in range(6):
        await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("val_out", int(dut.val_out.value), 6)

    # Suddenly drop limit_val below current count on falling edge
    await FallingEdge(dut.clk)
    dut.limit_val.value = 3
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("val_out", int(dut.val_out.value), 0)
