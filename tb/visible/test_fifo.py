"""
Visible Testbench for Synchronous FIFO Module
Covers TC-FIFO-01 to TC-FIFO-10 basic specification requirements.
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
    """Initializes clock, clears requests, and applies active-low reset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.push_req.value = 0
    dut.pop_req.value = 0
    dut.data_in.value = 0
    dut.reset_n.value = 0
    await Timer(15, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()


@cocotb.test()
async def test_fifo_reset(dut):
    """TC-FIFO-01: Verify reset state (empty=1, full=0, item_count=0, data_out=0)."""
    await setup_clock_and_reset(dut)
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 1)
    assert_signal("fifo_full", int(dut.fifo_full.value), 0)
    assert_signal("item_count", int(dut.item_count.value), 0)
    assert_signal("data_out", int(dut.data_out.value), 0)


@cocotb.test()
async def test_fifo_single_push_pop(dut):
    """TC-FIFO-02, TC-FIFO-03: Single push updates empty flag, single pop retrieves data with 1-cycle latency."""
    await setup_clock_and_reset(dut)

    # Push 0x42
    await FallingEdge(dut.clk)
    dut.push_req.value = 1
    dut.data_in.value = 0x42
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("item_count", int(dut.item_count.value), 1)
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 0)

    # Stop push, request pop
    await FallingEdge(dut.clk)
    dut.push_req.value = 0
    dut.pop_req.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("data_out", int(dut.data_out.value), 0x42)
    assert_signal("item_count", int(dut.item_count.value), 0)
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 1)

    # Clear pop
    await FallingEdge(dut.clk)
    dut.pop_req.value = 0


@cocotb.test()
async def test_fifo_fill_and_overflow(dut):
    """TC-FIFO-04, TC-FIFO-05: Fill FIFO to depth (8), verify full flag, and test overflow protection."""
    await setup_clock_and_reset(dut)

    # Push 8 elements: 0x10 to 0x17
    for val in range(0x10, 0x18):
        await FallingEdge(dut.clk)
        dut.push_req.value = 1
        dut.data_in.value = val
        await RisingEdge(dut.clk)

    await ReadOnly()
    assert_signal("item_count", int(dut.item_count.value), 8)
    assert_signal("fifo_full", int(dut.fifo_full.value), 1)
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 0)

    # TC-FIFO-05: Try 9th push when full (overflow protection)
    await FallingEdge(dut.clk)
    dut.push_req.value = 1
    dut.data_in.value = 0x99
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("item_count", int(dut.item_count.value), 8)
    assert_signal("fifo_full", int(dut.fifo_full.value), 1)

    # Pop all 8 items and verify first item is still 0x10 (not overwritten by 0x99)
    await FallingEdge(dut.clk)
    dut.push_req.value = 0
    dut.pop_req.value = 1

    for exp_val in range(0x10, 0x18):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("data_out", int(dut.data_out.value), exp_val)

    await FallingEdge(dut.clk)
    dut.pop_req.value = 0


@cocotb.test()
async def test_fifo_empty_and_underflow(dut):
    """TC-FIFO-06: Attempt pop when empty; data_out holds previous value and pointers stay unchanged."""
    await setup_clock_and_reset(dut)

    # First put and read one item to set data_out to a known pattern 0x5A
    await FallingEdge(dut.clk)
    dut.push_req.value = 1
    dut.data_in.value = 0x5A
    await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    dut.push_req.value = 0
    dut.pop_req.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("data_out", int(dut.data_out.value), 0x5A)
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 1)

    # Now attempt pop on empty FIFO
    for _ in range(3):
        await FallingEdge(dut.clk)
        dut.pop_req.value = 1
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("data_out", int(dut.data_out.value), 0x5A)  # Must hold previous value
        assert_signal("item_count", int(dut.item_count.value), 0)
        assert_signal("fifo_empty", int(dut.fifo_empty.value), 1)

    await FallingEdge(dut.clk)
    dut.pop_req.value = 0


@cocotb.test()
async def test_fifo_simultaneous_rw(dut):
    """TC-FIFO-07, TC-FIFO-08, TC-FIFO-09: Simultaneous read and write across empty, mid, and full."""
    await setup_clock_and_reset(dut)

    # TC-FIFO-09: Simultaneous R/W when EMPTY -> pop rejected (data_out holds 0), push accepted
    await FallingEdge(dut.clk)
    dut.push_req.value = 1
    dut.pop_req.value = 1
    dut.data_in.value = 0xA1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("item_count", int(dut.item_count.value), 1)
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 0)
    assert_signal("data_out", int(dut.data_out.value), 0)  # Pop was rejected, holds 0

    # Add second item
    await FallingEdge(dut.clk)
    dut.push_req.value = 1
    dut.pop_req.value = 0
    dut.data_in.value = 0xB2
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("item_count", int(dut.item_count.value), 2)

    # TC-FIFO-07: Simultaneous R/W when middle (count=2) -> count stays 2, pop 0xA1, push 0xC3
    await FallingEdge(dut.clk)
    dut.push_req.value = 1
    dut.pop_req.value = 1
    dut.data_in.value = 0xC3
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("data_out", int(dut.data_out.value), 0xA1)
    assert_signal("item_count", int(dut.item_count.value), 2)

    # Fill to full (currently has 0xB2, 0xC3) -> push 6 more
    for val in range(0xD4, 0xDA):
        await FallingEdge(dut.clk)
        dut.push_req.value = 1
        dut.pop_req.value = 0
        dut.data_in.value = val
        await RisingEdge(dut.clk)

    await ReadOnly()
    assert_signal("fifo_full", int(dut.fifo_full.value), 1)

    # TC-FIFO-08: Simultaneous R/W when FULL -> count stays 8, pops 0xB2, pushes 0xEE
    await FallingEdge(dut.clk)
    dut.push_req.value = 1
    dut.pop_req.value = 1
    dut.data_in.value = 0xEE
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("data_out", int(dut.data_out.value), 0xB2)
    assert_signal("fifo_full", int(dut.fifo_full.value), 1)
    assert_signal("item_count", int(dut.item_count.value), 8)

    await FallingEdge(dut.clk)
    dut.push_req.value = 0
    dut.pop_req.value = 0


@cocotb.test()
async def test_fifo_pointer_rollover(dut):
    """TC-FIFO-10: Continuously push and pop across multiple memory wraps."""
    await setup_clock_and_reset(dut)

    # Cycle 25 items through 8-deep FIFO (wrap pointers > 3 times)
    for i in range(25):
        val = (i * 7 + 3) & 0xFF
        await FallingEdge(dut.clk)
        dut.push_req.value = 1
        dut.pop_req.value = 0
        dut.data_in.value = val
        await RisingEdge(dut.clk)

        await FallingEdge(dut.clk)
        dut.push_req.value = 0
        dut.pop_req.value = 1
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("data_out", int(dut.data_out.value), val)

    await FallingEdge(dut.clk)
    dut.pop_req.value = 0
    await ReadOnly()
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 1)
