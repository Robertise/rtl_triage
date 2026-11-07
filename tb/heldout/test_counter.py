"""
Held-out Testbench for Counter Module
Exhaustive stress tests: maximum limit boundary, rapid enable modulation,
pseudo-random stimulus sequence (fixed seed), and mid-cycle asynchronous reset.
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer


def assert_signal(name: str, actual, expected):
    """Formats assertion failure with simulation timestamp, signal, expected, and actual."""
    sim_time = cocotb.utils.get_sim_time(unit="ns")
    assert actual == expected, (
        f"[{sim_time:.2f}ns] Signal '{name}' mismatch: expected {expected!r}, got {actual!r}"
    )


class ReferenceCounterModel:
    """Independent Python golden model for counter specification."""
    def __init__(self, width: int = 4):
        self.width = width
        self.val = 0

    def step(self, cnt_en: int, limit: int) -> int:
        if cnt_en:
            if self.val >= limit:
                self.val = 0
            else:
                self.val = self.val + 1
        return self.val


@cocotb.test()
async def test_heldout_counter_max_limit(dut):
    """Test full 4-bit range up to maximum limit (15) across two complete wrap cycles."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.reset_n.value = 0
    dut.cnt_en.value = 0
    dut.limit_val.value = 15
    await Timer(12, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1
    dut.cnt_en.value = 1

    for cycle in range(32):
        await RisingEdge(dut.clk)
        await ReadOnly()
        expected_val = (cycle + 1) % 16
        expected_term = 1 if expected_val == 15 else 0
        assert_signal("val_out", int(dut.val_out.value), expected_val)
        assert_signal("term_reached", int(dut.term_reached.value), expected_term)


@cocotb.test()
async def test_heldout_counter_random_sequence(dut):
    """Stress test with fixed-seed pseudo-random enable and limit changes compared against model."""
    random.seed(42)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.reset_n.value = 0
    dut.cnt_en.value = 0
    dut.limit_val.value = 7
    await Timer(15, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1

    model = ReferenceCounterModel(width=4)

    for step in range(120):
        en = random.choice([0, 1, 1, 1])
        limit = random.randint(1, 15)
        await FallingEdge(dut.clk)
        dut.cnt_en.value = en
        dut.limit_val.value = limit

        await RisingEdge(dut.clk)
        await ReadOnly()
        expected = model.step(en, limit)
        assert_signal("val_out", int(dut.val_out.value), expected)
        assert_signal("term_reached", int(dut.term_reached.value), 1 if expected == limit else 0)


@cocotb.test()
async def test_heldout_counter_async_reset_assertion(dut):
    """Verify asynchronous reset takes effect immediately mid-cycle, independent of clock edges."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.reset_n.value = 0
    dut.cnt_en.value = 1
    dut.limit_val.value = 8
    await Timer(12, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1

    # Count for a few cycles
    for _ in range(4):
        await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("val_out", int(dut.val_out.value), 4)

    # Assert reset between clock edges (asynchronous)
    await Timer(3, unit="ns")
    dut.reset_n.value = 0
    await Timer(1, unit="ns")
    assert_signal("val_out", int(dut.val_out.value), 0)
    assert_signal("term_reached", int(dut.term_reached.value), 0)

    # Release reset on falling edge
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("val_out", int(dut.val_out.value), 1)
