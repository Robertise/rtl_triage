"""
Held-out Testbench for Synchronous FIFO Module
Exhaustive stress tests: Independent Python software queue model, long pseudo-random
stimulus sequence (fixed seed), burst operations, and mid-cycle asynchronous reset verification.
"""

from collections import deque
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


class ReferenceFifoModel:
    """Independent software oracle tracking FIFO state according to specification."""
    def __init__(self, depth: int = 8):
        self.depth = depth
        self.queue = deque()
        self.last_data_out = 0

    def step(self, push_req: int, pop_req: int, data_in: int):
        is_full = len(self.queue) >= self.depth
        is_empty = len(self.queue) == 0

        do_push = push_req and (not is_full or pop_req)
        do_pop = pop_req and not is_empty

        if do_pop:
            self.last_data_out = self.queue.popleft()

        if do_push:
            self.queue.append(data_in)

        return (
            len(self.queue),
            1 if len(self.queue) == 0 else 0,
            1 if len(self.queue) == self.depth else 0,
            self.last_data_out,
        )


@cocotb.test()
async def test_heldout_fifo_random_stress(dut):
    """Stress test with 200 cycles of pseudo-random push and pop combinations against reference model."""
    random.seed(54321)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.reset_n.value = 0
    dut.push_req.value = 0
    dut.pop_req.value = 0
    dut.data_in.value = 0
    await Timer(15, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1

    model = ReferenceFifoModel(depth=8)

    for step in range(200):
        push = random.choice([0, 1, 1])
        pop = random.choice([0, 1, 1])
        data = random.randint(0, 255)

        await FallingEdge(dut.clk)
        dut.push_req.value = push
        dut.pop_req.value = pop
        dut.data_in.value = data

        await RisingEdge(dut.clk)
        await ReadOnly()

        exp_count, exp_empty, exp_full, exp_out = model.step(push, pop, data)
        assert_signal("item_count", int(dut.item_count.value), exp_count)
        assert_signal("fifo_empty", int(dut.fifo_empty.value), exp_empty)
        assert_signal("fifo_full", int(dut.fifo_full.value), exp_full)
        assert_signal("data_out", int(dut.data_out.value), exp_out)


@cocotb.test()
async def test_heldout_fifo_burst_alternating(dut):
    """Alternating bursts of full filling followed by complete emptying."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.reset_n.value = 0
    dut.push_req.value = 0
    dut.pop_req.value = 0
    await Timer(12, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1

    for burst in range(5):
        # Burst push 8 elements
        base_val = burst * 20
        for i in range(8):
            await FallingEdge(dut.clk)
            dut.push_req.value = 1
            dut.pop_req.value = 0
            dut.data_in.value = (base_val + i) & 0xFF
            await RisingEdge(dut.clk)

        await ReadOnly()
        assert_signal("fifo_full", int(dut.fifo_full.value), 1)

        # Burst pop 8 elements
        for i in range(8):
            await FallingEdge(dut.clk)
            dut.push_req.value = 0
            dut.pop_req.value = 1
            await RisingEdge(dut.clk)
            await ReadOnly()
            assert_signal("data_out", int(dut.data_out.value), (base_val + i) & 0xFF)

        assert_signal("fifo_empty", int(dut.fifo_empty.value), 1)


@cocotb.test()
async def test_heldout_fifo_async_reset_assertion(dut):
    """Verify asynchronous reset takes effect immediately mid-cycle, independent of clock edges."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.reset_n.value = 0
    dut.push_req.value = 0
    dut.pop_req.value = 0
    dut.data_in.value = 0
    await Timer(12, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1

    # Push 0xAA and 0xBB, then pop once so item_count > 0 and data_out != 0
    await FallingEdge(dut.clk)
    dut.push_req.value = 1
    dut.data_in.value = 0xAA
    await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    dut.push_req.value = 1
    dut.data_in.value = 0xBB
    await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    dut.push_req.value = 0
    dut.pop_req.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("item_count", int(dut.item_count.value), 1)
    assert_signal("data_out", int(dut.data_out.value), 0xAA)

    # Assert reset between clock edges (asynchronous)
    await Timer(3, unit="ns")
    dut.pop_req.value = 0
    dut.reset_n.value = 0
    await Timer(1, unit="ns")
    assert_signal("item_count", int(dut.item_count.value), 0)
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 1)
    assert_signal("fifo_full", int(dut.fifo_full.value), 0)
    assert_signal("data_out", int(dut.data_out.value), 0)

    # Release reset on falling edge and verify clean push & pop from scratch
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1
    dut.push_req.value = 1
    dut.pop_req.value = 0
    dut.data_in.value = 0x55
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("item_count", int(dut.item_count.value), 1)
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 0)

    await FallingEdge(dut.clk)
    dut.push_req.value = 0
    dut.pop_req.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("data_out", int(dut.data_out.value), 0x55)
    assert_signal("item_count", int(dut.item_count.value), 0)
    assert_signal("fifo_empty", int(dut.fifo_empty.value), 1)
