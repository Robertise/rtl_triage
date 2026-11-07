"""
Held-out Testbench for FSM Traffic Light Controller Module
Exhaustive stress tests: Independent reference model, continuous safety invariant
monitoring, pseudo-random sensor sequences (fixed seed), rapid pulse arrivals,
and mid-cycle asynchronous reset verification.
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer

# State codes
S_MAIN_GREEN = 0b00
S_MAIN_YELLOW = 0b01
S_SIDE_GREEN = 0b10
S_SIDE_YELLOW = 0b11

LIGHT_GREEN = 0b001
LIGHT_YELLOW = 0b010
LIGHT_RED = 0b100


def assert_signal(name: str, actual, expected):
    """Formats assertion failure with simulation timestamp, signal, expected, and actual."""
    sim_time = cocotb.utils.get_sim_time(unit="ns")
    assert actual == expected, (
        f"[{sim_time:.2f}ns] Signal '{name}' mismatch: expected {expected!r}, got {actual!r}"
    )


class ReferenceFSMModel:
    """Independent Python reference model implementing the traffic light specification."""
    def __init__(self, ticks_green: int = 6, ticks_yellow: int = 2):
        self.ticks_green = ticks_green
        self.ticks_yellow = ticks_yellow
        self.state = S_MAIN_GREEN
        self.timer = 0

    def step(self, car_present: int) -> int:
        if self.state == S_MAIN_GREEN:
            if self.timer < self.ticks_green - 1:
                self.timer += 1
            elif car_present:
                self.state = S_MAIN_YELLOW
                self.timer = 0
        elif self.state == S_MAIN_YELLOW:
            if self.timer == self.ticks_yellow - 1:
                self.state = S_SIDE_GREEN
                self.timer = 0
            else:
                self.timer += 1
        elif self.state == S_SIDE_GREEN:
            if not car_present or (self.timer == self.ticks_green - 1):
                self.state = S_SIDE_YELLOW
                self.timer = 0
            else:
                self.timer += 1
        elif self.state == S_SIDE_YELLOW:
            if self.timer == self.ticks_yellow - 1:
                self.state = S_MAIN_GREEN
                self.timer = 0
            else:
                self.timer += 1
        return self.state

    @staticmethod
    def expected_signals(state: int) -> tuple[int, int]:
        if state == S_MAIN_GREEN:
            return LIGHT_GREEN, LIGHT_RED
        elif state == S_MAIN_YELLOW:
            return LIGHT_YELLOW, LIGHT_RED
        elif state == S_SIDE_GREEN:
            return LIGHT_RED, LIGHT_GREEN
        elif state == S_SIDE_YELLOW:
            return LIGHT_RED, LIGHT_YELLOW
        return LIGHT_RED, LIGHT_RED


@cocotb.test()
async def test_heldout_fsm_continuous_invariant_and_random(dut):
    """300 cycles of pseudo-random sensor activity asserting state, signals, and safety invariants."""
    random.seed(67890)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.reset_n.value = 0
    dut.car_present.value = 0
    await Timer(12, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1

    model = ReferenceFSMModel(ticks_green=6, ticks_yellow=2)

    for step in range(300):
        sensor = random.choice([0, 1])
        await FallingEdge(dut.clk)
        dut.car_present.value = sensor

        await RisingEdge(dut.clk)
        await ReadOnly()

        exp_state = model.step(sensor)
        exp_main, exp_side = model.expected_signals(exp_state)

        assert_signal("state_debug", int(dut.state_debug.value), exp_state)
        assert_signal("main_signal", int(dut.main_signal.value), exp_main)
        assert_signal("side_signal", int(dut.side_signal.value), exp_side)

        # Continuous Safety Invariant checks
        main_l = int(dut.main_signal.value)
        side_l = int(dut.side_signal.value)
        sim_time = cocotb.utils.get_sim_time(unit="ns")

        assert not ((main_l & LIGHT_GREEN) and (side_l & LIGHT_GREEN)), (
            f"[{sim_time:.2f}ns] Safety invariant violated: both main and side are GREEN"
        )
        assert not ((main_l & LIGHT_YELLOW) and (side_l & LIGHT_YELLOW)), (
            f"[{sim_time:.2f}ns] Safety invariant violated: both main and side are YELLOW"
        )


@cocotb.test()
async def test_heldout_fsm_single_pulse_sensor(dut):
    """Test 1-cycle sensor pulses arriving at boundary tick conditions."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.reset_n.value = 0
    dut.car_present.value = 0
    await Timer(12, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1

    # Let main green run for 10 cycles (well past 6-tick minimum)
    for _ in range(10):
        await RisingEdge(dut.clk)

    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_GREEN)

    # Deliver exactly 1-cycle pulse of car_present
    await FallingEdge(dut.clk)
    dut.car_present.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_YELLOW)

    # Immediately drop sensor back to 0
    await FallingEdge(dut.clk)
    dut.car_present.value = 0

    # Yellow must still complete full 2 ticks (1st cycle was above, 2nd is next)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_YELLOW)

    # Then enter Side Green
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_SIDE_GREEN)

    # Since car_present is 0, Side Green must immediately transition to Side Yellow on next edge
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_SIDE_YELLOW)

    # Side Yellow runs 2 ticks (1st was above, 2nd is next) and returns to Main Green
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_SIDE_YELLOW)

    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_GREEN)


@cocotb.test()
async def test_heldout_fsm_async_reset_assertion(dut):
    """Verify asynchronous reset takes effect immediately mid-cycle, independent of clock edges."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.reset_n.value = 0
    dut.car_present.value = 1
    await Timer(12, unit="ns")
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1

    # Advance 6 clock cycles so FSM leaves S_MAIN_GREEN and enters S_MAIN_YELLOW
    for _ in range(6):
        await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_YELLOW)

    # Assert reset between clock edges (asynchronous)
    await Timer(3, unit="ns")
    dut.reset_n.value = 0
    await Timer(1, unit="ns")
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_GREEN)
    assert_signal("main_signal", int(dut.main_signal.value), LIGHT_GREEN)
    assert_signal("side_signal", int(dut.side_signal.value), LIGHT_RED)

    # Release reset on falling edge and verify full 6-cycle count from scratch
    await FallingEdge(dut.clk)
    dut.reset_n.value = 1
    dut.car_present.value = 1

    for _ in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_GREEN)

    # 6th clock edge transitions to S_MAIN_YELLOW
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert_signal("state_debug", int(dut.state_debug.value), S_MAIN_YELLOW)
    assert_signal("main_signal", int(dut.main_signal.value), LIGHT_YELLOW)
    assert_signal("side_signal", int(dut.side_signal.value), LIGHT_RED)
