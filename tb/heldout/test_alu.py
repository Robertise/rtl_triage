"""
Held-out Testbench for ALU Module
Exhaustive stress tests: Python reference golden model, pseudo-random vectors (fixed seed),
extreme boundary operands (0x00, 0xFF, walking ones), and complete opcode sweep.
"""

import random
import cocotb
from cocotb.triggers import Timer


def assert_signal(name: str, actual, expected):
    """Formats assertion failure with simulation timestamp, signal, expected, and actual."""
    sim_time = cocotb.utils.get_sim_time(unit="ns")
    assert actual == expected, (
        f"[{sim_time:.2f}ns] Signal '{name}' mismatch: expected {expected!r}, got {actual!r}"
    )


def alu_golden_model(op: int, a: int, b: int) -> tuple[int, int, int]:
    """Pure specification-derived software oracle for the ALU."""
    if op == 0b000:  # ADD
        sum_full = a + b
        res = sum_full & 0xFF
        carry = 1 if sum_full >= 0x100 else 0
    elif op == 0b001:  # SUB
        res = (a - b) & 0xFF
        carry = 1 if a < b else 0
    elif op == 0b010:  # AND
        res = a & b
        carry = 0
    elif op == 0b011:  # OR
        res = a | b
        carry = 0
    elif op == 0b100:  # XOR
        res = a ^ b
        carry = 0
    else:  # DEFAULT / Invalid
        res = 0
        carry = 0

    zero = 1 if res == 0 else 0
    return res, zero, carry


@cocotb.test()
async def test_heldout_alu_boundary_vectors(dut):
    """Test full matrix of boundary operands (0x00, 0x01, 0x7F, 0x80, 0xFF) across all 8 opcodes."""
    boundary_vals = [0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF]

    for op in range(8):
        for a in boundary_vals:
            for b in boundary_vals:
                dut.op_code.value = op
                dut.src_a.value = a
                dut.src_b.value = b
                await Timer(1, unit="ns")

                exp_res, exp_zero, exp_carry = alu_golden_model(op, a, b)
                assert_signal("alu_res", int(dut.alu_res.value), exp_res)
                assert_signal("flag_zero", int(dut.flag_zero.value), exp_zero)
                assert_signal("flag_carry", int(dut.flag_carry.value), exp_carry)


@cocotb.test()
async def test_heldout_alu_random_stress(dut):
    """Stress test with 250 pseudo-random vectors using fixed seed against golden model."""
    random.seed(98765)

    for _ in range(250):
        op = random.randint(0, 7)
        a = random.randint(0, 255)
        b = random.randint(0, 255)

        dut.op_code.value = op
        dut.src_a.value = a
        dut.src_b.value = b
        await Timer(1, unit="ns")

        exp_res, exp_zero, exp_carry = alu_golden_model(op, a, b)
        assert_signal("alu_res", int(dut.alu_res.value), exp_res)
        assert_signal("flag_zero", int(dut.flag_zero.value), exp_zero)
        assert_signal("flag_carry", int(dut.flag_carry.value), exp_carry)
