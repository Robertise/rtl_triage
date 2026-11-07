"""
Visible Testbench for ALU Module
Covers TC-ALU-01 to TC-ALU-09 basic specification requirements.
"""

import cocotb
from cocotb.triggers import Timer


def assert_signal(name: str, actual, expected):
    """Formats assertion failure with simulation timestamp, signal, expected, and actual."""
    sim_time = cocotb.utils.get_sim_time(unit="ns")
    assert actual == expected, (
        f"[{sim_time:.2f}ns] Signal '{name}' mismatch: expected {expected!r}, got {actual!r}"
    )


async def check_alu(dut, op_code: int, a: int, b: int, exp_res: int, exp_zero: int, exp_carry: int):
    """Applies combinational stimulus and checks outputs after propagation delay."""
    dut.op_code.value = op_code
    dut.src_a.value = a
    dut.src_b.value = b
    await Timer(1, unit="ns")
    assert_signal("alu_res", int(dut.alu_res.value), exp_res)
    assert_signal("flag_zero", int(dut.flag_zero.value), exp_zero)
    assert_signal("flag_carry", int(dut.flag_carry.value), exp_carry)


@cocotb.test()
async def test_alu_add(dut):
    """TC-ALU-01, TC-ALU-02: Addition without overflow and with overflow/zero."""
    # TC-ALU-01: Regular addition
    await check_alu(dut, op_code=0b000, a=0x15, b=0x23, exp_res=0x38, exp_zero=0, exp_carry=0)
    # TC-ALU-02: Overflow wrapping to 0
    await check_alu(dut, op_code=0b000, a=0xFF, b=0x01, exp_res=0x00, exp_zero=1, exp_carry=1)


@cocotb.test()
async def test_alu_sub(dut):
    """TC-ALU-03, TC-ALU-04, TC-ALU-05: Subtraction positive, zero, and borrow."""
    # TC-ALU-03: Positive result (A > B)
    await check_alu(dut, op_code=0b001, a=0x50, b=0x20, exp_res=0x30, exp_zero=0, exp_carry=0)
    # TC-ALU-04: Equal operands zero result (A == B)
    await check_alu(dut, op_code=0b001, a=0x4A, b=0x4A, exp_res=0x00, exp_zero=1, exp_carry=0)
    # TC-ALU-05: Borrow condition (A < B)
    await check_alu(dut, op_code=0b001, a=0x02, b=0x05, exp_res=0xFD, exp_zero=0, exp_carry=1)


@cocotb.test()
async def test_alu_logic_ops(dut):
    """TC-ALU-06, TC-ALU-07, TC-ALU-08: Bitwise AND, OR, and XOR operations."""
    # TC-ALU-06: AND operation
    await check_alu(dut, op_code=0b010, a=0xAA, b=0x55, exp_res=0x00, exp_zero=1, exp_carry=0)
    await check_alu(dut, op_code=0b010, a=0xFF, b=0x0F, exp_res=0x0F, exp_zero=0, exp_carry=0)

    # TC-ALU-07: OR operation
    await check_alu(dut, op_code=0b011, a=0xF0, b=0x0F, exp_res=0xFF, exp_zero=0, exp_carry=0)

    # TC-ALU-08: XOR operation
    await check_alu(dut, op_code=0b100, a=0xAA, b=0xAA, exp_res=0x00, exp_zero=1, exp_carry=0)
    await check_alu(dut, op_code=0b100, a=0xF0, b=0xAA, exp_res=0x5A, exp_zero=0, exp_carry=0)


@cocotb.test()
async def test_alu_invalid_opcode(dut):
    """TC-ALU-09: Unimplemented opcodes return safe defaults (alu_res=0, flag_carry=0, flag_zero=1)."""
    for invalid_op in [0b101, 0b110, 0b111]:
        await check_alu(dut, op_code=invalid_op, a=0x77, b=0x88, exp_res=0x00, exp_zero=1, exp_carry=0)
