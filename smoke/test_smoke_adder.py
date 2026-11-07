import cocotb
from cocotb.triggers import Timer

@cocotb.test()
async def test_smoke_adder_basic(dut):
    """Test basic addition cases on smoke_adder."""
    test_cases = [
        (0, 0, 0),
        (5, 3, 8),
        (15, 1, 16),
        (15, 15, 30),
        (7, 9, 16),
    ]

    for a, b, expected in test_cases:
        dut.a.value = a
        dut.b.value = b
        await Timer(1, unit="ns")
        actual = int(dut.sum.value)
        assert actual == expected, f"Mismatch: {a} + {b} = {actual}, expected {expected}"
