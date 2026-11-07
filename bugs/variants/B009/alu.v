// Verilog-2005
// Module: alu
// Combinational Arithmetic Logic Unit with zero and carry/borrow flags.

`timescale 1ns / 1ps

module alu #(
    parameter DATA_WIDTH = 8
) (
    input  wire [2:0]            op_code,
    input  wire [DATA_WIDTH-1:0] src_a,
    input  wire [DATA_WIDTH-1:0] src_b,
    output reg  [DATA_WIDTH-1:0] alu_res,
    output wire                  flag_zero,
    output reg                   flag_carry
);

    // Combinational zero flag
    assign flag_zero = (alu_res == {DATA_WIDTH{1'b0}});

    // Arithmetic and logic evaluation
    always @(*) begin
        case (op_code)
            3'b000: begin // ADD
                {flag_carry, alu_res} = {1'b0, src_a + src_b};
            end
            3'b001: begin // SUB (flag_carry is borrow)
                alu_res    = src_a - src_b;
                flag_carry = (src_a < src_b);
            end
            3'b010: begin // AND
                alu_res    = src_a & src_b;
                flag_carry = 1'b0;
            end
            3'b011: begin // OR
                alu_res    = src_a | src_b;
                flag_carry = 1'b0;
            end
            3'b100: begin // XOR
                alu_res    = src_a ^ src_b;
                flag_carry = 1'b0;
            end
            default: begin // Safe fallback
                alu_res    = {DATA_WIDTH{1'b0}};
                flag_carry = 1'b0;
            end
        endcase
    end

endmodule
