// Verilog-2005
// Module: counter
// Synchronous up-counter with async reset, enable, and dynamic wrap limit.

`timescale 1ns / 1ps

module counter #(
    parameter WIDTH_BITS = 4
) (
    input  wire                  clk,
    input  wire                  reset_n,
    input  wire                  cnt_en,
    input  wire [WIDTH_BITS-1:0] limit_val,
    output reg  [WIDTH_BITS-1:0] val_out,
    output wire                  term_reached
);

    // Combinational terminal count flag
    assign term_reached = (val_out == limit_val);

    // Sequential counting logic
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            val_out <= {WIDTH_BITS{1'b0}};
        end else if (cnt_en) begin
            if (val_out > limit_val) begin
                val_out <= {WIDTH_BITS{1'b0}};
            end else begin
                val_out <= val_out + 1'b1;
            end
        end
    end

endmodule
