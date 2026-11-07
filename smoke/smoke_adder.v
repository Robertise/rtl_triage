`timescale 1ns/1ps

module smoke_adder (
    input  wire [3:0] a,
    input  wire [3:0] b,
    output wire [4:0] sum
);
    assign sum = a + b;
endmodule
