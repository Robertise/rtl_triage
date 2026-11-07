// Verilog-2005
// Module: fifo
// Synchronous FIFO with independent status flags and simultaneous push/pop.

`timescale 1ns / 1ps

module fifo #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 3
) (
    input  wire                  clk,
    input  wire                  reset_n,
    input  wire                  push_req,
    input  wire                  pop_req,
    input  wire [DATA_WIDTH-1:0] data_in,
    output reg  [DATA_WIDTH-1:0] data_out,
    output wire                  fifo_full,
    output wire                  fifo_empty,
    output reg  [ADDR_WIDTH:0]   item_count
);

    localparam FIFO_DEPTH = (1 << ADDR_WIDTH);

    reg [DATA_WIDTH-1:0] mem [0:FIFO_DEPTH-1];
    reg [ADDR_WIDTH-1:0] wr_ptr;
    reg [ADDR_WIDTH-1:0] rd_ptr;

    // Status flags
    assign fifo_empty = (item_count == {(ADDR_WIDTH+1){1'b0}});
    assign fifo_full  = (item_count == (FIFO_DEPTH - 1));

    // Operation qualifiers
    wire do_push = push_req && (!fifo_full || pop_req);
    wire do_pop  = pop_req && !fifo_empty;

    // Sequential memory access and pointer tracking
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            wr_ptr     <= {ADDR_WIDTH{1'b0}};
            rd_ptr     <= {ADDR_WIDTH{1'b0}};
            item_count <= {(ADDR_WIDTH+1){1'b0}};
            data_out   <= {DATA_WIDTH{1'b0}};
        end else begin
            // Write port
            if (do_push) begin
                mem[wr_ptr] <= data_in;
                wr_ptr      <= wr_ptr + 1'b1;
            end

            // Read port (registered latency = 1 cycle)
            if (do_pop) begin
                data_out <= mem[rd_ptr];
                rd_ptr   <= rd_ptr + 1'b1;
            end

            // Occupancy counter update
            case ({do_push, do_pop})
                2'b10: item_count <= item_count + 1'b1;
                2'b01: item_count <= item_count - 1'b1;
                default: item_count <= item_count;
            endcase
        end
    end

endmodule
