// Verilog-2005
// Module: fsm_controller
// Moore FSM traffic light controller with side sensor and timer intervals.

`timescale 1ns / 1ps

module fsm_controller #(
    parameter TICKS_GREEN  = 4'd6,
    parameter TICKS_YELLOW = 4'd2
) (
    input  wire       clk,
    input  wire       reset_n,
    input  wire       car_present,
    output wire [2:0] main_signal,
    output wire [2:0] side_signal,
    output wire [1:0] state_debug
);

    // State definitions
    localparam S_MAIN_GREEN  = 2'b00;
    localparam S_MAIN_YELLOW = 2'b01;
    localparam S_SIDE_GREEN  = 2'b10;
    localparam S_SIDE_YELLOW = 2'b11;

    reg [1:0] current_state;
    reg [3:0] timer_cnt;

    // Moore combinational outputs
    assign state_debug = current_state;

    assign main_signal = (current_state == S_MAIN_GREEN)  ? 3'b001 :
                         (current_state == S_MAIN_YELLOW) ? 3'b010 : 3'b100;

    assign side_signal = (current_state == S_SIDE_GREEN)  ? 3'b001 :
                         (current_state == S_SIDE_YELLOW) ? 3'b010 : 3'b100;

    // Sequential state transition and interval counter
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            current_state <= S_MAIN_GREEN;
            timer_cnt     <= 4'd0;
        end else begin
            case (current_state)
                S_MAIN_GREEN: begin
                    if (timer_cnt < TICKS_GREEN - 1'b1) begin
                        timer_cnt <= timer_cnt + 1'b1;
                    end else if (car_present) begin
                        current_state <= S_MAIN_YELLOW;
                        timer_cnt     <= 4'd0;
                    end
                end

                S_MAIN_YELLOW: begin
                    if (timer_cnt == TICKS_YELLOW - 1'b1) begin
                        current_state <= S_SIDE_GREEN;
                    end else begin
                        timer_cnt <= timer_cnt + 1'b1;
                    end
                end

                S_SIDE_GREEN: begin
                    if (!car_present || (timer_cnt == TICKS_GREEN - 1'b1)) begin
                        current_state <= S_SIDE_YELLOW;
                        timer_cnt     <= 4'd0;
                    end else begin
                        timer_cnt <= timer_cnt + 1'b1;
                    end
                end

                S_SIDE_YELLOW: begin
                    if (timer_cnt == TICKS_YELLOW - 1'b1) begin
                        current_state <= S_MAIN_GREEN;
                        timer_cnt     <= 4'd0;
                    end else begin
                        timer_cnt <= timer_cnt + 1'b1;
                    end
                end

                default: begin
                    // Defensive unreachable state recovery
                    current_state <= S_MAIN_GREEN;
                    timer_cnt     <= 4'd0;
                end
            endcase
        end
    end

endmodule
