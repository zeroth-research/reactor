// A Moore FSM with `(* parallel_case *)` on its `case` statements, as in
// hand-written code for synthesis. The labels are distinct constants
// (integer localparams, compared with the 3-bit state at 32 bits; two
// labels in one item), so the attribute changes nothing: the first
// matching item is the only one.
module parcase(input logic clk, input logic go, input logic [7:0] d,
               output logic [1:0] mode, output logic [7:0] y);
  localparam IDLE = 0;
  localparam LOAD = 1;
  localparam RUN = 2;
  localparam HOLD = 3;
  localparam DONE = 4;

  logic [2:0] st = 3'd0;
  logic [7:0] acc;
  always_ff @(posedge clk) begin
    (* parallel_case *)
    case (st)
      IDLE: if (go) st <= LOAD;
      LOAD: begin acc <= d; st <= RUN; end
      RUN: begin acc <= acc + d; if (!go) st <= HOLD; end
      HOLD, DONE: st <= go ? RUN : DONE;
      default: st <= IDLE;
    endcase
  end

  always_comb begin
    (* parallel_case *)
    case (st)
      IDLE: begin mode = 2'd0; y = 8'd0; end
      LOAD, RUN: begin mode = 2'd1; y = acc; end
      HOLD: begin mode = 2'd2; y = acc; end
      default: begin mode = 2'd3; y = ~acc; end
    endcase
  end
endmodule
