// Blackboxes at the top level, in the shape of an FPGA top: a clock
// generator (its output clocks the design, so it is an input clock), a
// power-on reset, I/O cells on `inout` pins, and a cell with inputs only.
// A blackbox has ports and parameters but no body; its outputs may take
// any value in every time step, its inputs are not read.

(* blackbox *)
module clkgen (input logic clk_in, output logic clk_out, output logic locked);
endmodule

(* blackbox *)
module por (input logic clk, output logic rst);
endmodule

(* blackbox *)
module pad #(parameter [5:0] PIN_TYPE = 6'b000000) (
  inout        pin,
  input  logic clk,
  input  logic d_out,
  input  logic oe,
  output logic d_in,
  output logic d_in_neg
);
endmodule

(* blackbox *)
module boot (input logic go, input logic [1:0] image);
endmodule

module bbtop (
  input  logic       clk12,
  input  logic       en,
  inout              pin_a,
  inout              pin_b,
  output logic [3:0] count,
  output logic [1:0] seen,
  output logic       slow_seen
);
  logic clk, rst, a, b_pos, b_neg, slow;
  logic [1:0] img;
  clkgen g (.clk_in(clk12), .clk_out(clk), .locked());
  por r (.clk(clk), .rst(rst));
  pad #(.PIN_TYPE(6'b110000)) pa (
    .pin(pin_a), .clk(clk), .d_out(count[0]), .oe(1'b0), .d_in(a), .d_in_neg()
  );
  pad #(.PIN_TYPE(6'b110000)) pb (
    .pin(pin_b), .clk(clk), .d_out(en & a), .oe(en), .d_in(b_pos), .d_in_neg(b_neg)
  );
  boot bt (.go(count == 4'hf), .image(img));
  assign img = seen ^ {a, a};
  // clocked by a blackbox output, reset by another
  always_ff @(posedge clk or posedge rst)
    if (rst) begin
      count <= 4'd0;
      seen <= 2'b00;
    end else begin
      if (en) count <= count + 4'd1;
      seen <= {b_pos, b_neg};
    end
  // a clock derived from it, reading a blackbox output
  always_ff @(posedge clk) slow <= !slow;
  always_ff @(posedge slow) slow_seen <= a;
endmodule
