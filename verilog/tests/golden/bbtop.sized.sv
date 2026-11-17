(* blackbox *)
module clkgen (
  input  logic clk_in,
  output logic clk_out,
  output logic locked
);
endmodule

(* blackbox *)
module por (
  input  logic clk,
  output logic rst
);
endmodule

(* blackbox *)
module pad (
  input  logic clk,
  input  logic d_out,
  input  logic oe,
  output logic d_in,
  output logic d_in_neg,
  inout  wire pin
);
endmodule

(* blackbox *)
module boot (
  input  logic go,
  input  logic [1:0] image
);
endmodule

module bbtop (
  input  logic clk12,
  input  logic en,
  output logic [3:0] count,
  output logic [1:0] seen,
  output logic slow_seen,
  inout  wire pin_a,
  inout  wire pin_b
);
  logic clk;
  logic rst;
  logic a;
  logic b_pos;
  logic b_neg;
  logic slow;
  logic [1:0] img;
  clkgen g (.clk_in(clk12), .clk_out(clk), .locked());
  por r (.clk(clk), .rst(rst));
  pad pa (.clk(clk), .d_out(count[0]), .oe(1'd0), .d_in(a), .d_in_neg(), .pin(pin_a));
  pad pb (.clk(clk), .d_out((en & a)), .oe(en), .d_in(b_pos), .d_in_neg(b_neg), .pin(pin_b));
  boot bt (.go((count == 4'd15)), .image(img));
  assign img = (seen ^ {a, a});
  always_ff @(posedge clk or posedge rst) if (rst) begin count <= 4'd0; seen <= 2'd0; end else begin if (en) count <= (count + 4'd1); else begin end seen <= {b_pos, b_neg}; end
  always_ff @(posedge clk) slow <= (!slow);
  always_ff @(posedge slow) slow_seen <= a;
endmodule

