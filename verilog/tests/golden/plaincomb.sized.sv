module plaincomb (
  input  logic clk,
  input  logic go,
  input  logic [7:0] a,
  input  logic [7:0] b,
  input  logic [1:0] sel,
  output logic [7:0] y,
  output logic [7:0] m,
  output logic [1:0] state
);
  logic [1:0] next;
  logic [7:0] t;
  always_ff @(posedge clk) state <= next;
  always_comb case (state) 2'd0: next = (go ? 2'd1 : 2'd0); 2'd1: next = (go ? 2'd1 : 2'd2); default: next = 2'd0; endcase
  always_comb case (sel) 2'd0: m = a; 2'd1: m = b; default: m = (a & b); endcase
  always_comb begin t = (a + b); t = (t ^ {t[3:0], t[7:4]}); end
  always_comb y = ((state == 2'd1) ? t : m);
endmodule

