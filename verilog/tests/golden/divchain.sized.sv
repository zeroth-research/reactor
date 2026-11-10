module divchain (
  input  logic clk,
  input  logic [3:0] a,
  output logic [3:0] y
);
  logic s1;
  logic s2;
  always_ff @(posedge s2) y <= (y + a);
  always_ff @(negedge s1) s2 <= (!s2);
  always_ff @(posedge clk) s1 <= (!s1);
endmodule

