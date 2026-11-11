// A register file with signed elements (C-style size) and a second array
// with a descending range: combinational reads, element selects, a
// constant-index write after a variable-index one (the last write wins),
// a part of an element written with a variable element index, and a
// write with an index too narrow to reach the upper elements.
module regfile (
    input  logic              clk,
    input  logic              we,
    input  logic       [2:0]  wa,
    input  logic       [1:0]  ra,
    input  logic       [1:0]  rb,
    input  logic signed [7:0] d,
    output logic signed [7:0] a,
    output logic signed [8:0] sum,
    output logic       [3:0]  lo,
    output logic              b
);
  logic signed [7:0] r[8];
  logic [7:0] s[3:0];
  always_ff @(posedge clk) begin
    if (we) r[wa] <= d;
    else r[rb] <= -d;
    r[0] <= 8'sd0;
  end
  always_ff @(negedge clk) begin
    s[ra] <= d;
    if (we) s[rb][7:4] <= 4'b1010;
  end
  assign a = r[{1'b0, ra}];
  assign sum = r[wa] + r[3];
  assign lo = s[ra][3:0];
  assign b = s[rb][ra];
endmodule
