// A register written one nibble at a time, and flags assigned bit by bit
// in an `always_comb`. In `always_ff` the bits not written keep their
// value; in `always_comb` every bit must be assigned before it is read
// (here `flags[3]` reads `flags[2:0]`) and by the end.
module nibreg (
  input  logic       clk,
  input  logic [1:0] we,
  input  logic [3:0] d,
  output logic [7:0] r,
  output logic [3:0] flags
);
  always_ff @(posedge clk) begin
    if (we[0]) r[3:0] <= d;
    if (we[1]) r[7:4] <= d;
  end

  always_comb begin
    flags[0] = r == 8'd0;
    flags[1] = r[7];
    flags[2] = ^r;
    flags[3] = |flags[2:0];
  end
endmodule
