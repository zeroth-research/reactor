// Writes with variable indices that may reach outside the target: the
// bits outside are not written.
module oorwr (
    input  logic       clk,
    input  logic [3:0] i,
    input  logic [2:0] e,
    input  logic       d,
    output logic [7:0] p,
    output logic [7:0] q,
    output logic [7:0] r,
    output logic       f
);
  always_ff @(posedge clk) begin
    q[i] <= d;                  // nothing for i > 7
    p[i +: 3] <= e;             // only the bits up to 7
    r[{1'b0, i} + 5'd2 -: 3] <= e; // index 2 to 17: only the bits up to 7
    f <= d;
    q[i - 1] <= ~d;             // a 32-bit index: nothing for i = 0
    if (e[0]) q[i[2:0]] <= e[1]; // the last write wins
  end
endmodule
