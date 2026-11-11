// Assignments to selects with a variable index in the shapes where Yosys
// deviates from the standard (see tests/targets.rs).
module dev (
  input  logic [3:0] n,
  input  logic [7:0] a,
  input  logic [2:0] e,
  output logic [7:0] z,
  output logic [7:0] r,
  output logic       f
);
  always_comb begin z = a; z[n -: 3] = e; end
  always_comb begin r = a; {f, r[n]} = {e[2], e[0]}; end
endmodule
