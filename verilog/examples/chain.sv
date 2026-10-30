// Three `always_comb` blocks written in the reverse order of their data
// flow a -> x -> z -> y; the translation sorts them. Merging the first and
// the last block into one would be rejected as a combinational loop, since
// a block is translated as one atom.
module chain (
  input  logic [7:0] a,
  output logic [7:0] y
);
  logic [7:0] x, z;

  always_comb y = z ^ 8'h0f;
  always_comb z = x + x;
  always_comb x = a - 8'd3;
endmodule
