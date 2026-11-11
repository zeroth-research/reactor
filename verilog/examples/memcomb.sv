// Arrays driven combinationally: a decoder written element by element
// (every element first, then the one picked by the index), and an array
// driven by one `assign` of a concatenation, read with a variable index.
module memcomb (
    input  logic [1:0] sel,
    input  logic [1:0] k,
    input  logic [3:0] d,
    input  logic [3:0] a,
    input  logic [3:0] c,
    output logic [15:0] y,
    output logic [3:0] z,
    output logic [3:0] w
);
  logic [3:0] t[4];
  logic [3:0] u[0:1];
  always_comb begin
    t[0] = '0;
    t[1] = '0;
    t[2] = a;
    t[3] = '0;
    t[sel] = d;
    z = t[k] ^ t[2];
  end
  assign y = {t[3], t[2], t[1], t[0]};
  assign {u[1], u[0]} = {c, a};
  assign w = u[k[0]];
endmodule
