// An adder with carry in and carry out, written with a concatenation as
// the target: the sum is evaluated at the 9 bits of `{cout, sum}`, and
// the 1-bit `cin` is extended to them. And a decade counter with unsized
// literals: `cnt + 1` is 32 bits wide and truncated to the 4 bits of
// `cnt`.
module carry (
  input  logic       clk, cin,
  input  logic [7:0] a, b,
  output logic [7:0] sum,
  output logic       cout,
  output logic [3:0] cnt
);
  assign {cout, sum} = a + b + cin;
  always_ff @(posedge clk)
    if (cnt == 9) cnt <= 0;
    else cnt <= cnt + 1;
endmodule
