module rotate (
  input  logic [7:0] a,
  input  logic [2:0] n,
  output logic [7:0] rr,
  output logic [7:0] rl
);
  logic [15:0] t;
  assign rr = 8'(({a, a} >> n));
  assign t = ({a, a} << n);
  assign rl = t[15:8];
endmodule

