module gray (
  input  logic [7:0] b,
  output logic [7:0] g,
  output logic [7:0] back
);
  logic [7:0] t1;
  logic [7:0] t2;
  assign g = (b ^ (b >> 3'd1));
  assign t1 = (g ^ (g >> 3'd1));
  assign t2 = (t1 ^ (t1 >> 3'd2));
  assign back = (t2 ^ (t2 >> 3'd4));
endmodule

