module unsized (
  input  logic [3:0] q,
  input  logic [7:0] x,
  input  logic c,
  output logic [3:0] inc,
  output logic wrapped,
  output logic wrapped4,
  output logic full,
  output logic [7:0] ones,
  output logic [7:0] zeros,
  output logic [7:0] pick,
  output logic [7:0] based,
  output logic [39:0] big
);
  assign inc = 4'((32'(unsigned'(q)) + 32'd1));
  assign wrapped = ((32'(unsigned'(q)) + 32'd1) == 32'd0);
  assign wrapped4 = ((q + 4'd1) == 4'd0);
  assign ones = 8'd255;
  assign zeros = 8'd0;
  assign pick = (c ? x : 8'd255);
  assign full = (x == 8'd255);
  assign based = 8'((32'(unsigned'(x)) + 32'd3));
  assign big = 40'd4294967296;
endmodule

