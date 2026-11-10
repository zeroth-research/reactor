module castparam (
  input  logic [7:0] u,
  input  logic [7:0] a,
  input  logic [7:0] b,
  output logic [15:0] wide,
  output logic [8:0] h,
  output logic [7:0] m,
  output logic [3:0] low,
  output logic [3:0] s,
  output logic [31:0] k
);
  assign wide = 16'((16'(unsigned'(u)) + 16'(unsigned'(u))));
  assign h = (9'(unsigned'(8'((a + b)))) >> 32'sd1);
  assign low = 4'((a + b));
  assign m = 8'd255;
  assign s = 4'd5;
  assign k = 32'sd3;
endmodule

