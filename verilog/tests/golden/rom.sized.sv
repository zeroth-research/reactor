module rom (
  input  logic clk,
  input  logic [2:0] i,
  input  logic [1:0] j,
  output logic [4:0] t,
  output logic [4:0] t2,
  output logic [1:0] tb,
  output logic [7:0] h
);
  logic [4:0] tab [0:7];
  initial $readmemb("rom.bin", tab);
  logic [7:0] hex [0:3];
  initial $readmemh("rom.hex", hex);
  assign t = tab[i];
  assign t2 = tab[2];
  assign tb = tab[i][4:3];
  always_ff @(posedge clk) h <= (hex[j] + 8'(8'(unsigned'(tab[{1'd1, j}]))));
endmodule

