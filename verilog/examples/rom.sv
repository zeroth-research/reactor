// Two ROMs loaded from files: a binary one read with a variable and a
// constant index, and a hexadecimal one read synchronously.
module rom (
    input  logic       clk,
    input  logic [2:0] i,
    input  logic [1:0] j,
    output logic [4:0] t,
    output logic [4:0] t2,
    output logic [1:0] tb,
    output logic [7:0] h
);
  logic [4:0] tab[0:7];
  logic [7:0] hex[4];
  initial begin
    $readmemb("rom.bin", tab);
  end
  initial $readmemh("rom.hex", hex);
  assign t = tab[i];
  assign t2 = tab[2];
  assign tb = tab[i][4:3];
  always_ff @(posedge clk) h <= hex[j] + 8'(tab[{1'b1, j}]);
endmodule
