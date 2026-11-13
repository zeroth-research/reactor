// A RAM preloaded from a file that leaves addresses 4 and 5 without values
// (they start with any value), a RAM whose elements get initial values one
// by one, and a ROM loaded from the same file for comparison.
module ramload (
  input  logic       clk,
  input  logic       we,
  input  logic [2:0] wa, ra,
  input  logic [1:0] k,
  input  logic [7:0] d,
  output logic [7:0] q, p, r
);
  logic [7:0] ram [0:7];
  logic [7:0] vals [4];
  logic [7:0] rom [0:7];
  initial $readmemh("ramload.hex", ram);
  initial begin
    vals[0] = 8'd10;
    vals[2] = 8'd30;
    vals[3] = 8'd40;
  end
  always_ff @(posedge clk) begin
    if (we) ram[wa] <= d;
    q <= ram[ra];
  end
  always_ff @(posedge clk) begin
    vals[k] <= vals[k] + d;
    p <= vals[~k];
  end
  initial $readmemh("ramfull.hex", rom);
  assign r = rom[ra] ^ ram[ra];
endmodule
