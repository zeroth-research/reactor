module ramload (
  input  logic clk,
  input  logic we,
  input  logic [2:0] wa,
  input  logic [2:0] ra,
  input  logic [1:0] k,
  input  logic [7:0] d,
  output logic [7:0] q,
  output logic [7:0] p,
  output logic [7:0] r
);
  logic [7:0] ram [0:7];
  logic [7:0] vals [0:3];
  logic [7:0] rom [0:7];
  initial $readmemh("ramfull.hex", rom);
  initial vals[0] = 8'd10;
  initial vals[2] = 8'd30;
  initial vals[3] = 8'd40;
  initial ram[0] = 8'd17;
  initial ram[1] = 8'd34;
  initial ram[2] = 8'd51;
  initial ram[3] = 8'd68;
  initial ram[6] = 8'd102;
  initial ram[7] = 8'd119;
  always_ff @(posedge clk) begin if (we) ram[wa] <= d; else begin end q <= ram[ra]; end
  always_ff @(posedge clk) begin vals[k] <= (vals[k] + d); p <= vals[(~k)]; end
  assign r = (rom[ra] ^ ram[ra]);
endmodule

