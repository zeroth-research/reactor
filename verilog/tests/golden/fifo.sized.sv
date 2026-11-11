module fifo (
  input  logic clk,
  input  logic push,
  input  logic pop,
  input  logic [7:0] d,
  output logic [7:0] q,
  output logic [1:0] wp,
  output logic [1:0] rp
);
  logic [7:0] ram [0:3];
  always_ff @(posedge clk) begin if (push) ram[wp] <= d; else begin end end
  always_ff @(posedge clk) begin if (push) wp <= (wp + 2'd1); else begin end if (pop) rp <= (rp + 2'd1); else begin end end
  always_ff @(posedge clk) q <= ram[rp];
endmodule

