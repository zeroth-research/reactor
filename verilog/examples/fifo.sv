// A 4-entry FIFO in the shape of the lighthouse stream FIFOs: the memory
// is written in one block and read synchronously in another.
module fifo (
    input  logic       clk,
    input  logic       push,
    input  logic       pop,
    input  logic [7:0] d,
    output logic [7:0] q,
    output logic [1:0] wp,
    output logic [1:0] rp
);
  logic [7:0] ram[0:3];
  always_ff @(posedge clk) begin
    if (push) ram[wp] <= d;
  end
  always_ff @(posedge clk) begin
    if (push) wp <= wp + 2'd1;
    if (pop) rp <= rp + 2'd1;
  end
  always_ff @(posedge clk) q <= ram[rp];
endmodule
