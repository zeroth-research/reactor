// A register with synchronous reset, load and increment, and a flag set
// in the cycle after the register changed. `changed` gets a default that
// later assignments override: the last `<=` wins.
module enreg (
  input  logic       clk, rst, load, inc,
  input  logic [7:0] d,
  output logic [7:0] q,
  output logic       changed
);
  always_ff @(posedge clk) begin
    changed <= 1'b0;
    if (rst) begin
      q <= 8'd0;
      changed <= q != 8'd0;
    end else if (load) begin
      q <= d;
      changed <= q != d;
    end else if (inc) begin
      q <= q + 8'd1;
      changed <= 1'b1;
    end
  end
endmodule
