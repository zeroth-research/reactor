// Two registers that load, or swap their values: both right-hand sides
// are read before either register changes.
module swap (
  input  logic       clk, load, sw,
  input  logic [3:0] a, b,
  output logic [3:0] x, y
);
  always_ff @(posedge clk)
    if (load) begin
      x <= a;
      y <= b;
    end else if (sw) begin
      x <= y;
      y <= x;
    end
endmodule
