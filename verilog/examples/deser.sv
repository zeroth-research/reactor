// Serial to parallel, as in a UART receiver: on each enabled edge the
// input bit is stored at the position a counter gives.
module deser (
    input  logic       clk,
    input  logic       en,
    input  logic       d,
    output logic [7:0] q,
    output logic [2:0] cnt
);
  always_ff @(posedge clk) begin
    if (en) begin
      q[cnt] <= d;
      cnt <= cnt + 3'd1;
    end
  end
endmodule
