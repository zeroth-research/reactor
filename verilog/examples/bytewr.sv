// A byte-lane write into a 32-bit register.
module bytewr (
    input  logic        clk,
    input  logic        we,
    input  logic [ 1:0] lane,
    input  logic [ 7:0] d,
    output logic [31:0] w
);
  always_ff @(posedge clk) if (we) w[{lane, 3'b000} +: 8] <= d;
endmodule
