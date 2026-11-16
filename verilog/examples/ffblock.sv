// Blocking `=` in clocked blocks where it means the same as `<=`: nothing
// that a `=` assigns is read later in its block (as in the lighthouse
// configurator's `config_wait_counter = config_wait_counter + 1;`). Reads
// before it, also in the same statement, see the value before the edge.
module ffblock(input logic clk, input logic en, input logic [1:0] s, input logic [7:0] d,
               output logic [7:0] cnt, output logic hit, output logic [7:0] w,
               output logic [3:0] m);
  // `hit` (with `<=`) reads `cnt` before the `=` that counts
  always_ff @(posedge clk) begin
    if (cnt == 8'd4) hit <= 1'b1;
    else hit <= 1'b0;
    if (en) cnt = cnt + 8'd1;
  end

  // part selects: `w[7:4]` is read after `w[3:0] =`, but those bits were
  // not assigned; a variable index assigns any bit of `m`
  always_ff @(negedge clk) begin
    w[3:0] = d[3:0];
    case (s)
      2'd0: w[7:4] = d[7:4] ^ w[7:4];
      2'd1: w[7:4] = w[7:4] + 4'd1;
      default: ;
    endcase
    m[s] = d[0];
  end
endmodule
