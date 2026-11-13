module powerup (
  input  logic clk,
  input  logic en,
  input  logic [3:0] i,
  input  logic [7:0] d,
  output logic [3:0] st,
  output logic [7:0] y,
  output logic b,
  output logic [3:0] w
);
  logic [7:0] cnt;
  logic [7:0] last;
  logic [7:0] seen;
  logic armed;
  logic [11:0] wide;
  logic [15:0] pattern;
  logic [7:0] next;
  initial st = 4'd9;
  initial cnt = 8'd250;
  initial wide = 12'sd4093;
  initial armed = 1'sd1;
  assign next = (cnt + 8'd3);
  always_ff @(posedge clk) begin if (en) cnt <= next; else begin end last <= d; seen <= (seen | d); if ((armed && (cnt == 8'd255))) armed <= 1'd0; else begin end st <= (st - 4'd1); wide <= (wide + 12'(unsigned'(d))); end
  assign y = ((cnt ^ last) ^ seen);
  assign b = (pattern[i] & armed);
  assign w = wide[11:8];
  assign pattern = {15'd14635, 1'd0};
endmodule

