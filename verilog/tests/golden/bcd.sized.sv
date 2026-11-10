module bcd (
  input  logic [7:0] a,
  input  logic signed [7:0] s,
  output logic [3:0] hundreds,
  output logic [3:0] tens,
  output logic [3:0] ones,
  output logic signed [7:0] q,
  output logic signed [7:0] r
);
  logic [7:0] h;
  logic [7:0] t;
  logic [7:0] o;
  assign h = (a / 8'd100);
  assign t = ((a / 8'd10) % 8'd10);
  assign o = (a % 8'd10);
  assign {hundreds, tens, ones} = {h[3:0], t[3:0], o[3:0]};
  assign q = (s / 8'sd253);
  assign r = (s % 8'sd3);
endmodule

