module barrel (
  input  logic [1:0] op,
  input  logic [3:0] n,
  input  logic [7:0] a,
  input  logic signed [7:0] s,
  output logic [7:0] y,
  output logic signed [7:0] z
);
  always_comb begin case (op) 2'd0: y = (a << n); 2'd1: y = (a >> n); 2'd2: y = (a <<< n); default: y = (a >>> n); endcase end
  assign z = (s >>> n);
endmodule

