module fa (
  input  logic a,
  input  logic b,
  input  logic cin,
  output logic s,
  output logic cout
);
  assign s = ((a ^ b) ^ cin);
  assign cout = ((a & b) | (cin & (a ^ b)));
endmodule

module ripple (
  input  logic [3:0] a,
  input  logic [3:0] b,
  input  logic cin,
  output logic [3:0] s,
  output logic cout
);
  logic s0;
  logic s1;
  logic s2;
  logic s3;
  logic c0;
  logic c1;
  logic c2;
  fa f0 (.a(a[0]), .b(b[0]), .cin(cin), .s(s0), .cout(c0));
  fa f1 (.a(a[1]), .b(b[1]), .cin(c0), .s(s1), .cout(c1));
  fa f2 (.a(a[2]), .b(b[2]), .cin(c1), .s(s2), .cout(c2));
  fa f3 (.a(a[3]), .b(b[3]), .cin(c2), .s(s3), .cout(cout));
  assign s = {s3, s2, s1, s0};
endmodule

