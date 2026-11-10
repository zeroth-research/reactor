module prio (
  input  logic [7:0] req,
  output logic [2:0] idx,
  output logic valid
);
  always_comb begin idx = 3'd0; valid = (|req); if ((req >= 8'd128)) idx = 3'd7; else if ((req >= 8'd64)) idx = 3'd6; else if ((req >= 8'd32)) idx = 3'd5; else if ((req >= 8'd16)) idx = 3'd4; else if ((req >= 8'd8)) idx = 3'd3; else if ((req >= 8'd4)) idx = 3'd2; else if ((req >= 8'd2)) idx = 3'd1; else begin end end
endmodule

