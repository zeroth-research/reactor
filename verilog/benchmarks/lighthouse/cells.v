// Blackbox stubs of the vendor cells of LighthouseTopLevel, for v2rm and Yosys.
//
// A module marked (* blackbox *) declares only ports and parameters. v2rm treats every
// output of a blackbox as a free value in every time step (like an input of the top
// module) and reads none of its inputs; Yosys does the same with `cutpoint -blackbox`.
// Only the ports the design connects are declared (v2rm requires every input of an
// instance to be connected). Parameter values given by the design are accepted by name
// and have no effect.
//
// - pll: the wrapper around SB_PLL40_PAD (12 MHz in, 48 MHz out). Its output clock_out
//   is Core_clk, which thus becomes an input clock of its own (the 4:1 frequency
//   relation to io_clk12MHz is not modelled). locked is not read by the design.
// - SB_IO: the iCE40 I/O cell (PIN_TYPE 6'b110000 in the design: registered DDR input,
//   registered DDR output with a registered output enable, tristate pin). D_IN_0 and
//   D_IN_1 (the pin sampled at the rising and the falling edge of INPUT_CLK) are free;
//   PACKAGE_PIN is the top-level pin, which is no signal of the translation.
// - SB_WARMBOOT: reboots the FPGA into another image; inputs only.

(* blackbox *)
module pll (
  input  clock_in,
  output clock_out,
  output locked
);
endmodule

(* blackbox *)
module SB_IO #(
  parameter [5:0] PIN_TYPE = 6'b000000
) (
  inout  PACKAGE_PIN,
  input  CLOCK_ENABLE,
  input  INPUT_CLK,
  input  OUTPUT_CLK,
  input  OUTPUT_ENABLE,
  input  D_OUT_0,
  input  D_OUT_1,
  output D_IN_0,
  output D_IN_1
);
endmodule

(* blackbox *)
module SB_WARMBOOT (
  input BOOT,
  input S1,
  input S0
);
endmodule
