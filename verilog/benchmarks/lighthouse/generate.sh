#!/bin/bash
# Generates the Verilog of the Bitcraze Lighthouse deck FPGA (bitcraze/lighthouse-fpga, written
# in SpinalHDL) at a pinned commit, inside Bitcraze's own builder image (pinned by digest), and
# checks the result against expected.sha256.
#
# usage: generate.sh [OUT]       default OUT: ../sets/lighthouse (git-ignored)
#        generate.sh --check [OUT]   only check existing files against expected.sha256
# Needs git, docker and network (sbt downloads its dependencies inside the container).
#
# The generated files are third-party code (Bitcraze AB, GPL-3/LGPL-3, see README.md) and are
# not part of this repository. The build runs only inside the container: the repository is
# copied into it (no host directory is mounted) and only the generated files are copied back.
# OUT gets LighthouseTopLevel.v, the four ROM images it reads with $readmemb (next to it, since
# $readmemb names are relative to the source file) and the hand-written ts4231Configurator.v.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=6e5ed15aa012c28d66bea8f20b63b0092e89e186
IMAGE=bitcraze/fpga-builder@sha256:71f8f5ec942c3c2c6a636283aa4bf7539156007f05d816c5ba02814eb6286a71

check() {  # check OUT: every file of expected.sha256 present with that hash
  local bad=0 hash name actual
  while read -r hash name; do
    if [ ! -f "$1/$name" ]; then echo "missing: $name"; bad=1; continue; fi
    if [ "$name" = LighthouseTopLevel.v ]; then
      actual=$(sed 2d "$1/$name" | sha256sum | cut -d' ' -f1)   # line 2 = generation date
    else
      actual=$(sha256sum < "$1/$name" | cut -d' ' -f1)
    fi
    [ "$actual" = "$hash" ] || { echo "differs from expected.sha256: $name"; bad=1; }
  done < "$HERE/expected.sha256"
  [ $bad = 0 ] && echo "$1: all files as expected"
  return $bad
}

if [ "${1:-}" = --check ]; then check "$(realpath "${2:-$HERE/../sets/lighthouse}")"; exit; fi
OUT=$(realpath -m "${1:-$HERE/../sets/lighthouse}"); mkdir -p "$OUT"
WORK=$(mktemp -d); NAME=lhgen-$$
trap 'docker rm -f $NAME >/dev/null 2>&1; rm -rf "$WORK"' EXIT
git clone -q https://github.com/bitcraze/lighthouse-fpga.git "$WORK/src"
git -C "$WORK/src" -c advice.detachedHead=false checkout -q $COMMIT
docker create --name $NAME -w /work $IMAGE \
  bash -c 'cd /work/src && sbt -batch "runMain lighthouse.GenerateTopLevel"' >/dev/null
docker cp "$WORK/src" $NAME:/work/src
docker start -a $NAME | tail -6
docker cp $NAME:/work/src "$WORK/gen"
cp "$WORK/gen/LighthouseTopLevel.v" "$WORK"/gen/*.bin "$WORK/src/rtl/ts4231Configurator.v" "$OUT/"
check "$OUT"
