#!/usr/bin/env python3
"""Coverage of the benchmark sets by v2rm.

Usage (from anywhere; always run with -I, the sets are untrusted data):
    ./fetch.sh                                   # the sets of sets.conf into sets/
    python3 -I tools/measure.py [--sets verilog-eval,sv-tests,ebmc,sby] [--jobs 8] [--limit N]

For every selected case it runs
  raw:       v2rm on the file as is (several files concatenated, v2rm takes one),
  rewritten: v2rm on the file after `tools/rewrite.sed` (mechanical, meaning-preserving),
  yosys:     `v2rm --yosys` on the first variant that was accepted (raw, else rewritten).
Failures are classified by v2rm's first error, normalized (locations, names and numbers
stripped) and mapped to a coarse bucket (BUCKETS below).

Output in results/: <set>.tsv (one row per case), skipped-<set>.tsv (every file not
measured, with the reason), SUMMARY.md. Generated inputs go to work/.
Nothing from the clones is executed: files are only read, and sed/v2rm/yosys get paths.
"""

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS_DIR = ROOT / "sets"
TOOLS = ROOT / "tools"
WORK = ROOT / "work"
RESULTS = ROOT / "results"
V2RM = Path(os.environ.get("V2RM", Path.home() / ".cache/rm-target/release/v2rm"))
OSS = Path.home() / "tools/oss-cad-suite/bin"
TIMEOUT = 60
YOSYS_TIMEOUT = 300


class Case:
    def __init__(self, set_, id_, files, top=None):
        self.set, self.id, self.files, self.top = set_, id_, files, top


# ---------------------------------------------------------------- selection

def verilog_eval():
    d = SETS_DIR / "verilog-eval/dataset_spec-to-rtl"
    cases = [Case("verilog-eval", f.name[: -len("_ref.sv")], [f], "RefModule")
             for f in sorted(d.glob("*_ref.sv"))]
    skipped = [(str(f.relative_to(SETS_DIR)), "testbench (`*_test.sv`), not a design")
               for f in sorted(d.glob("*_test.sv"))]
    return cases, skipped


def metadata(text):
    meta = {}
    for m in re.finditer(r"^\s*:([a-z_]+):\s*(.*)$", text, re.M):
        meta.setdefault(m.group(1), m.group(2).strip())
    return meta


def sv_tests():
    d = SETS_DIR / "sv-tests/tests"
    cases, skipped = [], []
    for f in sorted(d.rglob("*.sv")):
        rel = str(f.relative_to(SETS_DIR))
        top_dir = f.relative_to(d).parts[0]
        if top_dir in ("uvm", "testbenches"):
            skipped.append((rel, f"`tests/{top_dir}/`: UVM / testbench infrastructure"))
            continue
        if top_dir == "generic":
            # files in tests/generic are no test cases with metadata of their own
            pass
        text = f.read_text(errors="replace")
        meta = metadata(text)
        if "should_fail_because" in meta:
            skipped.append((rel, "should-fail test (`:should_fail_because:`)"))
        elif meta.get("unsynthesizable") == "1":
            skipped.append((rel, "marked `:unsynthesizable: 1`"))
        elif "type" in meta and not re.search(r"elaboration|simulation|parsing", meta["type"]):
            skipped.append((rel, f"preprocessor-only test (`:type: {meta['type']}`)"))
        elif "defines" in meta:
            skipped.append((rel, "needs preprocessor defines (`:defines:`), v2rm has no `-D`"))
        elif not re.search(r"^\s*module\b", text, re.M):
            skipped.append((rel, "no `module` (package/class/interface/snippet only)"))
        else:
            cases.append(Case("sv-tests", str(f.relative_to(d)), [f], meta.get("top_module")))
    return cases, skipped


def ebmc():
    cases, skipped, seen = [], [], set()
    for sub in ("verilog", "ebmc"):
        for desc in sorted((SETS_DIR / "hw-cbmc/regression" / sub).rglob("*.desc")):
            rel = str(desc.relative_to(SETS_DIR))
            lines = desc.read_text(errors="replace").splitlines()
            if len(lines) < 3:
                skipped.append((rel, "malformed `.desc`"))
                continue
            mode, file, args = lines[0].split(), lines[1].strip(), lines[2].split()
            if mode[0] != "CORE":
                skipped.append((rel, f"test marked `{mode[0]}` (not CORE)"))
                continue
            f = desc.parent / file
            if f.suffix not in (".v", ".sv"):
                skipped.append((rel, f"input `{file}` is not Verilog"))
                continue
            if any(a in ("-I", "-D", "-y") or a.startswith(("-I", "-D")) for a in args):
                skipped.append((rel, "needs include dirs / defines / library dirs"))
                continue
            top = None
            for flag in ("--module", "--top"):
                if flag in args and args.index(flag) + 1 < len(args):
                    top = args[args.index(flag) + 1]
            key = (f, top)
            if key in seen:
                skipped.append((rel, "same file and top as an earlier test (other engine/bound)"))
                continue
            if not f.exists():
                skipped.append((rel, f"input `{file}` missing"))
                continue
            seen.add(key)
            id_ = str(f.relative_to(SETS_DIR / "hw-cbmc/regression")) + (f"#{top}" if top else "")
            cases.append(Case("ebmc", id_, [f], top))
    return cases, skipped


def sby():
    cases, skipped = [], []
    d = SETS_DIR / "sby/docs/examples"
    for job in sorted(d.rglob("*.sby")):
        rel = str(job.relative_to(SETS_DIR))
        section, files, top = None, [], None
        for line in job.read_text(errors="replace").splitlines():
            line = re.sub(r"^\s*[~\w ]+:\s+", "", line) if section != "files" else line
            s = line.strip()
            if re.fullmatch(r"\[\w+\]", s):
                section = s[1:-1]
            elif section == "files" and s:
                files.append(s.split()[-1])
            elif section == "script":
                m = re.search(r"\b(?:prep|hierarchy)\b.*-top\s+(\S+)", s)
                if m and not top:
                    top = m.group(1)
        srcs = [job.parent / f for f in files if Path(f).suffix in (".v", ".sv")]
        if not srcs:
            skipped.append((rel, "no Verilog files (VHDL or other)"))
        elif any(not f.exists() for f in srcs):
            skipped.append((rel, "a listed file is missing"))
        else:
            cases.append(Case("sby", str(job.relative_to(d)), srcs, top))
    return cases, skipped


SETS = {"verilog-eval": verilog_eval, "sv-tests": sv_tests, "ebmc": ebmc, "sby": sby}

# Not measured at all, documented in SUMMARY.md.
NOT_MEASURED = [
    ("riscv-formal", "case-study size (RISC-V cores checked against RVFI with SVA); "
     "future step, once the subset handles a whole core"),
    ("hw-cbmc/regression/{smv,vhdl,hw-cbmc,vlindex,rajdeep}", "not SystemVerilog designs "
     "for EBMC (other input languages, C co-verification, indexer tests)"),
]

# ---------------------------------------------------------------- running

def prepare(case, variant):
    """The file v2rm reads for `case` in `variant` ('raw' or 'rewritten')."""
    if variant == "raw" and len(case.files) == 1:
        return case.files[0]
    out = WORK / case.set / variant / (case.id.replace("/", "__") + ".sv")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f.read_text(errors="replace") + "\n" for f in case.files)
    if variant == "rewritten":
        text = subprocess.run(["sed", "-E", "-f", str(TOOLS / "rewrite.sed")], input=text,
                              capture_output=True, text=True, check=True).stdout
    out.write_text(text)
    return out


def run(args, timeout):
    env = dict(os.environ)
    env["PATH"] = f"{OSS}:{env['PATH']}"
    try:
        p = subprocess.run([str(V2RM), *args], capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=WORK)
    except subprocess.TimeoutExpired:
        return None
    return p


def accept(case, variant):
    path = prepare(case, variant)
    top = ["--top", case.top] if case.top else []
    p = run([*top, str(path)], TIMEOUT)
    if p is None:
        return path, "TIMEOUT"
    if p.returncode == 0:
        return path, "OK"
    first = next((l for l in p.stderr.splitlines() if l.strip()), "(no message)")
    if p.returncode == 101 or "panicked" in p.stderr:
        panic = next((l for l in p.stderr.splitlines() if "panicked" in l), first)
        return path, "PANIC " + panic
    return path, where(path, first.replace(str(path), "", 1).lstrip(":"))


def where(path, msg):
    """Adds the keyword at the error's location to generic messages, which name only an
    sv-parser node kind (`unsupported: ModuleCommonItem`) or nothing (`syntax error`)."""
    m = re.match(r"(\d+):(\d+): (unsupported: \w+|syntax error|`[^`]*` not supported)$", msg)
    if not m:
        return msg
    try:
        line = path.read_text(errors="replace").splitlines()[int(m.group(1)) - 1]
    except (IndexError, OSError):
        return msg
    # the column of a generic message often points into the item (a label, an operand):
    # take the first keyword of the line instead, after a statement label `p0:`
    w = re.match(r"\s*(?:\w+\s*:(?!:)\s*)?([`$#]?\w+)", line)
    if not w:
        return msg
    word = w.group(1)
    if word == "always" and re.match(r"\s*(assert|assume|cover)\b", line[w.end():]):
        word = "assert"  # EBMC's `always assert p: e` (not standard SystemVerilog)
    return f"{msg} [at `{word}`]"


def yosys(case, path):
    top = ["--top", case.top] if case.top else []
    p = run(["--yosys", *top, str(path)], YOSYS_TIMEOUT)
    if p is None:
        return "TIMEOUT"
    if p.returncode == 0:
        return "Equal"
    out = p.stdout.splitlines()
    if any(l.startswith("differs:") for l in out):
        return "Differs " + " ".join(l.split("`")[1] for l in out if l.startswith("differs:"))
    if any(l.startswith("unknown:") for l in out):
        return "Unknown"
    first = next((l for l in p.stderr.splitlines()
                  if l.strip() and ": warning: " not in l), "(no message)")
    first = first.replace(str(path), "", 1).lstrip(": ")
    return "Error " + first.replace(str(SETS_DIR) + "/", "").replace(str(ROOT) + "/", "")


def measure(case):
    raw_path, raw = accept(case, "raw")
    rw_path, rw = accept(case, "rewritten")
    target = raw_path if raw == "OK" else rw_path if rw == "OK" else None
    ys = yosys(case, target) if target else "-"
    return case, raw, rw, ys


# ---------------------------------------------------------------- classification

def normalize(msg):
    msg = re.sub(r"^\d+:\d+: ", "", msg)
    msg = re.sub(r" \(in instance `[^`]*`\)$", "", msg)
    msg = re.sub(r"^type `(\w+)[^`]*`", r"type `\1`", msg)  # keep the type's keyword
    msg = re.sub(r"(?<!\[at )`[^`]*`(?! not supported \[at)", "`_`", msg)
    return re.sub(r"\d+", "N", msg)


AT = r"\[at `({})`\]"
BUCKETS = [  # first match wins; pattern on the first error without its location
    (r"^PANIC", "internal error (panic): a bug in v2rm"),
    (r"^TIMEOUT", "timeout"),
    (r"`reg` not supported|plain `always`|" + AT.format("always"), "spelling: `reg` / plain `always`"),
    (r"unsized literal|resize explicitly|differs from selector|bits to \d+ bits|signedness",
     "width rules (explicit-width fragment)"),
    (r"Define not found|Include not found|Include error", "preprocessor: define / include missing"),
    (r"x/z digits", "x/z values (4-state)"),
    (r"initializer|initial value|may only load arrays from files|" + AT.format("initial|final"),
     "initializers / `initial` blocks"),
    (AT.format(r"assert|assume|cover|restrict|property|sequence|expect|default|\$past|"
               r"\$rose|\$fell|\$stable|clocking|always_ff_assert|checker"), "assertions / properties"),
    (r"unpacked dimension|instance array|more than one packed dimension|array range|is an array|"
     r"address beyond the last element|associative and queue|\$readmem|loaded array", "arrays"),
    (r"non-ANSI|positional (port|parameter)|port default", "port / parameter syntax (non-ANSI, positional)"),
    (r"variable index in an assignment target", "variable-index assignment targets"),
    (r"crossings between unrelated clocks", "clock-domain crossings"),
    (AT.format("function|task|return"), "functions / tasks"),
    (r"generate regions|" + AT.format("for|generate|genvar|while|repeat|foreach|if|case"),
     "generate / loops"),
    (r"^type `(int|integer|byte|shortint|longint|real|shortreal|realtime|time|string|chandle|event|bit|logic)`"
     r"|net type|" + AT.format("integer|int|real|time|string|event|byte|shortint|longint|tri\w*|wand|wor|supply\d|"
                               r"uwire|trireg|var"), "other data/net types (`int`, `integer`, `real`, nets…)"),
    (r"^type `(struct|union|enum)|type parameters|parameter type|" +
     AT.format(r"typedef|struct|union|enum|package|import|class|interface|modport|program|config|"
               r"primitive|`\w+|bind|let|nettype|parameter|localparam|virtual"),
     "types, packages, interfaces, classes (`typedef`, `struct`, …)"),
    (r"delay|" + AT.format(r"\$\w+|#"), "simulation constructs (delays, system tasks)"),
    (AT.format(r"and|or|nand|nor|xor|xnor|not|buf|bufif\d|notif\d|[a-z]*mos|[a-z]*tran\w*"),
     "gate primitives (`and`, `nor`, …)"),
    (r"range \[\d+:\d+\] not supported", "ranges other than `[N-1:0]`"),
    (r"several top|no module", "top-module selection"),
    (r"latch|read before|never driven|undriven|driven by|drives nothing|combinational loop",
     "design rejected by a checking rule (latch, loop, drivers)"),
    (r"syntax error", "parse error (sv-parser)"),
    (r"^unsupported: ", "other unsupported syntax"),
    (r"not supported", "other unsupported construct"),
]


def bucket(msg):
    msg = re.sub(r"^\d+:\d+: ", "", msg)
    for pat, name in BUCKETS:
        if re.search(pat, msg):
            return name
    return "other error"


# ---------------------------------------------------------------- report

def commit(repo):
    p = subprocess.run(["git", "-C", str(SETS_DIR / repo), "log", "-1", "--format=%H %ci"],
                       capture_output=True, text=True)
    return p.stdout.strip()


def table(counter, total, head):
    rows = [f"| {head} | count | % |", "|---|---:|---:|"]
    for k, n in counter.most_common():
        rows.append(f"| {k} | {n} | {100 * n / total:.0f} |")
    return "\n".join(rows)


def report(per_set, skipped_per_set):
    v = subprocess.run(["git", "-C", str(ROOT), "log", "-1",
                        "--format=%h %ci"], capture_output=True, text=True).stdout.strip()[:19]
    repos = {"verilog-eval": "verilog-eval", "sv-tests": "sv-tests", "ebmc": "hw-cbmc", "sby": "sby"}
    out = ["# Benchmark coverage of v2rm", "",
           f"Generated by `tools/measure.py`; v2rm at commit `{v}` of this repository.",
           "**raw** = file as is; **rewritten** = after `tools/rewrite.sed`; **Yosys** = "
           "`v2rm --yosys` on the first accepted variant. Failures are classified by v2rm's "
           "*first* error only, so a bucket counts the blocker that is hit first.", "",
           "| set | commit | cases | skipped | raw OK | rewritten OK | Yosys Equal | Differs | Yosys error/other |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, rows in per_set.items():
        n = len(rows)
        raw = sum(r[1] == "OK" for r in rows)
        rw = sum(r[2] == "OK" for r in rows)
        eq = sum(r[3] == "Equal" for r in rows)
        dif = sum(r[3].startswith("Differs") for r in rows)
        oth = sum(r[3] not in ("-", "Equal") and not r[3].startswith("Differs") for r in rows)
        out.append(f"| {s} | `{commit(repos[s])[:12]}` | {n} | {len(skipped_per_set[s])} | "
                   f"{raw} ({100 * raw / max(n, 1):.0f}%) | {rw} ({100 * rw / max(n, 1):.0f}%) | "
                   f"{eq} | {dif} | {oth} |")
    for s, rows in per_set.items():
        n = len(rows)
        if not n:
            continue
        out += ["", f"## {s}", ""]
        for col, name in ((1, "raw"), (2, "rewritten")):
            fails = [r[col] for r in rows if r[col] != "OK"]
            out += [f"### First error, {name} ({len(fails)} of {n} fail)", "",
                    table(Counter(bucket(m) for m in fails), n, "bucket"), ""]
        fails = [r[2] for r in rows if r[2] != "OK"]
        out += ["### Most frequent first errors, rewritten (normalized, top 25)", "",
                table(Counter(normalize(m) for m in fails[:]), n, "message"), ""]
        out[-2] = "\n".join(out[-2].splitlines()[:27])
        bad = [r for r in rows if r[3] not in ("-", "Equal")]
        if bad:
            out += ["### Yosys check: not Equal", "", "| case | result |", "|---|---|"]
            out += [f"| {r[0].id} | {r[3][:200]} |" for r in bad] + [""]
        out += [f"### Skipped files ({len(skipped_per_set[s])}, list in `skipped-{s}.tsv`)", "",
                table(Counter(re.sub(r"`[^`]*`", "`…`", why) if "input" in why else why
                              for _, why in skipped_per_set[s]),
                      max(len(skipped_per_set[s]), 1), "reason"), ""]
    out += ["## Not measured", "", "| set | why |", "|---|---|"]
    out += [f"| {a} | {b} |" for a, b in NOT_MEASURED]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="at most N cases per set (testing)")
    a = ap.parse_args()
    if not SETS_DIR.exists():
        sys.exit(f"{SETS_DIR} missing: run {ROOT / 'fetch.sh'} first")
    if not V2RM.exists():
        sys.exit(f"{V2RM} missing: build with CARGO_TARGET_DIR=~/.cache/rm-target "
                 "cargo build --release -p verilog --bin v2rm")
    WORK.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    per_set, skipped_per_set = {}, {}
    for s in a.sets.split(","):
        cases, skipped = SETS[s]()
        if a.limit:
            cases = cases[: a.limit]
        print(f"{s}: {len(cases)} cases, {len(skipped)} skipped", file=sys.stderr, flush=True)
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(measure, cases))
        rows = [(c, raw, rw, ys) for c, raw, rw, ys in rows]
        per_set[s], skipped_per_set[s] = rows, skipped
        with open(RESULTS / f"{s}.tsv", "w") as f:
            f.write("case\ttop\traw\trewritten\tyosys\tbucket\n")
            for c, raw, rw, ys in rows:
                b = "" if rw == "OK" else bucket(rw)
                f.write(f"{c.id}\t{c.top or ''}\t{raw}\t{rw}\t{ys}\t{b}\n")
        with open(RESULTS / f"skipped-{s}.tsv", "w") as f:
            f.write("file\treason\n")
            f.writelines(f"{p}\t{why}\n" for p, why in skipped)
    (RESULTS / "SUMMARY.md").write_text(report(per_set, skipped_per_set))
    print((RESULTS / "SUMMARY.md").read_text().split("\n## ")[0])


if __name__ == "__main__":
    main()
