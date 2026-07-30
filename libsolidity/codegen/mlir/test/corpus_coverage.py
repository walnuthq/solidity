#!/usr/bin/env python3
"""
Coverage instrument for the MLIR EVM backend over real contracts.

Compiles each Solidity input with `solc --ir-optimized`, then drives every Yul
object in the result down the ladder with `yul2evm` and records the furthest
stage each object reached:

    import -> promote -> evm -> asm -> bytecode

Prints a per-stage histogram and the distinct reasons objects stopped early, so
a regression shows up as objects moving to an earlier stage.

  ./corpus_coverage.py --solc ../../../../build/solc/solc \\
      --yul2evm ../../../../build/libsolidity/codegen/mlir/tools/yul2evm \\
      --corpus /path/to/contracts --json coverage.json
"""

import argparse
import collections
import json
import pathlib
import subprocess
import sys
import tempfile

STAGES = ["none", "import", "promote", "evm", "asm", "bytecode"]


def compile_to_yul(solc, path, timeout):
    try:
        result = subprocess.run(
            [solc, "--ir-optimized", "--optimize", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "solc timed out"
    if result.returncode != 0:
        message = (result.stderr or "").strip().splitlines()
        return None, message[-1] if message else "solc failed"
    # Strip the human-readable header preceding the first Yul object.
    lines = result.stdout.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("object "):
            return "\n".join(lines[index:]), None
    return None, "no Yul object in solc output"


def parse_results(stdout):
    objects = []
    for line in stdout.splitlines():
        if not line.startswith("RESULT "):
            continue
        fields = {}
        for token in line.split():
            if "=" in token and not token.startswith("detail"):
                key, _, value = token.partition("=")
                fields[key] = value.strip('"')
        detail = line.split('detail="', 1)[1].rstrip('"') if 'detail="' in line else ""
        objects.append(
            {
                "object": fields.get("object", "?"),
                "stage": fields.get("stage", "none"),
                "bytes": int(fields.get("bytes", 0)),
                "detail": detail,
            }
        )
    return objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, nargs="+", help="files or directories of .sol")
    parser.add_argument("--solc", required=True)
    parser.add_argument("--yul2evm", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json")
    args = parser.parse_args()

    inputs = []
    for entry in args.corpus:
        path = pathlib.Path(entry)
        inputs.extend(sorted(path.rglob("*.sol")) if path.is_dir() else [path])

    histogram = collections.Counter()
    reasons = collections.Counter()
    rejected = collections.Counter()
    records = []
    total_bytes = 0

    for source in inputs:
        yul, error = compile_to_yul(args.solc, source, args.timeout)
        if yul is None:
            rejected[error[:90]] += 1
            records.append({"source": str(source), "outcome": "solc-rejected", "detail": error})
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".yul", delete=False) as handle:
            handle.write(yul)
            yul_path = handle.name
        try:
            driver = subprocess.run(
                [args.yul2evm, yul_path], capture_output=True, text=True, timeout=args.timeout
            )
        except subprocess.TimeoutExpired:
            rejected["yul2evm timed out"] += 1
            continue
        finally:
            pathlib.Path(yul_path).unlink(missing_ok=True)

        objects = parse_results(driver.stdout)
        for record in objects:
            histogram[record["stage"]] += 1
            total_bytes += record["bytes"]
            if record["stage"] != "bytecode":
                reasons[record["detail"][:90]] += 1
        records.append({"source": str(source), "objects": objects})

    total = sum(histogram.values())
    print(f"\nsources: {len(inputs)}   yul objects: {total}   emitted bytes: {total_bytes}")
    if rejected:
        print(f"\nsolc rejected {sum(rejected.values())} sources:")
        for reason, count in rejected.most_common(10):
            print(f"  {count:4d}  {reason}")

    print("\nfurthest stage reached:")
    for stage in STAGES:
        count = histogram.get(stage, 0)
        if count:
            share = 100.0 * count / total if total else 0.0
            print(f"  {stage:<9} {count:5d}  {share:5.1f}%")

    if reasons:
        print("\nwhy objects stopped short:")
        for reason, count in reasons.most_common(15):
            print(f"  {count:4d}  {reason}")

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(
                {"histogram": dict(histogram), "reasons": dict(reasons), "records": records}, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
