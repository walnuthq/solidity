#!/usr/bin/env python3
"""Measure the complete Solidity-AST -> MLIR -> bytecode path.

Unlike corpus_coverage.py, this does not ask the legacy compiler for Yul. Each
source is passed directly to sol2evm and every RESULT line records the furthest
stage reached by one contract.
"""

import argparse
import collections
import concurrent.futures
import json
import pathlib
import re
import subprocess
import sys


RESULT = re.compile(
    r'^RESULT contract="([^"]+)" stage=([^ ]+) bytes=([^ ]+) detail="(.*)"$'
)


def compile_source(sol2evm, source, timeout):
    try:
        process = subprocess.run(
            [sol2evm, str(source), "--hex"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"source": str(source), "returncode": 124, "results": [], "detail": "timeout"}

    results = []
    for line in process.stdout.splitlines():
        match = RESULT.match(line)
        if match:
            results.append(
                {
                    "contract": match.group(1),
                    "stage": match.group(2),
                    "bytes": int(match.group(3)),
                    "detail": match.group(4),
                }
            )
    diagnostics = (process.stderr + process.stdout).strip().splitlines()
    return {
        "source": str(source),
        "returncode": process.returncode,
        "results": results,
        "detail": diagnostics[-1] if diagnostics else "no output",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, nargs="+", help="files or directories of .sol")
    parser.add_argument("--sol2evm", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--json")
    parser.add_argument("--require-all-bytecode", action="store_true")
    args = parser.parse_args()

    sources = []
    for entry in args.corpus:
        path = pathlib.Path(entry)
        sources.extend(sorted(path.rglob("*.sol")) if path.is_dir() else [path])

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(compile_source, args.sol2evm, source, args.timeout)
            for source in sources
        ]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())

    stages = collections.Counter()
    blockers = collections.Counter()
    no_result = collections.Counter()
    for record in records:
        if not record["results"]:
            no_result[record["detail"][:120]] += 1
        for result in record["results"]:
            stages[result["stage"]] += 1
            if result["stage"] != "bytecode":
                blockers[(result["stage"], result["detail"])] += 1

    contracts = sum(stages.values())
    bytecode = stages["bytecode"]
    share = 100.0 * bytecode / contracts if contracts else 0.0
    print(
        f"sources: {len(sources)}   reported contracts: {contracts}   "
        f"no-result sources: {sum(no_result.values())}"
    )
    print(f"bytecode: {bytecode}/{contracts} ({share:.2f}%)")
    print("furthest stage reached:")
    for stage, count in stages.most_common():
        print(f"  {stage:<9} {count:5d}")
    if blockers:
        print("why contracts stopped short:")
        for (stage, detail), count in blockers.most_common(20):
            print(f"  {count:4d}  {stage}: {detail}")
    if no_result:
        print("why sources produced no result:")
        for detail, count in no_result.most_common(10):
            print(f"  {count:4d}  {detail}")

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(
                {
                    "sources": len(sources),
                    "stages": dict(stages),
                    "blockers": [
                        {"stage": stage, "detail": detail, "count": count}
                        for (stage, detail), count in blockers.most_common()
                    ],
                    "records": sorted(records, key=lambda record: record["source"]),
                },
                indent=2,
            )
        )

    complete = bytecode == contracts and not no_result
    return 1 if args.require_all_bytecode and not complete else 0


if __name__ == "__main__":
    sys.exit(main())
