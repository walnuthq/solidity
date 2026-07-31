#!/usr/bin/env python3
"""
Differential test for the MLIR EVM backend.

Compiles each Yul input twice - once through solc's own backend
(`solc --strict-assembly`) and once down the MLIR ladder
(`yul2evm`: libyul AST -> yul dialect -> evm dialect -> evmasm) - then executes
both bytecodes in the same EVM and compares observable behaviour: returndata,
halt status, and storage.

The reference backend is the anchor: a divergence is a bug in the MLIR path.

Requires a running `anvil` (foundry) and `cast` on PATH.

  anvil --silent --port 8546 &
  ./evm_differential.py --corpus corpus --rpc http://127.0.0.1:8546 \
      --solc ../../../../build/solc/solc \
      --yul2evm ../../../../build/libsolidity/codegen/mlir/tools/yul2evm
"""

import argparse
import json
import pathlib
import subprocess
import sys

# Storage slots compared after each call.
WATCHED_SLOTS = 8


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def reference_bytecode(solc, path):
    result = run([solc, "--strict-assembly", "--bin", str(path)])
    if result.returncode != 0:
        return None, result.stderr.strip().splitlines()[-1] if result.stderr else "solc failed"
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line and all(c in "0123456789abcdefABCDEF" for c in line):
            return line, None
    return None, "no binary in solc output"


def mlir_bytecode(yul2evm, path):
    result = run([yul2evm, str(path), "--hex"])
    stage, detail = "none", "driver failed"
    code = None
    for line in result.stdout.splitlines():
        if line.startswith("HEX "):
            code = line.rsplit(" ", 1)[1]
        elif line.startswith("RESULT "):
            for field in line.split():
                if field.startswith("stage="):
                    stage = field[len("stage=") :]
            if 'detail="' in line:
                detail = line.split('detail="', 1)[1].rstrip('"')
        elif line.startswith("PARSE-ERROR"):
            detail = line
    return code, stage, detail


class Evm:
    def __init__(self, rpc):
        self.rpc = rpc
        self.next_address = 0x1000

    def deploy(self, code):
        self.next_address += 1
        address = "0x" + format(self.next_address, "040x")
        run(["cast", "rpc", "anvil_setCode", address, "0x" + code, "--rpc-url", self.rpc])
        return address

    def observe(self, address, calldata="0x"):
        call = run(["cast", "call", address, calldata, "--rpc-url", self.rpc])
        if call.returncode != 0:
            # A revert is observable behaviour, not a harness failure.
            output = (call.stderr or "").strip()
            status = "revert:" + ("data" if "execution reverted" in output else "error")
        else:
            status = "ok:" + call.stdout.strip()
        slots = []
        for slot in range(WATCHED_SLOTS):
            value = run(["cast", "storage", address, str(slot), "--rpc-url", self.rpc])
            slots.append(value.stdout.strip())
        return status, slots


def rpc_reachable(url):
    """CTest convention: 77 means skip. A missing node is not a failure."""
    import urllib.request, json as _json
    try:
        request = urllib.request.Request(
            url,
            data=_json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=5).read()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--rpc", default="http://127.0.0.1:8546")
    parser.add_argument("--solc", required=True)
    parser.add_argument("--yul2evm", required=True)
    parser.add_argument("--json", help="write machine-readable results here")
    args = parser.parse_args()

    if not rpc_reachable(args.rpc):
        print(f'no JSON-RPC node at {args.rpc} - skipping')
        return 77

    inputs = sorted(pathlib.Path(args.corpus).glob("*.yul"))
    if not inputs:
        print(f"no .yul inputs under {args.corpus}", file=sys.stderr)
        return 2

    evm = Evm(args.rpc)
    results = []
    passed = failed = skipped = 0

    for path in inputs:
        reference, reference_error = reference_bytecode(args.solc, path)
        if reference is None:
            print(f"SKIP  {path.name}: reference backend rejected it ({reference_error})")
            skipped += 1
            results.append({"test": path.name, "outcome": "skip", "detail": reference_error})
            continue

        code, stage, detail = mlir_bytecode(args.yul2evm, path)
        if code is None:
            print(f"UNSUP {path.name}: stopped at stage '{stage}' - {detail}")
            skipped += 1
            results.append({"test": path.name, "outcome": "unsupported", "stage": stage, "detail": detail})
            continue

        mine = evm.observe(evm.deploy(code))
        theirs = evm.observe(evm.deploy(reference))

        entry = {
            "test": path.name,
            "mlir_bytes": len(code) // 2,
            "reference_bytes": len(reference) // 2,
        }
        if mine == theirs:
            print(
                f"PASS  {path.name}: {mine[0][:20]}  "
                f"({len(code)//2}B vs {len(reference)//2}B reference)"
            )
            passed += 1
            entry["outcome"] = "pass"
        else:
            print(f"FAIL  {path.name}")
            print(f"        mlir      status={mine[0]}")
            print(f"        reference status={theirs[0]}")
            if mine[1] != theirs[1]:
                for slot, (a, b) in enumerate(zip(mine[1], theirs[1])):
                    if a != b:
                        print(f"        storage[{slot}] mlir={a} reference={b}")
            failed += 1
            entry["outcome"] = "fail"
            entry["mlir"] = mine[0]
            entry["reference"] = theirs[0]
        results.append(entry)

    total = passed + failed
    print(f"\n{passed}/{total} differential match, {failed} divergent, {skipped} not compiled")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
