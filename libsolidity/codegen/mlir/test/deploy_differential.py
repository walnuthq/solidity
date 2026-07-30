#!/usr/bin/env python3
"""
End-to-end differential for the object model.

Compiles each contract's *creation* code twice - once with
`solc --via-ir --bin` and once down the MLIR ladder - deploys both, and then
compares the deployed contracts by calling every selector the ABI declares.

Creation code is only proven by running it: a wrong `datasize` or `dataoffset`
still assembles, and only shows up as a contract that deploys the wrong runtime
bytes. Comparing per-selector status and returndata catches that.

Requires a running `anvil`. Talks JSON-RPC directly rather than shelling out to
`cast`, because this backend's bytecode is large enough to overrun the argument
list.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

# anvil's first unlocked account.
SENDER = "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


class Rpc:
    def __init__(self, url):
        self.url = url
        self.identifier = 0

    def call(self, method, params):
        self.identifier += 1
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": self.identifier, "method": method, "params": params}
        ).encode()
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.load(response)
        except Exception as problem:
            # A payload the node refuses outright (oversized creation code)
            # closes the socket; report it rather than aborting the run.
            return None, {"message": str(problem)}
        return body.get("result"), body.get("error")


def reference_creation(solc, path):
    result = run([solc, "--via-ir", "--optimize", "--bin", str(path)])
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Binary:") and index + 1 < len(lines):
            code = lines[index + 1].strip()
            if code and all(c in "0123456789abcdef" for c in code):
                return code
    return None


def selectors(solc, path):
    result = run([solc, "--via-ir", "--hashes", str(path)])
    return sorted(set(re.findall(r"^([0-9a-f]{8}):", result.stdout, re.MULTILINE)))


def mlir_creation(solc, yul2evm, path):
    compiled = run([solc, "--ir-optimized", "--optimize", str(path)])
    if compiled.returncode != 0:
        return None, "solc rejected the source"
    # Take the first contract's object tree only. solc emits one per contract;
    # an unindented `object` starts the next one, and concatenating them is not
    # valid Yul.
    lines = compiled.stdout.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("object ")), None)
    if start is None:
        return None, "no Yul object"
    end = next(
        (i + 1 for i in range(start + 1, len(lines)) if lines[i].rstrip() == "}"), len(lines)
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yul", delete=False) as handle:
        handle.write("\n".join(lines[start:end]))
        yul_path = handle.name
    try:
        driver = run([yul2evm, yul_path, "--hex"])
    finally:
        pathlib.Path(yul_path).unlink(missing_ok=True)

    # The first HEX line is the root object, i.e. the creation code.
    for line in driver.stdout.splitlines():
        if line.startswith("HEX "):
            return line.rsplit(" ", 1)[1], None
    reason = "did not reach bytecode"
    for line in driver.stdout.splitlines():
        if line.startswith("RESULT ") and 'detail="' in line:
            reason = line.split('detail="', 1)[1].rstrip('"')
            break
    return None, reason


def deploy(code, rpc):
    transaction, error = rpc.call(
        "eth_sendTransaction", [{"from": SENDER, "data": "0x" + code, "gas": "0x2000000"}]
    )
    if error or not transaction:
        return None
    # Mining is asynchronous even in auto-mine mode, so a receipt that is not
    # there yet must not be read as a failed deployment.
    for _ in range(50):
        receipt, _ = rpc.call("eth_getTransactionReceipt", [transaction])
        if receipt:
            return receipt.get("contractAddress") if receipt.get("status") == "0x1" else None
        time.sleep(0.2)
    return None


def probe(address, selector, rpc):
    result, error = rpc.call(
        "eth_call", [{"from": SENDER, "to": address, "data": "0x" + selector}, "latest"]
    )
    return "revert" if error else "ok:" + (result or "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, nargs="+")
    parser.add_argument("--rpc", default="http://127.0.0.1:8546")
    parser.add_argument("--solc", required=True)
    parser.add_argument("--yul2evm", required=True)
    args = parser.parse_args()

    inputs = []
    for entry in args.corpus:
        path = pathlib.Path(entry)
        inputs.extend(sorted(path.rglob("*.sol")) if path.is_dir() else [path])

    rpc = Rpc(args.rpc)
    passed = failed = skipped = 0
    for source in inputs:
        reference = reference_creation(args.solc, source)
        if reference is None:
            skipped += 1
            continue
        mine, reason = mlir_creation(args.solc, args.yul2evm, source)
        if mine is None:
            print(f"UNSUP {source.name}: {reason}")
            skipped += 1
            continue

        mine_address = deploy(mine, rpc)
        reference_address = deploy(reference, rpc)
        if not reference_address:
            # The anchor could not be deployed either, so there is nothing to
            # compare against - not evidence about this backend.
            print(f"SKIP  {source.name}: reference creation code does not deploy here")
            skipped += 1
            continue
        if not mine_address:
            # Almost always the size of what this backend emits rather than a
            # semantic fault; say so instead of implying a miscompile.
            print(f"FAIL  {source.name}: creation code does not deploy "
                  f"({len(mine)//2}B creation vs {len(reference)//2}B reference)")
            failed += 1
            continue

        mine_code, _ = rpc.call("eth_getCode", [mine_address, "latest"])
        if not mine_code or len(mine_code) <= 2:
            print(f"FAIL  {source.name}: deployed empty runtime code")
            failed += 1
            continue

        divergent = []
        for selector in selectors(args.solc, source):
            a = probe(mine_address, selector, rpc)
            b = probe(reference_address, selector, rpc)
            if a != b:
                divergent.append((selector, a, b))

        if divergent:
            print(f"FAIL  {source.name}: {len(divergent)} selector(s) diverge")
            for selector, a, b in divergent[:4]:
                print(f"        {selector}: mlir={a[:40]} reference={b[:40]}")
            failed += 1
        else:
            probed = len(selectors(args.solc, source))
            print(f"PASS  {source.name}: deployed, {probed} selector(s) agree "
                  f"({(len(mine_code)-2)//2}B vs runtime reference)")
            passed += 1

    print(f"\n{passed} contracts deploy and agree, {failed} diverge, {skipped} not compiled")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
