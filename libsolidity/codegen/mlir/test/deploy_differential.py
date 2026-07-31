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


def reference_contracts(solc, path):
    """Returns {contract name: (creation hex, [selectors])} from the reference build.

    solc emits one section per contract; pairing them by name matters, because
    a multi-contract file lists them in an order that need not match the IR
    output, and comparing one contract's code against another's selectors is
    what makes an unrelated contract look like a miscompile.
    """
    result = run([solc, "--via-ir", "--optimize", "--bin", "--hashes", str(path)])
    if result.returncode != 0:
        return {}

    contracts = {}
    name = None
    expecting_binary = False
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("======="):
            name = stripped.strip("= ").split(":")[-1]
            contracts.setdefault(name, ["", []])
            expecting_binary = False
        elif name is None:
            continue
        elif stripped == "Binary:":
            expecting_binary = True
        elif expecting_binary:
            if stripped and all(c in "0123456789abcdef" for c in stripped):
                contracts[name][0] = stripped
            expecting_binary = False
        else:
            match = re.match(r"^([0-9a-f]{8}):", stripped)
            if match:
                contracts[name][1].append(match.group(1))

    # A contract needing a library cannot be deployed as emitted, so there is
    # nothing meaningful to compare it against.
    return {
        contract: (code, selectors)
        for contract, (code, selectors) in contracts.items()
        if code and "__$" not in code
    }


def mlir_contracts(solc, yul2evm, path):
    """Returns {contract name: creation hex} for the same file via the ladder."""
    compiled = run([solc, "--ir-optimized", "--optimize", str(path)])
    if compiled.returncode != 0:
        return {}, "solc rejected the source"

    # One object tree per contract; an unindented `object` starts a tree and an
    # unindented brace closes it.
    trees, current = [], None
    for line in compiled.stdout.splitlines():
        if current is None:
            if line.startswith("object "):
                current = [line]
            continue
        current.append(line)
        if line.rstrip() == "}":
            trees.append("\n".join(current))
            current = None

    results, reason = {}, "did not reach bytecode"
    for tree in trees:
        with tempfile.NamedTemporaryFile("w", suffix=".yul", delete=False) as handle:
            handle.write(tree)
            yul_path = handle.name
        try:
            driver = run([yul2evm, yul_path, "--hex"])
        finally:
            pathlib.Path(yul_path).unlink(missing_ok=True)

        for line in driver.stdout.splitlines():
            if line.startswith("HEX "):
                # object="Name_42" -> Name; the driver quotes names doubly.
                label = re.search(r'object="+([^"]+)"+', line)
                if label:
                    results[label.group(1).rsplit("_", 1)[0]] = line.rsplit(" ", 1)[1]
                break
        else:
            for line in driver.stdout.splitlines():
                if line.startswith("RESULT ") and 'detail="' in line:
                    reason = line.split('detail="', 1)[1].rstrip('"')
                    break
    return results, reason


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
        reference = reference_contracts(args.solc, source)
        if not reference:
            skipped += 1
            continue
        mine, reason = mlir_contracts(args.solc, args.yul2evm, source)

        for name, (reference_code, selectors) in sorted(reference.items()):
            label = f"{source.name}:{name}"
            if name not in mine:
                print(f"UNSUP {label}: {reason}")
                skipped += 1
                continue

            mine_address = deploy(mine[name], rpc)
            reference_address = deploy(reference_code, rpc)
            if not reference_address:
                print(f"SKIP  {label}: reference creation code does not deploy here")
                skipped += 1
                continue
            if not mine_address:
                # Almost always the size of what this backend emits rather than
                # a semantic fault; say so instead of implying a miscompile.
                print(f"FAIL  {label}: creation code does not deploy "
                      f"({len(mine[name])//2}B creation vs {len(reference_code)//2}B reference)")
                failed += 1
                continue

            mine_code, _ = rpc.call("eth_getCode", [mine_address, "latest"])
            if not mine_code or len(mine_code) <= 2:
                print(f"FAIL  {label}: deployed empty runtime code")
                failed += 1
                continue

            divergent = []
            for selector in selectors:
                a = probe(mine_address, selector, rpc)
                b = probe(reference_address, selector, rpc)
                if a != b:
                    divergent.append((selector, a, b))

            if divergent:
                print(f"FAIL  {label}: {len(divergent)} of {len(selectors)} selector(s) diverge")
                for selector, a, b in divergent[:4]:
                    print(f"        {selector}: mlir={a[:50]} reference={b[:50]}")
                failed += 1
            else:
                print(f"PASS  {label}: deployed, {len(selectors)} selector(s) agree "
                      f"({(len(mine_code)-2)//2}B runtime)")
                passed += 1

    print(f"\n{passed} contracts deploy and agree, {failed} diverge, {skipped} not compiled")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
