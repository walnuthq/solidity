#!/usr/bin/env python3
"""Prove the production MLIR backend is reached through the real solc CLI."""

import argparse
import pathlib
import re
import subprocess
import sys


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def solc_bytecode(output):
    bytecode = {}
    contract = None
    want_binary = False
    for line in output.splitlines():
        if line.startswith("======= "):
            contract = line.strip("= ")
            want_binary = False
        elif line.strip() == "Binary (MLIR pipeline):":
            want_binary = True
        elif want_binary and line.strip():
            if contract is None or not re.fullmatch(r"[0-9a-f]+", line.strip()):
                raise RuntimeError(f"malformed MLIR binary output:\n{output}")
            # solc may preserve a workspace-relative source-unit name while
            # sol2evm prints the resolved path. This test has one source, so
            # contract identity is the suffix after the final colon.
            bytecode[contract.rsplit(":", 1)[-1]] = line.strip()
            want_binary = False
    return bytecode


def sol2evm_bytecode(output):
    return {
        match.group(1).rsplit(":", 1)[-1]: match.group(2)
        for match in re.finditer(r'^HEX contract="([^"]+)" ([0-9a-f]+)$', output, re.MULTILINE)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solc", required=True)
    parser.add_argument("--sol2evm", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    source = str(pathlib.Path(args.source).resolve())

    # --mlir-bin owns its backend selection. --via-ir must not change its
    # result, and the standalone typed-sol route must remain independently
    # usable for legacy-codegen compatibility tests.
    direct = solc_bytecode(run([args.solc, "--mlir-bin", source]))
    via_ir = solc_bytecode(run([args.solc, "--via-ir", "--mlir-bin", source]))
    standalone = sol2evm_bytecode(run([args.sol2evm, source, "--hex"]))

    if not direct:
        raise RuntimeError("solc --mlir-bin emitted no bytecode")
    if direct != via_ir:
        raise RuntimeError("--via-ir changed the full MLIR pipeline output")
    if not standalone:
        raise RuntimeError("standalone legacy-compatibility MLIR route emitted no bytecode")
    if direct.keys() != standalone.keys():
        raise RuntimeError(
            "production and legacy-compatibility routes found different contracts\n"
            f"solc:    {sorted(direct)}\n"
            f"sol2evm: {sorted(standalone)}"
        )

    print(
        "solc reached the production Yul-IR -> MLIR backend and the typed-sol "
        f"compatibility route compiled {len(direct)} contract(s)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print(exception, file=sys.stderr)
        sys.exit(1)
