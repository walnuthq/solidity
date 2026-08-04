#!/usr/bin/env python3
"""
Differential for the sol rung: Solidity through the whole MLIR ladder against
the same source through solc, both runtime objects installed at an address
and every ABI selector called on each.

Only possible since SolToYul started emitting a dispatcher - before that the
ladder's output had no entry point to call. It earned its keep immediately,
catching a dispatcher that read past the end of calldata instead of reverting.

  anvil --silent --port 8546 &
  python3 sol_differential.py --corpus <dir-of-sol-files> --max-files 60
"""
import argparse
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import deploy_differential as D


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


def checked_run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def sources(path, max_files):
    source_path = pathlib.Path(path)
    candidates = [source_path] if source_path.is_file() else sorted(source_path.rglob("*.sol"))
    return candidates[:max_files] if max_files is not None else candidates


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus")
    source.add_argument("--source")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--solc", default="./build/solc/solc")
    parser.add_argument("--sol2evm", default="./build/libsolidity/codegen/mlir/tools/sol2evm")
    parser.add_argument("--rpc-url", default="http://127.0.0.1:8546")
    args = parser.parse_args()

    if not rpc_reachable(args.rpc_url):
        print(f"no JSON-RPC node at {args.rpc_url} - skipping")
        return 77

    rpc = D.Rpc(args.rpc_url)
    if not D.preflight(rpc):
        return 77

    ok = bad = skip = 0
    for src in sources(args.source or args.corpus, args.max_files):
        mine = {}
        for line in checked_run([args.sol2evm, str(src), "--hex"]).splitlines():
            match = re.match(r'HEX contract="([^"]+)" ([0-9a-f]+)', line)
            if match:
                mine[match.group(1).split(":")[-1]] = match.group(2)
        if not mine:
            continue

        reference_output = checked_run(
            [args.solc, "--via-ir", "--optimize", "--bin", "--hashes", str(src)]
        )
        # Per-contract creation code and selectors.
        current = None
        want_binary = False
        references = {}
        for line in reference_output.splitlines():
            stripped = line.strip()
            if stripped.startswith("======="):
                current = stripped.strip("= ").split(":")[-1]
                references.setdefault(current, ["", []])
                want_binary = False
            elif current is None:
                continue
            elif stripped == "Binary:":
                want_binary = True
            elif want_binary:
                if stripped and all(character in "0123456789abcdef" for character in stripped):
                    references[current][0] = stripped
                want_binary = False
            else:
                match = re.match(r"^([0-9a-f]{8}):", stripped)
                if match:
                    references[current][1].append(match.group(1))

        for name, code in mine.items():
            if name not in references or not references[name][0] or "__$" in references[name][0]:
                skip += 1
                continue
            reference_code, selectors = references[name]
            if not selectors:
                skip += 1
                continue

            # Same snapshot for each so both land at the same address and
            # neither sees the other's storage.
            snapshot, _ = rpc.call("evm_snapshot", [])
            reference_address = D.deploy(reference_code, rpc)
            reference_results = (
                [D.probe(reference_address, selector, rpc) for selector in selectors]
                if reference_address
                else None
            )
            rpc.call("evm_revert", [snapshot])
            mlir_address = D.deploy(code, rpc)
            if not reference_address:
                skip += 1
                continue
            if not mlir_address:
                print(f"FAIL {src.name}:{name}  creation code does not deploy")
                bad += 1
                continue

            differences = [
                (selector, D.probe(mlir_address, selector, rpc), reference_results[index])
                for index, selector in enumerate(selectors)
            ]
            differences = [difference for difference in differences if difference[1] != difference[2]]
            if differences:
                print(f"DIFF {src.name}:{name}  {len(differences)}/{len(selectors)}")
                for selector, mlir_result, reference_result in differences[:2]:
                    print(
                        f"      {selector}: mlir={mlir_result[:32]} "
                        f"ref={reference_result[:32]}"
                    )
                bad += 1
            else:
                print(f"OK   {src.name}:{name}  {len(selectors)} selector(s)")
                ok += 1

    print(f"\n{ok} agree, {bad} diverge, {skip} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print(exception, file=sys.stderr)
        sys.exit(1)
