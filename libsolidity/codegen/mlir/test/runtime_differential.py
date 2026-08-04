#!/usr/bin/env python3
"""Stateful reference-vs-MLIR EVM runtime differential and performance probe."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import tempfile
import time
from typing import Any

import runtime_harness as harness


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[4]
U64_MAX = (1 << 64) - 1
U256_MAX = (1 << 256) - 1
RUNTIME_SEED_CASES = (
    (0, 0, 0, "0x"),
    (1, 1, 1, "0x00"),
    (7, 7, 8, "0x" + bytes(range(31)).hex()),
    (1023, 255, 256, "0x" + bytes(range(32)).hex()),
    (U256_MAX, U64_MAX, U64_MAX - 1, "0x" + bytes(range(33)).hex()),
    (0x45D9F3B, 0, U64_MAX, "0x" + bytes([0xFF] * 16).hex()),
    (0xDEADBEEF, U64_MAX, 0, "0x" + bytes([0x80] + [0] * 32).hex()),
    (0x123456789ABCDEF, 2**32 - 1, 2**32, "0x010305080d15"),
)


def workload_files(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.json"))


def workload_source(
    manifest: pathlib.Path,
    workload: dict[str, Any],
    repository_root: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    repository_source = workload.get("repository_source")
    if repository_source is not None:
        base_path = workload.get("repository_base_path")
        return (
            repository_root / repository_source,
            repository_root / base_path if base_path else repository_root,
        )
    return manifest.parent / workload["source"], repository_root


def expand_workload(workload: dict[str, Any]) -> dict[str, Any]:
    if workload.get("matrix") != "runtime-seeds-v1":
        return workload
    expanded = copy.deepcopy(workload)
    expanded.pop("matrix")
    steps = expanded.setdefault("steps", [])
    for index, (seed, left, right, data) in enumerate(RUNTIME_SEED_CASES):
        key = (left + right) & 7
        steps.extend(
            (
                {
                    "label": f"case-{index}-setup",
                    "kind": "tx",
                    "signature": "setup(uint256)",
                    "args": [str(seed)],
                },
                {
                    "label": f"case-{index}-run",
                    "kind": "tx",
                    "signature": "run(uint256,uint256,bytes)",
                    "args": [str(left), str(right), data],
                },
                {
                    "label": f"case-{index}-observe",
                    "kind": "call",
                    "signature": "observe(uint256)",
                    "args": [str(key)],
                },
            )
        )
    return expanded


def isolated_run(
    rpc: harness.Rpc,
    artifact: harness.Artifact,
    workload: dict[str, Any],
) -> dict[str, Any]:
    snapshot = rpc.snapshot()
    try:
        return harness.deploy_and_run(rpc, artifact, workload)
    finally:
        rpc.revert(snapshot)


def ratios(reference: dict[str, Any], mlir: dict[str, Any]) -> dict[str, Any]:
    def ratio(left: int | float | None, right: int | float | None) -> float | None:
        if left in (None, 0) or right is None:
            return None
        return round(float(right) / float(left), 4)

    result = {
        "compile_time": ratio(reference.get("compile_ms"), mlir.get("compile_ms")),
        "creation_size": ratio(reference.get("creation_bytes"), mlir.get("creation_bytes")),
        "runtime_size": ratio(reference.get("runtime_bytes"), mlir.get("runtime_bytes")),
        "deployment_gas": ratio(reference.get("deployment_gas"), mlir.get("deployment_gas")),
        "steps": [],
    }
    reference_steps = {entry["label"]: entry.get("gas") for entry in reference.get("steps", [])}
    for entry in mlir.get("steps", []):
        label = entry["label"]
        result["steps"].append(
            {"label": label, "gas": ratio(reference_steps.get(label), entry.get("gas"))}
        )
    return result


def execute_workload(
    rpc: harness.Rpc,
    solc: str,
    source: pathlib.Path,
    workload: dict[str, Any],
    evm_version: str,
    timeout: float,
    base_path: pathlib.Path,
) -> dict[str, Any]:
    contract = workload["contract"]
    reference_artifact = harness.compile_artifact(
        solc, source, contract, "reference", evm_version, timeout, base_path
    )
    mlir_artifact = harness.compile_artifact(
        solc, source, contract, "mlir", evm_version, timeout, base_path
    )
    if reference_artifact.selectors != mlir_artifact.selectors:
        raise harness.HarnessError(
            "reference and MLIR compiler output reported different function selectors\n"
            f"reference: {reference_artifact.selectors}\n"
            f"MLIR: {mlir_artifact.selectors}"
        )

    reference = isolated_run(rpc, reference_artifact, workload)
    mlir = isolated_run(rpc, mlir_artifact, workload)
    same = reference["semantics"] == mlir["semantics"]
    deployed = (
        reference["semantics"].get("deployment", {}).get("status") == "ok"
        and mlir["semantics"].get("deployment", {}).get("status") == "ok"
    )
    return {
        "name": workload.get("name", source.stem),
        "contract": contract,
        "source": str(source),
        "evm_version": evm_version,
        "status": "pass" if same and deployed else "fail",
        "reference": reference,
        "mlir": mlir,
        "ratios": ratios(reference["metrics"], mlir["metrics"]),
    }


def write_failure(
    directory: pathlib.Path,
    source: pathlib.Path,
    workload: dict[str, Any],
    result: dict[str, Any],
    evm_version: str,
) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"runtime-{source.stem}-{time.time_ns()}.json"
    payload = {
        "schema": "solc-mlir-runtime-failure-v1",
        "source_name": source.name,
        "source_text": source.read_text(),
        "workload": workload,
        "evm_version": evm_version,
        "result": result,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def replay(
    rpc: harness.Rpc,
    solc: str,
    failure: pathlib.Path,
    timeout: float,
) -> dict[str, Any]:
    payload = json.loads(failure.read_text())
    if payload.get("schema") != "solc-mlir-runtime-failure-v1":
        raise harness.HarnessError(f"unsupported replay schema in {failure}")
    with tempfile.TemporaryDirectory(prefix="solc-mlir-runtime-replay-") as temporary:
        source = pathlib.Path(temporary) / payload["source_name"]
        source.write_text(payload["source_text"])
        return execute_workload(
            rpc,
            solc,
            source,
            payload["workload"],
            payload["evm_version"],
            timeout,
            source.parent,
        )


def compact_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def report_result(result: dict[str, Any]) -> None:
    runtime_ratio = compact_ratio(result["ratios"]["runtime_size"])
    deploy_ratio = compact_ratio(result["ratios"]["deployment_gas"])
    print(
        f"{result['status'].upper():4} {result['name']} [{result['evm_version']}]: "
        f"runtime-size={runtime_ratio} deploy-gas={deploy_ratio}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--workloads", type=pathlib.Path)
    selection.add_argument("--replay", type=pathlib.Path)
    parser.add_argument("--solc", required=True)
    parser.add_argument("--rpc-url", default="http://127.0.0.1:8546")
    parser.add_argument("--spawn-anvil", action="store_true")
    parser.add_argument("--anvil")
    parser.add_argument("--evm-version", default="cancun")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--failure-dir", type=pathlib.Path, default=pathlib.Path("runtime-failures"))
    parser.add_argument("--report-json", type=pathlib.Path)
    parser.add_argument("--repository-root", type=pathlib.Path, default=REPOSITORY_ROOT)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    try:
        if args.replay:
            replay_payload = json.loads(args.replay.read_text())
            evm_version = replay_payload.get("evm_version", args.evm_version)
            with harness.runtime_rpc(
                args.rpc_url,
                args.spawn_anvil,
                args.anvil,
                evm_version,
                args.timeout,
            ) as rpc:
                results.append(replay(rpc, args.solc, args.replay, args.timeout))
                report_result(results[-1])
        else:
            manifests = workload_files(args.workloads)
            if not manifests:
                raise harness.HarnessError(f"no workload manifests under {args.workloads}")
            jobs_by_revision: dict[
                str, list[tuple[pathlib.Path, pathlib.Path, dict[str, Any]]]
            ] = {}
            for manifest in manifests:
                workload = expand_workload(json.loads(manifest.read_text()))
                source, base_path = workload_source(
                    manifest, workload, args.repository_root.resolve()
                )
                evm_version = workload.get("evm_version", args.evm_version)
                jobs_by_revision.setdefault(evm_version, []).append(
                    (source, base_path, workload)
                )

            if not args.spawn_anvil and len(jobs_by_revision) > 1:
                raise harness.HarnessError(
                    "mixed EVM revisions require --spawn-anvil so each revision "
                    "executes on a matching isolated node"
                )

            for evm_version, jobs in jobs_by_revision.items():
                with harness.runtime_rpc(
                    args.rpc_url,
                    args.spawn_anvil,
                    args.anvil,
                    evm_version,
                    args.timeout,
                ) as rpc:
                    for source, base_path, workload in jobs:
                        result = execute_workload(
                            rpc,
                            args.solc,
                            source,
                            workload,
                            evm_version,
                            args.timeout,
                            base_path,
                        )
                        results.append(result)
                        report_result(result)
                        if result["status"] != "pass":
                            failure = write_failure(
                                args.failure_dir,
                                source,
                                workload,
                                result,
                                evm_version,
                            )
                            print(f"     replay: {failure}")
    except harness.HarnessSkip as skip:
        print(f"runtime differential skipped: {skip}")
        return 77
    except Exception as error:
        print(error, file=sys.stderr)
        return 1

    report = {
        "schema": "solc-mlir-runtime-report-v1",
        "evm_version": args.evm_version,
        "evm_versions": sorted({result["evm_version"] for result in results}),
        "workloads": results,
        "summary": {
            "total": len(results),
            "passed": sum(result["status"] == "pass" for result in results),
            "failed": sum(result["status"] != "pass" for result in results),
        },
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], separators=(",", ":")))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
