#!/usr/bin/env python3
"""Reusable compiler, ABI, and JSON-RPC support for MLIR runtime differentials."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Iterator


DEFAULT_SENDER = "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"
DEFAULT_ACCOUNTS = (
    DEFAULT_SENDER,
    "0x70997970c51812dc3a010c7d01b50e0d17dc79c8",
    "0x3c44cdddb6a900fa2b585dd299e03d12fa4293bc",
    "0x90f79bf6eb2c4f870365e785982e1f101e93b906",
    "0x15d34aaf54267db7d7c367839aaf71a00a2c6a65",
)
DEFAULT_GAS = 0x20000000


class HarnessError(RuntimeError):
    pass


class HarnessSkip(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Artifact:
    creation: str
    selectors: dict[str, str]
    compile_ms: float


def compile_artifact(
    solc: str,
    source: pathlib.Path,
    contract: str,
    backend: str,
    evm_version: str,
    timeout: float,
    base_path: pathlib.Path | None = None,
) -> Artifact:
    common = [
        solc,
        "--evm-version",
        evm_version,
        "--optimize",
        "--no-cbor-metadata",
    ]
    if base_path is not None:
        common.extend(("--base-path", str(base_path)))
    if backend == "reference":
        command = [*common, "--via-ir", "--bin", "--hashes", str(source)]
        binary_heading = "Binary:"
    elif backend == "mlir":
        command = [*common, "--mlir-bin", "--hashes", str(source)]
        binary_heading = "Binary (MLIR pipeline):"
    else:
        raise ValueError(f"unknown backend: {backend}")

    started = time.perf_counter()
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    elapsed = (time.perf_counter() - started) * 1000
    if result.returncode != 0:
        raise HarnessError(
            f"{backend} compilation failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    contracts = _parse_solc_output(result.stdout, binary_heading)
    if contract not in contracts:
        raise HarnessError(
            f"{backend} output has no contract {contract}; found {sorted(contracts)}"
        )
    entry = contracts[contract]
    creation = entry["creation"]
    if not creation:
        raise HarnessError(f"{backend} output contains no creation bytecode for {contract}")
    selectors = {
        signature: selector
        for candidate in contracts.values()
        for signature, selector in candidate["selectors"].items()
    }
    return Artifact(creation=creation, selectors=selectors, compile_ms=elapsed)


def _parse_solc_output(output: str, binary_heading: str) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    current: str | None = None
    want_binary = False
    in_hashes = False
    for line in output.splitlines():
        stripped = line.strip()
        header = re.match(r"^======= .*:([^:]+) =======$", stripped)
        if header:
            current = header.group(1)
            contracts.setdefault(current, {"creation": "", "selectors": {}})
            want_binary = False
            in_hashes = False
            continue
        if current is None:
            continue
        if stripped == binary_heading:
            want_binary = True
            in_hashes = False
            continue
        if stripped == "Function signatures:":
            in_hashes = True
            want_binary = False
            continue
        if want_binary:
            if stripped and re.fullmatch(r"[0-9a-fA-F]+", stripped):
                contracts[current]["creation"] = stripped.lower()
                want_binary = False
            continue
        if in_hashes:
            signature = re.match(r"^([0-9a-fA-F]{8}):\s*(.+)$", stripped)
            if signature:
                contracts[current]["selectors"][signature.group(2)] = signature.group(1).lower()
            elif stripped:
                in_hashes = False
    return contracts


def calldata(signature: str, arguments: list[Any], selectors: dict[str, str]) -> str:
    if signature not in selectors:
        raise HarnessError(f"compiler did not report selector for {signature}")
    open_paren = signature.find("(")
    if open_paren < 0 or not signature.endswith(")"):
        raise HarnessError(f"invalid function signature: {signature}")
    types = _split_types(signature[open_paren + 1 : -1])
    return "0x" + selectors[signature] + encode_abi(types, arguments).hex()


def encode_abi(types: list[str], arguments: list[Any]) -> bytes:
    if len(types) != len(arguments):
        raise HarnessError(f"ABI arity mismatch: {len(types)} types, {len(arguments)} values")
    heads: list[bytes | None] = []
    tails: list[bytes] = []
    head_size = 32 * len(types)
    for abi_type, argument in zip(types, arguments):
        if _dynamic_type(abi_type):
            encoded = _encode_dynamic(abi_type, argument)
            heads.append(None)
            tails.append(encoded)
        else:
            heads.append(_encode_static(abi_type, argument))

    result = bytearray()
    tail_index = 0
    tail_offset = head_size
    for head in heads:
        if head is not None:
            result.extend(head)
            continue
        tail = tails[tail_index]
        result.extend(_word(tail_offset))
        tail_offset += len(tail)
        tail_index += 1
    for tail in tails:
        result.extend(tail)
    return bytes(result)


def _split_types(types: str) -> list[str]:
    if not types:
        return []
    result: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(types):
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(types[start:index].strip())
            start = index + 1
    result.append(types[start:].strip())
    return result


def _dynamic_type(abi_type: str) -> bool:
    return abi_type in ("bytes", "string") or abi_type.endswith("[]")


def _encode_static(abi_type: str, argument: Any) -> bytes:
    integer = re.fullmatch(r"(u?int)([0-9]*)", abi_type)
    if integer:
        bits = int(integer.group(2) or "256")
        value = _integer(argument)
        if integer.group(1) == "uint":
            if value < 0 or value >= 1 << bits:
                raise HarnessError(f"{value} is outside {abi_type}")
        else:
            minimum, maximum = -(1 << (bits - 1)), 1 << (bits - 1)
            if value < minimum or value >= maximum:
                raise HarnessError(f"{value} is outside {abi_type}")
            value %= 1 << 256
        return _word(value)
    if abi_type == "bool":
        if argument not in (True, False, 0, 1, "0", "1"):
            raise HarnessError(f"invalid bool: {argument}")
        return _word(1 if argument in (True, 1, "1") else 0)
    if abi_type == "address":
        raw = _hex_bytes(argument)
        if len(raw) != 20:
            raise HarnessError(f"address must contain 20 bytes: {argument}")
        return raw.rjust(32, b"\x00")
    fixed_bytes = re.fullmatch(r"bytes([0-9]+)", abi_type)
    if fixed_bytes:
        size = int(fixed_bytes.group(1))
        raw = _hex_bytes(argument)
        if not 1 <= size <= 32 or len(raw) != size:
            raise HarnessError(f"{abi_type} requires exactly {size} bytes")
        return raw.ljust(32, b"\x00")
    raise HarnessError(f"unsupported static ABI type: {abi_type}")


def _encode_dynamic(abi_type: str, argument: Any) -> bytes:
    if abi_type == "bytes":
        raw = _hex_bytes(argument)
        return _word(len(raw)) + _pad32(raw)
    if abi_type == "string":
        raw = str(argument).encode()
        return _word(len(raw)) + _pad32(raw)
    if abi_type.endswith("[]"):
        element_type = abi_type[:-2]
        if _dynamic_type(element_type):
            raise HarnessError(f"nested dynamic array is not supported: {abi_type}")
        if not isinstance(argument, list):
            raise HarnessError(f"{abi_type} value must be a JSON list")
        return _word(len(argument)) + b"".join(
            _encode_static(element_type, element) for element in argument
        )
    raise HarnessError(f"unsupported dynamic ABI type: {abi_type}")


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text, 16 if text.lower().startswith("0x") else 10)


def _hex_bytes(value: Any) -> bytes:
    if not isinstance(value, str):
        raise HarnessError(f"expected hex string, got {value!r}")
    text = value[2:] if value.startswith("0x") else value
    if len(text) % 2:
        text = "0" + text
    try:
        return bytes.fromhex(text)
    except ValueError as error:
        raise HarnessError(f"invalid hex bytes: {value}") from error


def _word(value: int) -> bytes:
    if value < 0 or value >= 1 << 256:
        raise HarnessError(f"word value outside uint256: {value}")
    return value.to_bytes(32, "big")


def _pad32(value: bytes) -> bytes:
    return value + bytes((-len(value)) % 32)


class Rpc:
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout
        self.identifier = 0

    def request(self, method: str, params: list[Any]) -> dict[str, Any]:
        self.identifier += 1
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": self.identifier, "method": method, "params": params}
        ).encode()
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            try:
                return json.loads(error.read().decode())
            except Exception as decode_error:
                raise HarnessError(f"JSON-RPC HTTP error for {method}: {error}") from decode_error
        except (urllib.error.URLError, TimeoutError) as error:
            raise HarnessError(f"JSON-RPC transport error for {method}: {error}") from error

    def result(self, method: str, params: list[Any]) -> Any:
        response = self.request(method, params)
        if "result" not in response:
            raise HarnessError(f"{method} failed: {response.get('error')}")
        return response["result"]

    def reachable(self) -> bool:
        try:
            return isinstance(self.result("eth_chainId", []), str)
        except HarnessError:
            return False

    def snapshot(self) -> str:
        return self.result("evm_snapshot", [])

    def revert(self, snapshot: str) -> None:
        if self.result("evm_revert", [snapshot]) is not True:
            raise HarnessError(f"could not revert snapshot {snapshot}")

    def call(self, transaction: dict[str, str]) -> dict[str, str]:
        response = self.request("eth_call", [transaction, "latest"])
        if "result" in response:
            return {"status": "ok", "data": _normalize_hex(response["result"])}
        return {"status": "revert", "data": _revert_data(response.get("error"))}

    def send(self, transaction: dict[str, str]) -> dict[str, Any]:
        response = self.request("eth_sendTransaction", [transaction])
        if "result" not in response:
            return {
                "status": "rejected",
                "data": _revert_data(response.get("error")),
            }
        receipt = self.wait_receipt(response["result"])
        if receipt is None:
            return {"status": "no-receipt"}
        return receipt

    def wait_receipt(self, transaction_hash: str) -> dict[str, Any] | None:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            receipt = self.result("eth_getTransactionReceipt", [transaction_hash])
            if receipt:
                return receipt
            time.sleep(0.05)
        return None

    def storage_root(self, address: str) -> str:
        proof = self.result("eth_getProof", [address, [], "latest"])
        if not isinstance(proof, dict) or not isinstance(proof.get("storageHash"), str):
            raise HarnessError("eth_getProof returned no storageHash")
        return proof["storageHash"].lower()

    def account_state(self, address: str) -> dict[str, Any]:
        code = self.result("eth_getCode", [address, "latest"])
        return {
            "storage": self.storage_root(address),
            "balance": _normalize_quantity(self.result("eth_getBalance", [address, "latest"])),
            "nonce": _normalize_quantity(
                self.result("eth_getTransactionCount", [address, "latest"])
            ),
            "code_exists": bool(code and code != "0x"),
        }

    def mine(self, blocks: int = 1) -> dict[str, str]:
        if blocks < 1:
            raise HarnessError(f"block count must be positive: {blocks}")
        self.result("anvil_mine", [hex(blocks)])
        block = self.result("eth_getBlockByNumber", ["latest", False])
        if not isinstance(block, dict):
            raise HarnessError("eth_getBlockByNumber returned no block")
        return {
            "number": _normalize_quantity(block.get("number")),
            "timestamp": _normalize_quantity(block.get("timestamp")),
        }

    def advance_to(self, timestamp: int) -> dict[str, str]:
        self.result("evm_setNextBlockTimestamp", [timestamp])
        return self.mine()


@contextlib.contextmanager
def runtime_rpc(
    rpc_url: str,
    spawn_anvil: bool,
    anvil: str | None,
    hardfork: str,
    timeout: float,
) -> Iterator[Rpc]:
    if not spawn_anvil:
        rpc = Rpc(rpc_url, timeout)
        if not rpc.reachable():
            raise HarnessSkip(f"no JSON-RPC node at {rpc_url}")
        yield rpc
        return

    executable = anvil or shutil.which("anvil")
    if not executable:
        raise HarnessSkip("anvil is not installed")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            executable,
            "--silent",
            "--port",
            str(port),
            "--hardfork",
            hardfork,
            "--timestamp",
            "1",
            "--gas-limit",
            "1000000000",
            "--disable-code-size-limit",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        rpc = Rpc(url, timeout)
        deadline = time.monotonic() + min(timeout, 15)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                raise HarnessError(f"anvil exited during startup: {stderr}")
            if rpc.reachable():
                yield rpc
                return
            time.sleep(0.1)
        raise HarnessError("timed out waiting for anvil")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def deploy_and_run(
    rpc: Rpc,
    artifact: Artifact,
    workload: dict[str, Any],
    sender: str = DEFAULT_SENDER,
) -> dict[str, Any]:
    aliases = {f"$account{index}": account for index, account in enumerate(DEFAULT_ACCOUNTS)}
    aliases["$sender"] = sender
    constructor = workload.get("constructor", {})
    constructor_types = constructor.get("types", [])
    constructor_args = _resolve_value(constructor.get("args", []), aliases)
    creation = artifact.creation + encode_abi(constructor_types, constructor_args).hex()
    transaction = {
        "from": sender,
        "data": "0x" + creation,
        "gas": hex(int(workload.get("deploy_gas", DEFAULT_GAS))),
        "value": hex(_integer(constructor.get("value", 0))),
    }
    receipt = rpc.send(transaction)
    if not isinstance(receipt, dict) or receipt.get("status") != "0x1":
        return {
            "semantics": {"deployment": {"status": _receipt_status(receipt)}},
            "metrics": {
                "compile_ms": artifact.compile_ms,
                "creation_bytes": len(creation) // 2,
            },
        }
    address = receipt.get("contractAddress")
    if not isinstance(address, str):
        raise HarnessError("deployment receipt contains no contract address")
    runtime = rpc.result("eth_getCode", [address, "latest"])
    semantics: dict[str, Any] = {
        "deployment": {
            "status": "ok",
            "address": address.lower(),
            "state": rpc.account_state(address),
        },
        "steps": [],
    }
    aliases["$contract"] = address
    watched = _watched_accounts(rpc, workload.get("watch", []), aliases)
    if watched:
        semantics["deployment"]["watched"] = watched
    metrics: dict[str, Any] = {
        "compile_ms": artifact.compile_ms,
        "creation_bytes": len(creation) // 2,
        "runtime_bytes": max(0, (len(runtime) - 2) // 2),
        "deployment_gas": int(receipt.get("gasUsed", "0x0"), 16),
        "steps": [],
    }

    for index, step in enumerate(workload.get("steps", [])):
        label = step.get("label", f"step-{index}")
        kind = step.get("kind", "call")
        if kind == "mine":
            blocks = int(step.get("blocks", 1))
            rpc.mine(blocks)
            semantics["steps"].append(
                {
                    "label": label,
                    "kind": kind,
                    "blocks": blocks,
                }
            )
            continue
        if kind == "time":
            timestamp = int(step["timestamp"])
            rpc.advance_to(timestamp)
            semantics["steps"].append(
                {
                    "label": label,
                    "kind": kind,
                    "timestamp": timestamp,
                }
            )
            continue

        signature = step.get("signature")
        data = step.get("data")
        if data is None:
            if not isinstance(signature, str):
                raise HarnessError(f"{label}: step needs signature or data")
            data = calldata(
                signature,
                _resolve_value(step.get("args", []), aliases),
                artifact.selectors,
            )
        sender_value = _resolve_value(step.get("from", sender), aliases)
        if not isinstance(sender_value, str):
            raise HarnessError(f"{label}: transaction sender must be an address")
        destination = _resolve_value(step.get("to", address), aliases)
        if not isinstance(destination, str):
            raise HarnessError(f"{label}: transaction destination must be an address")
        tx = {
            "from": sender_value,
            "to": destination,
            "data": data,
            "gas": hex(int(step.get("gas", DEFAULT_GAS))),
            "value": hex(_integer(step.get("value", 0))),
        }
        if kind == "call":
            semantics["steps"].append(
                {"label": label, "kind": kind, "call": rpc.call(tx)}
            )
            continue
        if kind != "tx":
            raise HarnessError(f"{label}: unknown step kind {kind}")

        preview = rpc.call(tx)
        step_receipt = rpc.send(tx)
        capture = step.get("capture")
        if capture is not None:
            _capture_alias(preview, capture, aliases, label)
        semantic_step = {
            "label": label,
            "kind": kind,
            "preview": preview,
            "status": _receipt_status(step_receipt),
            "logs": _normalize_logs(step_receipt.get("logs", [])),
            "state": rpc.account_state(address),
        }
        watched = _watched_accounts(
            rpc,
            [*workload.get("watch", []), *step.get("watch", [])],
            aliases,
        )
        if watched:
            semantic_step["watched"] = watched
        semantics["steps"].append(semantic_step)
        metrics["steps"].append(
            {
                "label": label,
                "gas": int(step_receipt.get("gasUsed", "0x0"), 16)
                if "gasUsed" in step_receipt
                else None,
            }
        )

    semantics["final_state"] = rpc.account_state(address)
    watched = _watched_accounts(rpc, workload.get("watch", []), aliases)
    if watched:
        semantics["final_watched"] = watched
    return {"semantics": semantics, "metrics": metrics}


def _resolve_value(value: Any, aliases: dict[str, str]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        if value not in aliases:
            raise HarnessError(f"unknown runtime value alias: {value}")
        return aliases[value]
    if isinstance(value, list):
        return [_resolve_value(entry, aliases) for entry in value]
    return value


def _capture_alias(
    preview: dict[str, str],
    capture: dict[str, Any],
    aliases: dict[str, str],
    label: str,
) -> None:
    alias = capture.get("alias")
    if not isinstance(alias, str) or not alias.startswith("$"):
        raise HarnessError(f"{label}: capture alias must start with '$'")
    if preview.get("status") != "ok":
        raise HarnessError(f"{label}: cannot capture from reverted preview")
    data = preview.get("data", "0x")[2:]
    word_index = int(capture.get("word", 0))
    start = word_index * 64
    word = data[start : start + 64]
    if len(word) != 64:
        raise HarnessError(f"{label}: return data has no word {word_index}")
    capture_type = capture.get("type", "address")
    if capture_type != "address":
        raise HarnessError(f"{label}: unsupported capture type {capture_type}")
    aliases[alias] = "0x" + word[-40:]


def _watched_accounts(
    rpc: Rpc,
    values: list[Any],
    aliases: dict[str, str],
) -> dict[str, dict[str, Any]]:
    result = {}
    for value in values:
        address = _resolve_value(value, aliases)
        if not isinstance(address, str):
            raise HarnessError(f"watched account must be an address: {value!r}")
        result[str(value)] = rpc.account_state(address)
    return result


def _receipt_status(receipt: dict[str, Any]) -> str:
    status = receipt.get("status")
    if status == "0x1":
        return "ok"
    if status == "0x0":
        return "revert"
    return str(status or "unknown")


def _normalize_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "address": str(log.get("address", "")).lower(),
            "topics": [_normalize_hex(topic) for topic in log.get("topics", [])],
            "data": _normalize_hex(log.get("data")),
        }
        for log in logs
    ]


def _revert_data(error: Any) -> str:
    if not isinstance(error, dict):
        return "0x"
    data = error.get("data")
    while isinstance(data, dict):
        data = data.get("data") or data.get("result")
    return _normalize_hex(data)


def _normalize_hex(value: Any) -> str:
    if isinstance(value, str) and value.startswith("0x"):
        text = value[2:].lower()
        return "0x" + ("0" + text if len(text) % 2 else text)
    return "0x"


def _normalize_quantity(value: Any) -> str:
    if not isinstance(value, str):
        raise HarnessError(f"invalid JSON-RPC quantity: {value!r}")
    return hex(int(value, 16))
