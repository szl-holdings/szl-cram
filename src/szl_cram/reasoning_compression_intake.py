"""Offline exact-byte admission for SZL's synthetic compression/energy SFT dataset.

Source-only candidate. No GPU, provider, network, training or deployment calls.
This is a *data-integrity* gate, not a signature, license review, or proof that a
trained model is accurate. The official CRAM/enwik9 datasets are never fetched.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

TRAIN_SHA256 = "36048599efb5abc1b4ca0ef21ac1eae072c7412728f1dacd9cbe437b623df502"
MANIFEST_SHA256 = "d8cb77a517e69ded7314747df5e68db3481aaba3639a8529ed1e3eb47c448904"
TRAIN_COUNT = 1320
MAX_TRAIN_BYTES = 2_000_000
TASKS = {"graph-energy", "ideal-bits", "mdl-transfer"}
COSTS = {"A": 1, "B": 2, "C": 3, "D": 3}


class AdmissionHold(ValueError):
    """A refused, unsupported, or unauthoritative training input."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise AdmissionHold(code)


def _no_dupes(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for key, value in pairs:
        _require(key not in out, "DUPLICATE_JSON_KEY")
        out[key] = value
    return out


def _json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8", "strict"),
                          object_pairs_hook=_no_dupes,
                          parse_constant=lambda _: _require(False, "NONFINITE_JSON"))
    except (ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise AdmissionHold("INVALID_JSON") from exc


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_gold(task: str, packet: dict) -> dict:
    """Separately recompute finite targets. Not Lean or archaeological proof."""
    _require(type(packet) is dict and task in TASKS, "TASK_CONTRACT")
    if task == "graph-energy":
        n = packet.get("nodes")
        colors = packet.get("candidate")
        edges = packet.get("edges")
        _require(type(n) is int and 5 <= n <= 7 and packet.get("colors") == 3,
                 "GRAPH_SIZE_CONTRACT")
        _require(type(colors) is list and len(colors) == n and
                 all(type(c) is int and 0 <= c < 3 for c in colors),
                 "GRAPH_COLOR_CONTRACT")
        _require(type(edges) is list and len(edges) <= n * (n-1) // 2 and
                 all(type(e) is list and len(e) == 2 and
                     type(e[0]) is int and type(e[1]) is int and
                     0 <= e[0] < e[1] < n for e in edges),
                 "GRAPH_EDGES_CONTRACT")
        _require(len({tuple(e) for e in edges}) == len(edges), "GRAPH_DUPLICATE_EDGE")
        conflicts = sum(colors[a] == colors[b] for a, b in edges)
        return {"verdict": "PASS" if conflicts == 0 else "HOLD",
                "code": "GRAPH_ZERO_CONFLICT_ENERGY" if conflicts == 0 else "GRAPH_EDGE_CONFLICT",
                "conflicts": conflicts, "energy": conflicts}

    if task == "ideal-bits":
        symbols = packet.get("symbols")
        reported = packet.get("reported_ideal_bits")
        _require(type(symbols) is str and 1 <= len(symbols) <= 64
                 and set(symbols) <= COSTS.keys(), "BITS_SYMBOL_CONTRACT")
        _require(type(reported) is int and 0 <= reported <= 200,
                 "BITS_NUMERIC_CONTRACT")
        exact = sum(COSTS[x] for x in symbols)
        return {"verdict": "PASS" if exact == reported else "HOLD",
                "code": "EXACT_IDEAL_BITS" if exact == reported else "BITS_MISREPORTED",
                "correct_ideal_bits": exact, "symbol_count": len(symbols)}

    fields = ("online_payload_bytes", "online_decoder_bytes",
              "transfer_payload_bytes", "transfer_decoder_bytes",
              "prior_weights_bytes")
    _require(all(type(packet.get(k)) is int and 0 <= packet[k] <= 150_000
                 for k in fields), "MODEL_BYTES_CONTRACT")
    _require(type(packet.get("reported_net_win")) is bool, "MODEL_BOOLEAN_CONTRACT")
    online = packet["online_payload_bytes"] + packet["online_decoder_bytes"]
    transfer = packet["transfer_payload_bytes"] + packet["transfer_decoder_bytes"] + packet["prior_weights_bytes"]
    win = transfer < online
    correct = packet["reported_net_win"] is win
    return {"verdict": "PASS" if correct else "HOLD",
            "code": "MODEL_INCLUDED_IN_TOTAL" if correct else "MODEL_COST_ACCOUNTING_MISMATCH",
            "online_total_bytes": online, "transfer_total_bytes": transfer,
            "net_saving_bytes": online-transfer, "candidate_wins": win}


def check_messages(raw: bytes, expected_count: int = TRAIN_COUNT) -> dict:
    """Validate each projection; never accept hidden dev/test rows as training."""
    _require(0 < len(raw) <= MAX_TRAIN_BYTES, "TRAIN_BYTE_BUDGET")
    rows = raw.splitlines()
    _require(len(rows) == expected_count, "TRAIN_ROW_COUNT_MISMATCH")
    domains: dict[str, int] = {}
    for line in rows:
        _require(0 < len(line) < 5000, "TRAIN_ROW_BYTE_BUDGET")
        row = _json(line)
        _require(type(row) is dict and set(row) == {"messages"}, "TRAIN_EXACT_SCHEMA")
        messages = row["messages"]
        _require(type(messages) is list and len(messages) == 3, "MESSAGES_SHAPE")
        _require(all(type(m) is dict and set(m) == {"role", "content"}
                     and type(m["content"]) is str and len(m["content"]) <= 4096
                     for m in messages), "MESSAGES_CONTRACT")
        _require([m["role"] for m in messages] == ["system", "user", "assistant"],
                 "MESSAGES_ROLES")
        user = _json(messages[1]["content"].encode("utf-8"))
        _require(type(user) is dict and set(user) == {"task", "data"},
                 "TASK_ENVELOPE")
        expected = exact_gold(user["task"], user["data"])
        reported = _json(messages[2]["content"].encode("utf-8"))
        _require(reported == expected and type(reported) is dict and
                 set(reported) == set(expected), "TRAIN_LABEL_DISAGREEMENT")
        domains[user["task"]] = domains.get(user["task"], 0) + 1
    return {"state": "SYNTHETIC_TRAIN_INPUT_VERIFIED_ONLY",
            "rows": len(rows), "domains": dict(sorted(domains.items())),
            "model_quality_verified": False, "signature_verified": False,
            "gpu_training_started": False, "production_authorized": False}


def _load_bytes(directory: Path, name: str, cap: int) -> bytes:
    _require(not directory.is_symlink(), "BUNDLE_DIRECTORY_SYMLINK_REFUSED")
    p = directory / name
    _require(p.is_file() and not p.is_symlink(), "BUNDLE_FILE_MISSING_OR_LINKED")
    _require(p.stat().st_size <= cap, "BUNDLE_FILE_TOO_LARGE")
    return p.read_bytes()


def verify_bundle(directory: Path) -> tuple[dict, bytes]:
    manifest_raw = _load_bytes(directory, "manifest.json", 40_000)
    train_raw = _load_bytes(directory, "train_sft.jsonl", MAX_TRAIN_BYTES)
    _require(hmac.compare_digest(_hash(manifest_raw), MANIFEST_SHA256),
             "MANIFEST_SOURCE_DRIFT")
    _require(hmac.compare_digest(_hash(train_raw), TRAIN_SHA256),
             "TRAIN_BYTES_DRIFT")
    m = _json(manifest_raw)
    _require(m.get("schema") == "szl.reasoning-compression.dataset-manifest/v0",
             "DATASET_SCHEMA")
    _require(m.get("classification") == "AUTHOR_SYNTHETIC_SOFTWARE_FIXTURES_ONLY",
             "DATASET_CLASS")
    _require(m.get("train_input_path") == "train_sft.jsonl"
             and m.get("dev_test_training_admission") == "DENY; evaluation only",
             "SPLIT_POLICY")
    _require(m.get("sizes", {}).get("train") == TRAIN_COUNT
             and m.get("sha256", {}).get("train_sft.jsonl") == TRAIN_SHA256,
             "TRAIN_MANIFEST_BINDING")
    _require(m.get("training_run_started") is False
             and m.get("upstream_paper_code_data_ingested") is False
             and m.get("receipt_signed") is False, "FALSE_AUTHORITY_CLAIM")
    verdict = check_messages(train_raw)
    verdict["train_sha256"] = TRAIN_SHA256
    verdict["manifest_sha256"] = MANIFEST_SHA256
    return verdict, train_raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline synthetic SFT integrity gate; never launches training.")
    parser.add_argument("--bundle", required=True, type=Path,
                        help="Directory containing manifest.json and train_sft.jsonl")
    parser.add_argument("--export", type=Path, help="Optional local create-only trainer input path")
    parser.add_argument("--owner-admitted-export", action="store_true",
                        help="A source owner has separately reviewed local export; not paid-job approval")
    args = parser.parse_args(argv)
    try:
        report, data = verify_bundle(args.bundle)
        if args.export is not None:
            _require(args.owner_admitted_export, "EXPORT_REQUIRES_SOURCE_OWNER_ADMISSION")
            _require(not args.export.is_symlink(), "EXPORT_LINK_REFUSED")
            with args.export.open("xb") as handle:
                handle.write(data)
            report["exported_to_local_input_only"] = True
        print(json.dumps(report, sort_keys=True))
        return 0
    except (AdmissionHold, FileExistsError, OSError) as exc:
        print(json.dumps({"state": "HOLD", "reason": type(exc).__name__ if not
                          isinstance(exc, AdmissionHold) else str(exc),
                          "training_started": False, "source_qualified": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
