"""Offline negative/positive controls for the synthetic training-only intake."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "szl_cram" / "reasoning_compression_intake.py"
SPEC = importlib.util.spec_from_file_location("cram_reasoning_compression_intake", SOURCE)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(gate)


def envelope(task, packet, *, label=None):
    gold = gate.exact_gold(task, packet) if label is None else label
    return {"messages": [
        {"role": "system", "content": "finite synthetic contract"},
        {"role": "user", "content": json.dumps({"task":task, "data":packet})},
        {"role": "assistant", "content": json.dumps(gold)},
    ]}


class IntakeTests(unittest.TestCase):
    def test_exact_bit_math(self):
        packet = {"symbols":"AABBCD", "reported_ideal_bits":12}
        result = gate.exact_gold("ideal-bits", packet)
        self.assertEqual(result["correct_ideal_bits"],12)
        self.assertEqual(result["verdict"],"PASS")

    def test_incorrect_bit_cost_refused(self):
        result = gate.exact_gold("ideal-bits", {"symbols":"AABC", "reported_ideal_bits":2})
        self.assertEqual(result["verdict"],"HOLD")

    def test_unknown_task_type_refused(self):
        with self.assertRaises(gate.AdmissionHold):
            gate.exact_gold([],{})

    def test_boolean_bit_count_refused(self):
        with self.assertRaises(gate.AdmissionHold):
            gate.exact_gold("ideal-bits", {"symbols":"AA", "reported_ideal_bits":True})

    def test_graph_conflict_energy(self):
        packet={"nodes":5,"colors":3,"edges":[[0,1],[1,2]],"candidate":[0,1,0,1,2]}
        self.assertEqual(gate.exact_gold("graph-energy",packet)["energy"],0)
        packet["candidate"][1] = 0
        self.assertEqual(gate.exact_gold("graph-energy",packet)["verdict"],"HOLD")

    def test_duplicate_graph_edge_refused(self):
        packet={"nodes":5,"colors":3,"edges":[[0,1],[0,1]],"candidate":[0,1,0,1,2]}
        with self.assertRaises(gate.AdmissionHold):
            gate.exact_gold("graph-energy",packet)

    def test_model_cost_includes_prior(self):
        packet={"online_payload_bytes":1000,"online_decoder_bytes":50,
                "transfer_payload_bytes":900,"transfer_decoder_bytes":50,
                "prior_weights_bytes":200,"reported_net_win":False}
        result=gate.exact_gold("mdl-transfer",packet)
        self.assertEqual(result["transfer_total_bytes"],1150)
        self.assertEqual(result["verdict"],"PASS")
        packet["reported_net_win"]=True
        self.assertEqual(gate.exact_gold("mdl-transfer",packet)["verdict"],"HOLD")

    def test_one_valid_message(self):
        packet={"symbols":"ABC", "reported_ideal_bits":6}
        raw=(json.dumps(envelope("ideal-bits",packet))+"\n").encode()
        out=gate.check_messages(raw, expected_count=1)
        self.assertEqual(out["rows"],1)
        self.assertFalse(out["gpu_training_started"])

    def test_wrong_assistant_label_refused(self):
        packet={"symbols":"ABC", "reported_ideal_bits":6}
        raw=(json.dumps(envelope("ideal-bits",packet,label={"verdict":"PASS"}))+"\n").encode()
        with self.assertRaises(gate.AdmissionHold):
            gate.check_messages(raw, expected_count=1)

    def test_row_count_refused(self):
        with self.assertRaises(gate.AdmissionHold):
            gate.check_messages(b"",expected_count=0)

    def test_train_extra_split_field_refused(self):
        packet={"symbols":"ABC","reported_ideal_bits":6}
        row=envelope("ideal-bits",packet)
        row["split"]="test"
        with self.assertRaises(gate.AdmissionHold):
            gate.check_messages((json.dumps(row)+"\n").encode(), expected_count=1)

    def test_duplicate_json_key_refused(self):
        blob=b'{"messages":[],"messages":[]}\n'
        with self.assertRaises(gate.AdmissionHold):
            gate.check_messages(blob, expected_count=1)

    def test_wrong_manifest_hash_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"manifest.json").write_text('{"schema":"forged"}\n')
            (root/"train_sft.jsonl").write_text('{"messages":[]}\n')
            with self.assertRaisesRegex(gate.AdmissionHold,"MANIFEST_SOURCE_DRIFT"):
                gate.verify_bundle(root)

    def test_cli_never_launches_training(self):
        with mock.patch.object(gate,"verify_bundle",return_value=({"rows":1},b"x\n")):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/"train.jsonl"
                self.assertEqual(gate.main(["--bundle",d,"--export",str(p)]),1)
                self.assertFalse(p.exists())

    def test_source_file_never_imports_http_or_gpu(self):
        raw=SOURCE.read_text()
        self.assertNotIn("import requests",raw)
        self.assertNotIn("from huggingface_hub",raw)
        self.assertNotIn("from torch",raw)
        self.assertNotIn("subprocess.",raw)

    def test_expected_source_digest_format(self):
        for s in (gate.TRAIN_SHA256,gate.MANIFEST_SHA256):
            self.assertEqual(len(s),64)
            self.assertTrue(all(c in "0123456789abcdef" for c in s))


if __name__ == "__main__":
    unittest.main()
