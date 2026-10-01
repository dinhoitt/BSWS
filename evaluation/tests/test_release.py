import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import finite, paired_summary, read_manifest, sha256
from legacy_metrics import METRICS, MetricEvaluator
from utmos_metric import SPEECHMOS_COMMIT


class ReleaseTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(len(METRICS), 8)
        self.assertEqual(MetricEvaluator().crepe_batch_size, 512)
        self.assertEqual(SPEECHMOS_COMMIT, "ed25eacbfa42b99156c36ebec67a733b5dbb9b79")

    def test_invalid_not_zero(self):
        for value in (None, "", "nan", math.inf, -math.inf):
            self.assertIsNone(finite(value))
        self.assertEqual(finite(0), 0)

    def test_common_valid(self):
        result = paired_summary(
            [{"utterance_id": "U01", "mcd": 1}, {"utterance_id": "U02", "mcd": 100}],
            [{"utterance_id": "U01", "mcd": 2}, {"utterance_id": "U02", "mcd": None}],
            "mcd", expected_ids=["U01", "U02", "U03"])
        self.assertEqual(result["included_ids"], ["U01"])
        self.assertEqual(result["excluded_ids"], ["U02", "U03"])
        self.assertEqual(result["benefit_a"], 1)
        self.assertIsNone(result["ci95_low"])

    def test_paired_bootstrap(self):
        a = [{"utterance_id": str(i), "utmos": i + 1} for i in range(20)]
        b = [{"utterance_id": str(i), "utmos": i} for i in range(20)]
        first, second = paired_summary(a, b, "utmos"), paired_summary(a, list(reversed(b)), "utmos")
        self.assertEqual(first, second)
        self.assertEqual(first["ci95_low"], 1)
        self.assertEqual(first["ci95_high"], 1)

    def test_duplicates(self):
        with self.assertRaises(ValueError):
            paired_summary([{"utterance_id": "U01"}] * 2, [], "pesq")

    def test_empty_is_unavailable(self):
        result = paired_summary([], [], "pesq")
        self.assertEqual(result["n_common"], 0)
        self.assertIsNone(result["mean_a"])
        json.dumps(result, allow_nan=False)

    def test_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "clip.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\0\0" * 240)
            manifest = root / "manifest.csv"
            header = "utterance_id,model_key,path,sha256\n"
            good = f"U01,GT,clip.wav,{sha256(audio)}\nU01,model,clip.wav,{sha256(audio)}\n"
            manifest.write_text(header + good, encoding="utf-8")
            self.assertEqual(len(read_manifest(manifest, root)), 2)
            manifest.write_text(header + good.replace(sha256(audio), "bad"), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_manifest(manifest, root)
            manifest.write_text(header + "U01,model,clip.wav,\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_manifest(manifest, root)
            manifest.write_text(header + "U01,GT,../outside.wav,\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
