import csv
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SnapshotIntegrityTests(unittest.TestCase):
    def test_results_table_has_provenance_columns(self):
        with (ROOT / "results_summary.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 10)
        required = {"experiment", "fid_legacy_project", "comparison_scope", "entry_point"}
        self.assertTrue(required.issubset(rows[0]))
        for row in rows:
            if row["entry_point"] != "metrics_only_no_script_in_snapshot":
                self.assertTrue((ROOT / row["entry_point"]).exists(), row["entry_point"])

    def test_json_metrics_are_valid(self):
        files = list((ROOT / "03_metrics_and_logs").rglob("*.json"))
        self.assertGreater(len(files), 20)
        for path in files:
            with path.open(encoding="utf-8") as handle:
                json.load(handle)

    def test_canonical_figures_are_valid_svg(self):
        for name in ("interview_results_roadmap.svg", "clip_control_sweep.svg"):
            root = ET.parse(ROOT / "04_visual_assets" / name).getroot()
            self.assertTrue(root.tag.endswith("svg"))

    def test_readme_targets_exist(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for target in (
            "04_visual_assets/interview_results_roadmap.svg",
            "04_visual_assets/clip_control_sweep.svg",
            "04_visual_assets/qualitative_samples_compact.png",
            "results_summary.csv",
            "docs/data_quality_and_sdxl_extension.md",
            "docs/experiment_process.md",
        ):
            self.assertIn(target, readme)
            if target.endswith(".md"):
                self.assertTrue((ROOT / target).exists(), target)
            else:
                self.assertTrue((ROOT / target).exists(), target)

    def test_data_audit_and_b1_evidence(self):
        with (ROOT / "03_metrics_and_logs/phase6_data_audit/audit_summary.json").open(encoding="utf-8") as handle:
            audit = json.load(handle)
        self.assertEqual(audit["image_count_before_dedup"], 21551)
        self.assertEqual(audit["unique_sha256_count"], 17029)
        self.assertEqual(audit["bad_file_count"], 0)

        with (ROOT / "03_metrics_and_logs/phase6_b1_formal/metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        self.assertEqual(metrics["dataset_size"], 17029)
        self.assertAlmostEqual(metrics["FID"], 45.07, places=2)

    def test_full_process_sources_and_plain_baseline_are_present(self):
        for folder in (
            "phase1_early_tuning",
            "phase2_module_tuning",
            "phase3_generator_strengthening",
            "phase5_clip_tuning",
        ):
            scripts = list((ROOT / "02_selected_experiments/full_process" / folder).glob("*.py"))
            self.assertGreater(len(scripts), 0, folder)
        self.assertTrue((ROOT / "04_visual_assets/phase2_baseline_no_modules_epoch200.png").exists())
        self.assertTrue((ROOT / "02_selected_experiments/full_process/phase5_clip_tuning/clip_C2_lambda_0025.py").exists())


if __name__ == "__main__":
    unittest.main()
