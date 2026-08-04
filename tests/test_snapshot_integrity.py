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
        for target in ("04_visual_assets/interview_results_roadmap.svg", "04_visual_assets/clip_control_sweep.svg", "results_summary.csv"):
            self.assertIn(target, readme)
            self.assertTrue((ROOT / target).exists(), target)


if __name__ == "__main__":
    unittest.main()
