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
            with path.open(encoding="utf-8-sig") as handle:
                json.load(handle)

    def test_canonical_figures_are_valid_svg(self):
        for name in ("interview_results_roadmap.svg", "clip_control_sweep.svg", "sdxl_fid_coverage_tradeoff.svg"):
            root = ET.parse(ROOT / "04_visual_assets" / name).getroot()
            self.assertTrue(root.tag.endswith("svg"))

    def test_readme_targets_exist(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for target in (
            "04_visual_assets/stage_figures/01_前期训练与增强/01_训练轮数_FID.svg",
            "04_visual_assets/stage_figures/02_G_D模块调优/06_D端模块_FID.svg",
            "04_visual_assets/stage_figures/03_G强化与训练策略/03_G结构强化_FID.svg",
            "04_visual_assets/stage_figures/04_CLIP调优/07_CLIP_FID.svg",
            "04_visual_assets/stage_figures/05_部署与量化/27_混合精度_FID.svg",
            "04_visual_assets/stage_figures/06_服务压测/32_并发_P99.svg",
            "04_visual_assets/stage_figures/06_服务压测/37_Soak阶段_P99.svg",
            "docs/experiment_process.md",
            "docs/baseline_map.md",
            "docs/interview_playbook.md",
            "docs/month1_audit_2026-08.md",
            "docs/next_phase_deployment_plan.md",
            "docs/deployment_optimization.md",
            "docs/dcgan_core_experiment_record.md",
            "03_metrics_and_logs/deployment_optimization/deployment_task_status.csv",
            "03_metrics_and_logs/deployment_optimization/deployment_quantization_summary.csv",
            "03_metrics_and_logs/deployment_optimization/service_operational_summary_v08.csv",
            "03_metrics_and_logs/stage_figures_map.csv",
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

    def test_dcgan_core_catalog_and_figure_audit(self):
        with (ROOT / "03_metrics_and_logs/dcgan_core/全实验指标汇总.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 49)
        self.assertTrue(all(row["protocol"] for row in rows))
        width3_ema = next(row for row in rows if row["experiment"] == "11_G_DiffAug_EMA_20K")
        self.assertAlmostEqual(float(width3_ema["fid"]), 38.88, places=2)
        clip_c0 = next(row for row in rows if row["experiment"] == "C0_continue_L0")
        self.assertAlmostEqual(float(clip_c0["fid"]), 33.78459378302617, places=8)

        with (ROOT / "03_metrics_and_logs/dcgan_core/figure_audit.csv").open(newline="", encoding="utf-8") as handle:
            audit = list(csv.DictReader(handle))
        self.assertEqual(len(audit), 16)
        self.assertEqual(sum(row["source_disk_status"] == "present" for row in audit), 7)
        self.assertEqual(sum(row["source_disk_status"] == "intentionally_removed_per_owner_note" for row in audit), 7)
        self.assertEqual(sum(row["source_disk_status"] == "missing_current_source" for row in audit), 2)
        self.assertTrue(all((ROOT / row["public_snapshot_file"]).exists() for row in audit if row["public_snapshot_file"]))

        stage_figures = list((ROOT / "04_visual_assets/stage_figures").rglob("*.svg"))
        self.assertEqual(len(stage_figures), 34)
        for path in stage_figures:
            self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"), path)

        with (ROOT / "03_metrics_and_logs/stage_figures_map.csv").open(newline="", encoding="utf-8") as handle:
            stage_map = list(csv.DictReader(handle))
        self.assertEqual(len(stage_map), 34)
        self.assertTrue(all(row["status"] == "regenerated" for row in stage_map))

        deployment_figures = list((ROOT / "04_visual_assets/source_figures/deployment_quantization_service").glob("*.svg"))
        self.assertEqual(len(deployment_figures), 26)
        for path in deployment_figures:
            self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"), path)

    def test_sdxl_controlled_study_evidence_is_present(self):
        scripts = list((ROOT / "02_selected_experiments/full_process/phase7_sdxl_controlled_study").rglob("*.py"))
        self.assertGreaterEqual(len(scripts), 10)
        for group, expected_fid, expected_coverage in (
            ("A0", 37.91, 0.6687),
            ("A10", 37.99, 0.6525),
            ("A20", 41.58, 0.6108),
            ("A30", 44.92, 0.5423),
            ("A50", 49.94, 0.4397),
        ):
            with (ROOT / "03_metrics_and_logs/phase7_sdxl_controlled_study" / group / "metrics.json").open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertAlmostEqual(metrics["FID"], expected_fid, places=2)
            self.assertAlmostEqual(metrics["Coverage"], expected_coverage, places=4)
            self.assertTrue((ROOT / "04_visual_assets" / f"sdxl_{group.lower()}_epoch100.png").exists())
        self.assertTrue((ROOT / "04_visual_assets/sdxl_ratio_contact_sheet.png").exists())

    def test_deployment_evidence_and_staged_stress_are_present(self):
        with (ROOT / "03_metrics_and_logs/deployment_optimization/deployment_quantization_summary.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        mixed = next(row for row in rows if row["label"] == "Mixed_net0_net12")
        self.assertAlmostEqual(float(mixed["standard_fid"]), 31.177551, places=5)
        self.assertAlmostEqual(float(mixed["throughput_images_per_s"]), 23971.694, places=2)

        with (ROOT / "03_metrics_and_logs/deployment_optimization/06_Service_Stress/service_stress_summary.csv").open(newline="", encoding="utf-8") as handle:
            stress = list(csv.DictReader(handle))
        self.assertEqual(len(stress), 11)
        self.assertEqual(stress[-1]["concurrency"], "128")
        self.assertEqual(sum(int(row["failures"]) for row in stress), 0)
        self.assertAlmostEqual(float(stress[-1]["p99_ms"]), 490.0, places=2)

        with (ROOT / "03_metrics_and_logs/deployment_optimization/06_Service_Stress/service_monitor_summary.json").open(encoding="utf-8") as handle:
            monitor = json.load(handle)
        self.assertEqual(monitor["max_tested_concurrency"], 128)
        self.assertFalse(monitor["hard_crash_observed"])
        self.assertFalse(monitor["soak_run_included"])

    def test_current_service_freeze_evidence_is_present(self):
        with (ROOT / "03_metrics_and_logs/deployment_optimization/06_Service_Stress/06E/06BC_stage_resource_summary.csv").open(newline="", encoding="utf-8") as handle:
            fixed = list(csv.DictReader(handle))
        self.assertEqual(fixed[-1]["concurrency"], "512")
        self.assertEqual(sum(int(row["failures"]) for row in fixed), 0)
        self.assertAlmostEqual(float(fixed[-1]["p99_ms"]), 1600.0, places=2)

        with (ROOT / "03_metrics_and_logs/deployment_optimization/06_Service_Stress/06E/06D_soak_summary.csv").open(newline="", encoding="utf-8") as handle:
            soak = list(csv.DictReader(handle))
        steady = next(row for row in soak if row["phase"] == "soak_steady_u0016")
        self.assertEqual(steady["failures"], "0")
        self.assertEqual(steady["requests"], "1226890")
        self.assertAlmostEqual(float(steady["p99_ms"]), 63.0, places=2)

        dynamic_manifest = json.loads((ROOT / "03_metrics_and_logs/deployment_optimization/06_Service_Stress/06F/report/dynamic_batch_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(dynamic_manifest["execution_status"], "complete")
        self.assertEqual(dynamic_manifest["report_status"], "complete_with_packaging_gaps")


if __name__ == "__main__":
    unittest.main()
