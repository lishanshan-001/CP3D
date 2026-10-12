# -*- coding: utf-8 -*-

"""Validate corrected Step 5 reconstruction metrics and physical plausibility."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CORE_METHODS = ["RawStereo3D", "GeomGate3D", "TQSFiltered3D", "ReliabilityGuided3D"]
PAPER_METHODS = [
    "Direct triangulation",
    "Uniform smoothing",
    "Reliability-guided refinement",
]
CHAMBER_DIMS_MM = np.asarray([2400.0, 1000.0, 1200.0], dtype=np.float64)
CHAMBER_DIAGONAL_MM = float(np.linalg.norm(CHAMBER_DIMS_MM))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step5-root", required=True)
    parser.add_argument("--save-root", required=True)
    return parser.parse_args()


def read_mot_keys(path: Path) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.split(",")
            keys.add((int(float(parts[0])), int(float(parts[1]))))
    return keys


def count_test_reference_keys(metadata: dict[str, object]) -> int:
    step3_root = Path(str(metadata["step3_test_root"]))
    step4_meta = json.loads(
        (Path(str(metadata["step4_root"])) / "run_metadata.json").read_text(encoding="utf-8")
    )
    total = 0
    for sequence in step4_meta["test_sequences"]:
        left = read_mot_keys(
            step3_root
            / "gt"
            / "mot_challenge"
            / "CP3D-left-all"
            / sequence
            / "gt"
            / "gt.txt"
        )
        right = read_mot_keys(
            step3_root
            / "gt"
            / "mot_challenge"
            / "CP3D-right-all"
            / sequence
            / "gt"
            / "gt.txt"
        )
        total += len(left & right)
    return total


def deduplicate_reference_points(group: pd.DataFrame) -> pd.DataFrame:
    valid = group.dropna(subset=["gt_x_mm", "gt_y_mm", "gt_z_mm"]).copy()
    valid = valid.sort_values(
        ["seq_name", "gt_id", "frame", "reliability"],
        ascending=[True, True, True, False],
    )
    return valid.drop_duplicates(["seq_name", "gt_id", "frame"], keep="first")


def compute_jitter(group: pd.DataFrame) -> float:
    terms: list[float] = []
    for _key, segment in group.groupby(["seq_name", "component_id", "segment_id"]):
        ordered = segment.sort_values("frame")
        xyz = ordered[["x_mm", "y_mm", "z_mm"]].to_numpy(np.float64)
        if len(xyz) >= 3:
            acceleration = np.diff(xyz, n=2, axis=0)
            if acceleration.size:
                terms.extend(np.linalg.norm(acceleration, axis=1).tolist())
    return float(np.mean(terms)) if terms else float("nan")


def metric_row(group: pd.DataFrame, denominator: int) -> dict[str, float]:
    unique = deduplicate_reference_points(group)
    pred = unique[["x_mm", "y_mm", "z_mm"]].to_numpy(np.float64)
    ref = unique[["gt_x_mm", "gt_y_mm", "gt_z_mm"]].to_numpy(np.float64)
    errors = np.linalg.norm(pred - ref, axis=1)
    return {
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "mae_mm": float(np.mean(errors)),
        "medae_mm": float(np.median(errors)),
        "jitter_mm": compute_jitter(group),
        "coverage_pct": 100.0 * len(unique) / max(1, denominator),
        "evaluated_unique_points": int(len(unique)),
    }


def physical_rows(frame: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    method_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    for method, group in frame.groupby("method", sort=False):
        xyz = group[["x_mm", "y_mm", "z_mm"]].to_numpy(np.float64)
        method_rows.append(
            {
                "method": method,
                "points": len(group),
                "finite_fraction": float(np.isfinite(xyz).all(axis=1).mean()),
                "nonpositive_z_points": int(np.sum(xyz[:, 2] <= 0)),
                "x_min_mm": float(np.min(xyz[:, 0])),
                "x_max_mm": float(np.max(xyz[:, 0])),
                "y_min_mm": float(np.min(xyz[:, 1])),
                "y_max_mm": float(np.max(xyz[:, 1])),
                "z_min_mm": float(np.min(xyz[:, 2])),
                "z_max_mm": float(np.max(xyz[:, 2])),
            }
        )
        for key, segment in group.groupby(["seq_name", "component_id", "segment_id"]):
            segment_xyz = segment[["x_mm", "y_mm", "z_mm"]].to_numpy(np.float64)
            spans = np.ptp(segment_xyz, axis=0)
            span_diagonal = float(np.linalg.norm(spans))
            segment_rows.append(
                {
                    "method": method,
                    "seq_name": key[0],
                    "component_id": int(key[1]),
                    "segment_id": int(key[2]),
                    "points": len(segment),
                    "x_span_mm": float(spans[0]),
                    "y_span_mm": float(spans[1]),
                    "z_span_mm": float(spans[2]),
                    "bounding_span_diagonal_mm": span_diagonal,
                    "exceeds_chamber_diagonal": int(span_diagonal > CHAMBER_DIAGONAL_MM),
                }
            )
    return method_rows, segment_rows


def compare_metric(
    calculated: float,
    reported: float,
    label: str,
    tolerance: float = 1e-8,
) -> None:
    if not math.isclose(calculated, reported, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"Metric mismatch for {label}: {calculated} vs {reported}")


def main() -> None:
    args = parse_args()
    root = Path(args.step5_root).resolve()
    save_root = Path(args.save_root).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    for key in ("step4_root", "step3_train_root", "step3_test_root", "step1_rectification_root"):
        input_path = Path(str(metadata[key])).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Recorded input path does not exist for {key}: {input_path}")
    parameters = metadata["reliability_guided_parameters"]
    expected_parameters = {
        "frame_reliability_threshold": 0.45,
        "temporal_radius_frames": 2,
        "temporal_decay": 1.5,
        "raw_fusion_weight": "0.55 + 0.35 * frame reliability",
        "short_gap_completion_max_gap_frames": 3,
    }
    for key, expected in expected_parameters.items():
        if parameters[key] != expected:
            raise AssertionError(f"Unexpected reconstruction parameter {key}: {parameters[key]}")

    denominator = count_test_reference_keys(metadata)
    core_points = pd.read_csv(root / "reports" / "reconstruction_points.csv")
    if set(core_points["method"].unique()) != set(CORE_METHODS):
        raise AssertionError("Unexpected core reconstruction method set.")
    core_report = {
        row["method"]: row
        for row in csv.DictReader(
            (root / "reports" / "reconstruction_metrics.csv").open("r", encoding="utf-8")
        )
    }
    calculated_core: list[dict[str, object]] = []
    for method, group in core_points.groupby("method", sort=False):
        calculated = metric_row(group, denominator)
        reported = core_report[method]
        for metric in ("rmse_mm", "mae_mm", "medae_mm", "jitter_mm"):
            compare_metric(calculated[metric], float(reported[metric]), f"{method}/{metric}")
        compare_metric(
            calculated["coverage_pct"],
            100.0 * float(reported["coverage"]),
            f"{method}/coverage",
        )
        calculated_core.append({"method": method, **calculated})

    paper_points = pd.read_csv(root / "paper_assets" / "reports" / "reconstruction_points_compag.csv")
    if set(paper_points["method"].unique()) != set(PAPER_METHODS):
        raise AssertionError("Unexpected paper reconstruction method set.")
    paper_report = {
        row["method"]: row
        for row in csv.DictReader(
            (
                root
                / "paper_assets"
                / "tables"
                / "Table_3D_reconstruction_comparison.csv"
            ).open("r", encoding="utf-8")
        )
    }
    calculated_paper: list[dict[str, object]] = []
    for method, group in paper_points.groupby("method", sort=False):
        calculated = metric_row(group, denominator)
        reported = paper_report[method]
        for metric in ("rmse_mm", "mae_mm", "medae_mm", "jitter_mm", "coverage_pct"):
            compare_metric(calculated[metric], float(reported[metric]), f"{method}/{metric}")
        calculated_paper.append({"method": method, **calculated})

    method_physical, segment_physical = physical_rows(core_points)
    if any(float(row["finite_fraction"]) != 1.0 for row in method_physical):
        raise AssertionError("Non-finite reconstructed coordinates were found.")
    if any(int(row["nonpositive_z_points"]) != 0 for row in method_physical):
        raise AssertionError("Nonpositive depth was found.")
    if any(int(row["exceeds_chamber_diagonal"]) for row in segment_physical):
        raise AssertionError("At least one segment exceeds the chamber diagonal.")

    tqs_rows = list(
        csv.DictReader(
            (root / "reports" / "quality_strategy_metrics.csv").open("r", encoding="utf-8")
        )
    )
    proposed_tqs = next(row for row in tqs_rows if row["method"] == "ProposedTQS")
    tqs_supported = (
        float(proposed_tqs["spearman_rho"]) <= -0.2
        and float(proposed_tqs["f1"]) >= max(float(row["f1"]) for row in tqs_rows if row["method"] != "ProposedTQS")
    )

    pd.DataFrame(calculated_core).to_csv(save_root / "recomputed_core_metrics.csv", index=False)
    pd.DataFrame(calculated_paper).to_csv(save_root / "recomputed_paper_metrics.csv", index=False)
    pd.DataFrame(method_physical).to_csv(save_root / "physical_validity_by_method.csv", index=False)
    pd.DataFrame(segment_physical).to_csv(save_root / "physical_validity_by_segment.csv", index=False)

    checks = {
        "recorded_input_paths_exist": True,
        "reconstruction_parameters_match_final_algorithm": True,
        "test_reference_denominator": denominator,
        "reported_metrics_recomputed_exactly": True,
        "coverage_uses_unique_test_reference_keys": True,
        "all_coordinates_finite": True,
        "all_depths_positive": True,
        "no_segment_exceeds_chamber_diagonal": True,
        "tqs_validation_supports_manuscript_claim": bool(tqs_supported),
    }
    report = {
        "checks": checks,
        "chamber_dimensions_mm": CHAMBER_DIMS_MM.tolist(),
        "chamber_diagonal_mm": CHAMBER_DIAGONAL_MM,
        "core_metrics": calculated_core,
        "paper_metrics": calculated_paper,
        "physical_summary": method_physical,
        "tqs_proposed_test_result": proposed_tqs,
        "interpretation": {
            "physical_scope": (
                "The chamber-diagonal check assesses within-segment displacement only. "
                "Absolute chamber placement cannot be verified without a camera-to-chamber transform."
            ),
            "pseudo_3d_scope": (
                "Pseudo-3D references are triangulated from rectified ground-truth centers and "
                "measure internal reconstruction consistency, not independent metrological accuracy."
            ),
            "tqs_scope": (
                "The corrected experiment does not support a strong TQS discrimination claim "
                "when tqs_validation_supports_manuscript_claim is false."
            ),
        },
    }
    (save_root / "step5_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
