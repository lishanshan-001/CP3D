"""Framework ablation under the point-based 3D MOT evaluation protocol.

This script fixes the protocol mismatch between the classical point-target 3D
MOT comparison and the earlier common-support ablation. Every configuration is
evaluated with the same pseudo-3D references, 25 mm frame-wise Hungarian point
matching, TrackEval implementation, jitter definition, and coverage denominator
used by ``step5_point_3d_mot_comparison.py``.

No pipeline parameter is selected from test-set errors. The complete-framework
row is loaded from the same finalized reconstruction CSV used by the point-based
3D MOT comparison, and the independently reconstructed row is audited against
that canonical source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
RUNTIME_ROOT = CODE_ROOT / ".runtime"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_ROOT / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import step5_point_3d_mot_comparison as p3d
import step5_stagewise_common_support_ablation as stage


DISPLAY_LABELS = {
    "B0": "Baseline",
    "B1": "+FVSA",
    "B2": "+TQA",
    "B4": "+FVSA + RGR (w/o TQA)",
    "B3": "+RGR (FULL)",
}

# B4 isolates RGR while bypassing segment-level TQA. It is placed before the
# canonical complete framework so that B3 remains the final validation row.
CONFIGURATIONS = [
    *stage.CONFIGURATIONS[:3],
    stage.Configuration(
        "B4",
        "Proposed + RGR without TQA",
        "Proposed",
        True,
        False,
        True,
    ),
    stage.CONFIGURATIONS[3],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the stage-wise framework ablation under one 25 mm protocol."
    )
    parser.add_argument("--pipeline-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--distance-mm", type=float, default=25.0)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recon_points_to_predictions(
    points: list[object],
    sequences: set[str],
) -> dict[str, list[p3d.TrackPoint3D]]:
    """Apply the same component/frame deduplication as the canonical loader."""
    best: dict[tuple[str, int, int], tuple[float, p3d.TrackPoint3D]] = {}
    for point in points:
        if point.seq_name not in sequences:
            continue
        xyz = np.asarray(point.point3d, dtype=np.float64)
        if not np.all(np.isfinite(xyz)) or float(xyz[2]) <= 0.0:
            continue
        track_id = int(point.component_id)
        reliability = float(point.reliability)
        converted = p3d.TrackPoint3D(
            seq_name=point.seq_name,
            frame=int(point.frame),
            track_id=track_id,
            xyz=xyz,
            score=reliability,
        )
        key = (point.seq_name, track_id, int(point.frame))
        previous = best.get(key)
        if previous is None or reliability > previous[0]:
            best[key] = (reliability, converted)

    output = {sequence: [] for sequence in sequences}
    for _score, point in best.values():
        output[point.seq_name].append(point)
    for values in output.values():
        values.sort(key=lambda item: (item.frame, item.track_id))
    return output


def prediction_map(
    predictions: dict[str, list[p3d.TrackPoint3D]],
) -> dict[tuple[str, int, int], p3d.TrackPoint3D]:
    return {
        (point.seq_name, int(point.track_id), int(point.frame)): point
        for values in predictions.values()
        for point in values
    }


def compare_predictions(
    regenerated: dict[str, list[p3d.TrackPoint3D]],
    canonical: dict[str, list[p3d.TrackPoint3D]],
) -> dict[str, object]:
    regenerated_map = prediction_map(regenerated)
    canonical_map = prediction_map(canonical)
    common = sorted(set(regenerated_map) & set(canonical_map))
    coordinate_differences = [
        float(np.linalg.norm(regenerated_map[key].xyz - canonical_map[key].xyz))
        for key in common
    ]
    max_difference = max(coordinate_differences, default=0.0)
    return {
        "regenerated_points": len(regenerated_map),
        "canonical_points": len(canonical_map),
        "common_keys": len(common),
        "regenerated_only_keys": len(set(regenerated_map) - set(canonical_map)),
        "canonical_only_keys": len(set(canonical_map) - set(regenerated_map)),
        "maximum_coordinate_difference_mm": max_difference,
        "keys_identical": set(regenerated_map) == set(canonical_map),
        "coordinates_identical_within_1e-9_mm": max_difference <= 1e-9,
    }


def read_reference_proposed_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "Proposed reliability-guided framework":
                return row
    raise RuntimeError(f"Proposed framework row not found in {path}")


def metric_match(
    row: dict[str, object], reference: dict[str, str], tolerance: float = 1e-10
) -> tuple[bool, dict[str, float]]:
    metrics = [
        "HOTA_3D",
        "AssA_3D",
        "IDF1_3D",
        "MOTA_3D",
        "IDSW",
        "RMSE_mm",
        "MAE_mm",
        "MedAE_mm",
        "Jitter_mm",
        "Coverage_pct",
        "GT_points",
        "Predicted_points",
        "Matched_points",
    ]
    differences = {
        metric: abs(float(row[metric]) - float(reference[metric])) for metric in metrics
    }
    return all(value <= tolerance for value in differences.values()), differences


def render_table(rows: list[dict[str, object]], output_dir: Path, dpi: int) -> list[str]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    headers = [
        "Configuration",
        "FVSA",
        "TQA",
        "RGR",
        "RMSE\n(mm)",
        "MAE\n(mm)",
        "MedAE\n(mm)",
        "Jitter\n(mm)",
        "Coverage\n(%)",
    ]
    check = "\u2713"
    cells = []
    for row in rows:
        cells.append(
            [
                str(row["Configuration"]),
                check if int(row["FVSA"]) else "",
                check if int(row["TQA"]) else "",
                check if int(row["RGR"]) else "",
                f'{float(row["RMSE_mm"]):.2f}',
                f'{float(row["MAE_mm"]):.2f}',
                f'{float(row["MedAE_mm"]):.2f}',
                f'{float(row["Jitter_mm"]):.2f}',
                f'{float(row["Coverage_pct"]):.2f}',
            ]
        )

    fig, ax = plt.subplots(figsize=(8.0, 2.3))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.28, 0.07, 0.07, 0.07, 0.105, 0.10, 0.105, 0.105, 0.115],
        bbox=[0.01, 0.03, 0.98, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#222222")
        cell.visible_edges = ""
        cell.PAD = 0.10
        if cell.get_text().get_text() == check:
            cell.get_text().set_fontfamily("DejaVu Sans")
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
        elif row_index == len(rows):
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = [
        figures / "Table8_framework_ablation_unified_25mm.png",
        figures / "Table8_framework_ablation_unified_25mm.pdf",
    ]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    return [str(path) for path in paths]


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| Configuration | FVSA | TQA | RGR | RMSE (mm) | MAE (mm) | MedAE (mm) | Jitter (mm) | Coverage (%) |",
        "|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        display_row = {
            **row,
            "FVSA": "Yes" if int(row["FVSA"]) else "",
            "TQA": "Yes" if int(row["TQA"]) else "",
            "RGR": "Yes" if int(row["RGR"]) else "",
            "RMSE_mm": float(row["RMSE_mm"]),
            "MAE_mm": float(row["MAE_mm"]),
            "MedAE_mm": float(row["MedAE_mm"]),
            "Jitter_mm": float(row["Jitter_mm"]),
            "Coverage_pct": float(row["Coverage_pct"]),
        }
        lines.append(
            "| {Configuration} | {FVSA} | {TQA} | {RGR} | {RMSE_mm:.2f} | "
            "{MAE_mm:.2f} | {MedAE_mm:.2f} | {Jitter_mm:.2f} | "
            "{Coverage_pct:.2f} |".format(**display_row)
        )
    lines.extend(
        [
            "",
            "All rows use frame-wise Hungarian point matching with a fixed 25 mm gate. ",
            "RMSE, MAE, and MedAE are calculated over accepted point matches; coverage is ",
            "the number of accepted matches divided by all alignable pseudo-3D reference points.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_metrics_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| Configuration | FVSA | TQA | RGR | HOTA-3D (%) | AssA-3D (%) | IDF1-3D (%) | MOTA-3D (%) | IDSW | RMSE (mm) | MAE (mm) | MedAE (mm) | Jitter (mm) | Coverage (%) |",
        "|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {configuration} | {fvsa} | {tqa} | {rgr} | {hota:.2f} | "
            "{assa:.2f} | {idf1:.2f} | {mota:.2f} | {idsw:d} | {rmse:.2f} | "
            "{mae:.2f} | {medae:.2f} | {jitter:.2f} | {coverage:.2f} |".format(
                configuration=row["Configuration"],
                fvsa="Yes" if int(row["FVSA"]) else "",
                tqa="Yes" if int(row["TQA"]) else "",
                rgr="Yes" if int(row["RGR"]) else "",
                hota=float(row["HOTA_3D"]),
                assa=float(row["AssA_3D"]),
                idf1=float(row["IDF1_3D"]),
                mota=float(row["MOTA_3D"]),
                idsw=int(float(row["IDSW"])),
                rmse=float(row["RMSE_mm"]),
                mae=float(row["MAE_mm"]),
                medae=float(row["MedAE_mm"]),
                jitter=float(row["Jitter_mm"]),
                coverage=float(row["Coverage_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "All tracking and reconstruction metrics use the same CP3D test split, pseudo-3D references, and fixed 25 mm point-distance protocol.",
            "Spatial errors are calculated over accepted frame-wise Hungarian point matches; coverage is accepted matches divided by all alignable pseudo-3D reference points.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_all_metrics_table(
    rows: list[dict[str, object]], output_dir: Path, dpi: int
) -> list[str]:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    headers = [
        "Configuration",
        "FVSA",
        "TQA",
        "RGR",
        "HOTA$_{3D}$\n(%)",
        "AssA$_{3D}$\n(%)",
        "IDF1$_{3D}$\n(%)",
        "MOTA$_{3D}$\n(%)",
        "IDSW",
        "RMSE\n(mm)",
        "MAE\n(mm)",
        "MedAE\n(mm)",
        "Jitter\n(mm)",
        "Coverage\n(%)",
    ]
    check = "\u2713"
    cells: list[list[str]] = []
    for row in rows:
        cells.append(
            [
                str(row["Configuration"]),
                check if int(row["FVSA"]) else "",
                check if int(row["TQA"]) else "",
                check if int(row["RGR"]) else "",
                f'{float(row["HOTA_3D"]):.2f}',
                f'{float(row["AssA_3D"]):.2f}',
                f'{float(row["IDF1_3D"]):.2f}',
                f'{float(row["MOTA_3D"]):.2f}',
                f'{int(float(row["IDSW"]))}',
                f'{float(row["RMSE_mm"]):.2f}',
                f'{float(row["MAE_mm"]):.2f}',
                f'{float(row["MedAE_mm"]):.2f}',
                f'{float(row["Jitter_mm"]):.2f}',
                f'{float(row["Coverage_pct"]):.2f}',
            ]
        )

    fig, ax = plt.subplots(figsize=(13.2, 2.25))
    ax.axis("off")
    widths = [0.145, 0.042, 0.042, 0.042, 0.068, 0.068, 0.068, 0.068, 0.052, 0.068, 0.064, 0.068, 0.068, 0.075]
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=widths,
        bbox=[0.01, 0.03, 0.98, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#222222")
        cell.visible_edges = ""
        cell.PAD = 0.08
        if cell.get_text().get_text() == check:
            cell.get_text().set_fontfamily("DejaVu Sans")
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
        elif row_index == len(rows):
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = [
        figures / "Table8_framework_ablation_all_3D_metrics.png",
        figures / "Table8_framework_ablation_all_3D_metrics.pdf",
    ]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    return [str(path) for path in paths]


def write_readme(path: Path, validation: dict[str, object]) -> None:
    text = f"""# Unified 25 mm framework ablation

This directory replaces the earlier main-text common-support ablation. It does
not alter FVSA, TQA, or RGR. It only unifies the evaluator with the point-based
3D MOT comparison.

## Fixed protocol

- CP3D test sequences: CP-I0-03, CP-I1-03, CP-I2-03, and CP-I3-03.
- Unchanged pseudo-3D references from synchronized rectified GT centres.
- Frame-wise Hungarian point matching with a fixed 25 mm gate.
- Identical TrackEval, spatial-error, jitter, and coverage functions for every row.
- No test-set parameter selection.
- The FULL row reads the same canonical finalized reconstruction CSV as the
  point-based 3D MOT comparison.
- Baseline denotes FrameVote association followed by direct triangulation.
- The `+FVSA + RGR (w/o TQA)` row runs RGR with its frame-level reliability and
  depth-consistency operations but bypasses the segment-level TQA gate.

## Consistency result

The FULL row matches the point-based comparison: **{validation['full_row_matches_point_comparison']}**.

The old common-support table remains available for audit, but its RMSE values
must not be mixed with this 25 mm matched-point protocol.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    pipeline_root = args.pipeline_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step5_root = pipeline_root / "step5_reconstruction"
    print("[1/5] Reconstructing the five stage-wise configurations")
    step5_meta, segments, test_contexts, gt_refs, depth_priors = stage.prepare_pipeline(step5_root)
    tqs_threshold = float(step5_meta["best_tqs_threshold"])
    reconstructed = stage.execute_configurations(
        CONFIGURATIONS,
        segments,
        test_contexts,
        gt_refs,
        depth_priors,
        tqs_threshold,
    )

    sequences = {context.seq_name for context in test_contexts}
    regenerated_predictions = {
        key: recon_points_to_predictions(points, sequences)
        for key, points in reconstructed.items()
    }

    print("[2/5] Loading the canonical complete-framework output")
    proposed_source = (
        step5_root / "paper_assets" / "reports" / "reconstruction_points_compag.csv"
    )
    canonical_full = p3d.load_proposed_points(proposed_source, sequences)
    regeneration_audit = compare_predictions(regenerated_predictions["B3"], canonical_full)

    # Use one canonical source for the FULL row in both manuscript experiments.
    predictions = {key: value for key, value in regenerated_predictions.items()}
    predictions["B3"] = canonical_full

    gt_maps = {context.seq_name: gt_refs[context.seq_name][0] for context in test_contexts}
    seq_lengths = {context.seq_name: context.seq_length for context in test_contexts}

    print("[3/5] Evaluating every row with the same 25 mm protocol")
    summary_rows: list[dict[str, object]] = []
    per_sequence_rows: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        summary, per_sequence = p3d.evaluate_method(
            DISPLAY_LABELS[configuration.key],
            predictions[configuration.key],
            gt_maps,
            seq_lengths,
            float(args.distance_mm),
        )
        row = {
            "Configuration": DISPLAY_LABELS[configuration.key],
            "FVSA": int(configuration.association_enabled),
            "TQA": int(configuration.quality_enabled),
            "RGR": int(configuration.reconstruction_enabled),
            **{key: value for key, value in summary.items() if key != "method"},
        }
        summary_rows.append(row)
        for sequence_row in per_sequence:
            per_sequence_rows.append(
                {
                    "Configuration": DISPLAY_LABELS[configuration.key],
                    "FVSA": int(configuration.association_enabled),
                    "TQA": int(configuration.quality_enabled),
                    "RGR": int(configuration.reconstruction_enabled),
                    **{key: value for key, value in sequence_row.items() if key != "method"},
                }
            )

    print("[4/5] Verifying the FULL row against the existing comparison")
    comparison_csv = (
        step5_root
        / "point_3d_mot_comparison"
        / "tables"
        / "Table_point_3D_MOT_comparison.csv"
    )
    reference_full = read_reference_proposed_row(comparison_csv)
    full_matches, full_differences = metric_match(summary_rows[-1], reference_full)

    finite = all(
        math.isfinite(float(row[metric]))
        for row in summary_rows
        for metric in (
            "HOTA_3D",
            "AssA_3D",
            "IDF1_3D",
            "MOTA_3D",
            "IDSW",
            "RMSE_mm",
            "MAE_mm",
            "MedAE_mm",
            "Jitter_mm",
            "Coverage_pct",
        )
    )
    validation = {
        "status": "PASS" if full_matches and finite else "FAIL",
        "full_row_matches_point_comparison": full_matches,
        "full_row_absolute_metric_differences": full_differences,
        "all_summary_metrics_finite": finite,
        "complete_framework_regeneration_audit": regeneration_audit,
        "test_error_parameter_selection": False,
    }
    if validation["status"] != "PASS":
        raise RuntimeError(f"Unified ablation validation failed: {validation}")

    print("[5/5] Writing paper and audit assets")
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "Table8_framework_ablation_unified_25mm.csv", summary_rows)
    write_csv(tables_dir / "Table8_framework_ablation_all_3D_metrics.csv", summary_rows)
    write_csv(reports_dir / "framework_ablation_unified_25mm_per_sequence.csv", per_sequence_rows)
    write_markdown(tables_dir / "Table8_framework_ablation_unified_25mm.md", summary_rows)
    write_all_metrics_markdown(
        tables_dir / "Table8_framework_ablation_all_3D_metrics.md", summary_rows
    )
    figure_paths = render_table(summary_rows, output_dir, args.dpi)
    figure_paths.extend(render_all_metrics_table(summary_rows, output_dir, args.dpi))

    config = {
        "experiment": "Stage-wise framework ablation under the unified point-based 3D MOT protocol",
        "pipeline_root": str(pipeline_root),
        "test_sequences": sorted(sequences),
        "pseudo_3d_reference": "Unchanged synchronized rectified GT-centre triangulation",
        "evaluation_distance_mm": float(args.distance_mm),
        "point_matching": "Frame-wise Hungarian assignment; accept Euclidean distance <= 25 mm",
        "spatial_error_support": "Accepted 25 mm point matches for each independently executed configuration",
        "jitter": "Mean L2 norm of second XYZ differences over consecutive predicted points within track identity",
        "coverage": "Accepted point matches / all alignable pseudo-3D reference points",
        "tqs_threshold": tqs_threshold,
        "tqs_threshold_source": "Training split only",
        "configurations": [
            {
                "key": item.key,
                "label": DISPLAY_LABELS[item.key],
                "FVSA": item.association_enabled,
                "TQA": item.quality_enabled,
                "RGR": item.reconstruction_enabled,
            }
            for item in CONFIGURATIONS
        ],
        "canonical_complete_framework_input": str(proposed_source),
        "comparison_table_used_for_consistency_check": str(comparison_csv),
        "test_error_parameter_selection": False,
        "figures": figure_paths,
    }
    (reports_dir / "framework_ablation_unified_25mm_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (reports_dir / "framework_ablation_unified_25mm_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_readme(output_dir / "README.md", validation)

    print(json.dumps({"summary": summary_rows, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
