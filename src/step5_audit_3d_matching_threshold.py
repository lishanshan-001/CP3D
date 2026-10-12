"""Audit the point-distance evaluation protocol without changing predictions.

This script is deliberately evaluation-only. It loads the frozen 3D prediction
CSV produced by ``step5_point_3d_mot_comparison.py`` and rebuilds the unchanged
annotation-derived pseudo-3D references. No detector, tracker, FVSA, or HRTR
parameter is trained or selected here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
RUNTIME_ROOT = CODE_ROOT / ".runtime"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_ROOT / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.metrics import CLEAR, HOTA, Identity

import step4_stereo_association_eval as s4
import step5_point_3d_mot_comparison as p3d
import step5_reliability_guided_3d_reconstruction as s5


DEFAULT_THRESHOLDS_MM = (15.0, 20.0, 25.0, 30.0, 35.0)
CURRENT_PROTOCOL = "current_anchored_continuous"
FIXED_BINARY_PROTOCOL = "scheme_A_fixed_binary"
BOUNDED_CONTINUOUS_PROTOCOL = "scheme_B_bounded_continuous"
PROTOCOLS = (CURRENT_PROTOCOL, FIXED_BINARY_PROTOCOL, BOUNDED_CONTINUOUS_PROTOCOL)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def excel_column(index: int) -> str:
    result = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_simple_xlsx(path: Path, sheets: list[tuple[str, list[dict[str, object]], list[str]]]) -> None:
    """Write a small standards-compliant XLSX using only the Python standard library."""
    ensure_dir(path.parent)
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    workbook_sheets = []
    workbook_rels = []
    sheet_xml: list[str] = []
    for sheet_index, (name, rows, fields) in enumerate(sheets, start=1):
        safe_name = name[:31].replace("&", "and")
        workbook_sheets.append(f'<sheet name="{escape(safe_name)}" sheetId="{sheet_index}" r:id="rId{sheet_index}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{sheet_index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{sheet_index}.xml"/>'
        )
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{sheet_index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        all_rows: list[list[object]] = [fields]
        all_rows.extend([[row.get(field, "") for field in fields] for row in rows])
        xml_rows = []
        for row_index, values in enumerate(all_rows, start=1):
            cells = []
            for column_index, value in enumerate(values):
                reference = f"{excel_column(column_index)}{row_index}"
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    cells.append(f'<c r="{reference}"><v>{value}</v></c>')
                else:
                    cells.append(
                        f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
                    )
            xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
        sheet_xml.append(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
        )
    content_types.append("</Types>")
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rel_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(workbook_rels)}</Relationships>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rel_xml)
        for index, xml in enumerate(sheet_xml, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", xml)


def load_frozen_predictions(path: Path) -> dict[str, dict[str, list[p3d.TrackPoint3D]]]:
    output: dict[str, dict[str, list[p3d.TrackPoint3D]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            point = p3d.TrackPoint3D(
                seq_name=row["sequence"],
                frame=int(row["frame"]),
                track_id=int(row["track_id"]),
                xyz=np.asarray([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])], dtype=np.float64),
                score=float(row["score"]),
            )
            output[row["method"]][row["sequence"]].append(point)
    for by_sequence in output.values():
        for points in by_sequence.values():
            points.sort(key=lambda item: (item.frame, item.track_id))
    return {method: dict(by_sequence) for method, by_sequence in output.items()}


def similarity_from_distance(distances: np.ndarray, tolerance_mm: float, protocol: str) -> np.ndarray:
    if protocol == CURRENT_PROTOCOL:
        return np.maximum(0.0, 1.0 - distances / (2.0 * tolerance_mm))
    if protocol == FIXED_BINARY_PROTOCOL:
        return (distances <= tolerance_mm).astype(np.float64)
    if protocol == BOUNDED_CONTINUOUS_PROTOCOL:
        return np.maximum(0.0, 1.0 - distances / tolerance_mm)
    raise ValueError(f"Unknown protocol: {protocol}")


def build_trackeval_data(
    gt_frames: dict[int, list[tuple[int, np.ndarray]]],
    predictions: list[p3d.TrackPoint3D],
    seq_length: int,
    tolerance_mm: float,
    protocol: str,
) -> dict[str, object]:
    pred_frames: dict[int, list[p3d.TrackPoint3D]] = defaultdict(list)
    for point in predictions:
        pred_frames[point.frame].append(point)
    gt_values = sorted({gt_id for items in gt_frames.values() for gt_id, _xyz in items})
    pred_values = sorted({point.track_id for point in predictions})
    gt_remap = {value: index for index, value in enumerate(gt_values)}
    pred_remap = {value: index for index, value in enumerate(pred_values)}
    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    similarities: list[np.ndarray] = []
    for frame in range(1, seq_length + 1):
        gt_items = gt_frames.get(frame, [])
        pred_items = pred_frames.get(frame, [])
        gt_ids.append(np.asarray([gt_remap[item[0]] for item in gt_items], dtype=np.int64))
        tracker_ids.append(np.asarray([pred_remap[item.track_id] for item in pred_items], dtype=np.int64))
        if not gt_items or not pred_items:
            similarities.append(np.zeros((len(gt_items), len(pred_items)), dtype=np.float64))
            continue
        gt_xyz = np.asarray([item[1] for item in gt_items], dtype=np.float64)
        pred_xyz = np.asarray([item.xyz for item in pred_items], dtype=np.float64)
        distances = np.linalg.norm(gt_xyz[:, None, :] - pred_xyz[None, :, :], axis=2)
        similarities.append(similarity_from_distance(distances, tolerance_mm, protocol))
    return {
        "num_timesteps": seq_length,
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
        "num_gt_dets": int(sum(len(items) for items in gt_ids)),
        "num_tracker_dets": int(sum(len(items) for items in tracker_ids)),
        "num_gt_ids": len(gt_values),
        "num_tracker_ids": len(pred_values),
    }


def evaluate_tracking_protocol(
    predictions_by_sequence: dict[str, list[p3d.TrackPoint3D]],
    gt_maps: dict[str, dict[int, dict[int, np.ndarray]]],
    sequence_lengths: dict[str, int],
    tolerance_mm: float,
    protocol: str,
) -> dict[str, float]:
    hota_metric = HOTA()
    identity_metric = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    clear_metric = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    hota_results = {}
    identity_results = {}
    clear_results = {}
    for sequence in sorted(gt_maps):
        data = build_trackeval_data(
            p3d.gt_by_frame(gt_maps[sequence]),
            predictions_by_sequence.get(sequence, []),
            sequence_lengths[sequence],
            tolerance_mm,
            protocol,
        )
        hota_results[sequence] = hota_metric.eval_sequence(data)
        identity_results[sequence] = identity_metric.eval_sequence(data)
        clear_results[sequence] = clear_metric.eval_sequence(data)
    hota = hota_metric.combine_sequences(hota_results)
    identity = identity_metric.combine_sequences(identity_results)
    clear = clear_metric.combine_sequences(clear_results)
    alpha_index = int(np.where(np.isclose(hota_metric.array_labels, 0.5))[0][0])
    return {
        "HOTA3D_pct": 100.0 * float(np.mean(hota["HOTA"])),
        "AssA3D_pct": 100.0 * float(np.mean(hota["AssA"])),
        "HOTA3D_at_alpha_0_5_pct": 100.0 * float(hota["HOTA"][alpha_index]),
        "AssA3D_at_alpha_0_5_pct": 100.0 * float(hota["AssA"][alpha_index]),
        "IDF13D_pct": 100.0 * float(identity["IDF1"]),
        "MOTA3D_pct": 100.0 * float(clear["MOTA"]),
        "IDSW": int(clear["IDSW"]),
        "CLR_TP": int(clear["CLR_TP"]),
        "CLR_FP": int(clear["CLR_FP"]),
        "CLR_FN": int(clear["CLR_FN"]),
    }


def gate_aware_spatial_errors(
    gt_frames: dict[int, list[tuple[int, np.ndarray]]],
    predictions: list[p3d.TrackPoint3D],
    tolerance_mm: float,
) -> tuple[list[float], int]:
    """Maximize valid cardinality first, then minimize distance within the gate."""
    pred_frames: dict[int, list[p3d.TrackPoint3D]] = defaultdict(list)
    for point in predictions:
        pred_frames[point.frame].append(point)
    errors: list[float] = []
    matches = 0
    invalid_cost = 1e9
    for frame, gt_items in gt_frames.items():
        pred_items = pred_frames.get(frame, [])
        if not gt_items or not pred_items:
            continue
        gt_xyz = np.asarray([item[1] for item in gt_items], dtype=np.float64)
        pred_xyz = np.asarray([item.xyz for item in pred_items], dtype=np.float64)
        distances = np.linalg.norm(gt_xyz[:, None, :] - pred_xyz[None, :, :], axis=2)
        gated_cost = np.where(distances <= tolerance_mm, distances, invalid_cost)
        rows, cols = linear_sum_assignment(gated_cost)
        for row, col in zip(rows.tolist(), cols.tolist()):
            if float(gated_cost[row, col]) >= invalid_cost:
                continue
            errors.append(float(distances[row, col]))
            matches += 1
    return errors, matches


def train_tolerance_statistics(
    train_sequences: Iterable[str],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    consecutive_motion: list[float] = []
    nearest_identity_spacing: list[float] = []
    sequence_rows = []
    for sequence in train_sequences:
        gt_map = s5.build_gt_reference(sequence)[0]
        sequence_motion: list[float] = []
        sequence_spacing: list[float] = []
        frame_points: dict[int, list[np.ndarray]] = defaultdict(list)
        for frames in gt_map.values():
            ordered = sorted(frames.items())
            for (frame_a, point_a), (frame_b, point_b) in zip(ordered, ordered[1:]):
                if frame_b == frame_a + 1 and np.all(np.isfinite(point_a)) and np.all(np.isfinite(point_b)):
                    sequence_motion.append(float(np.linalg.norm(point_b - point_a)))
            for frame, point in frames.items():
                if np.all(np.isfinite(point)) and float(point[2]) > 0.0:
                    frame_points[frame].append(point)
        for points in frame_points.values():
            if len(points) < 2:
                continue
            array = np.asarray(points, dtype=np.float64)
            distances = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            sequence_spacing.extend(np.min(distances, axis=1).tolist())
        consecutive_motion.extend(sequence_motion)
        nearest_identity_spacing.extend(sequence_spacing)
        sequence_rows.append(
            {
                "sequence": sequence,
                "consecutive_motion_samples": len(sequence_motion),
                "consecutive_motion_p95_mm": float(np.percentile(sequence_motion, 95)),
                "nearest_identity_samples": len(sequence_spacing),
                "nearest_identity_spacing_p01_mm": float(np.percentile(sequence_spacing, 1)),
            }
        )
    summary = {
        "consecutive_motion_samples": len(consecutive_motion),
        "consecutive_motion_p95_mm": float(np.percentile(consecutive_motion, 95)),
        "nearest_identity_samples": len(nearest_identity_spacing),
        "nearest_identity_spacing_p01_mm": float(np.percentile(nearest_identity_spacing, 1)),
    }
    return sequence_rows, summary


def make_line_figure(
    rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
    path_stem: Path,
    dpi: int,
) -> None:
    colors = ["#1B6CA8", "#E07A1F", "#4C956C", "#C43D3D"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    methods = list(dict.fromkeys(str(row["Method"]) for row in rows))
    for method, color, marker in zip(methods, colors, markers):
        selected = sorted(
            (row for row in rows if row["Method"] == method),
            key=lambda row: float(row["Threshold_mm"]),
        )
        ax.plot(
            [float(row["Threshold_mm"]) for row in selected],
            [float(row[metric]) for row in selected],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5.2,
            label=method.replace(" tracking", "").replace(" reliability-guided framework", " framework"),
        )
    ax.set_xlabel("3D matching tolerance (mm)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({float(row["Threshold_mm"]) for row in rows}))
    ax.grid(True, color="#D7D7D7", linewidth=0.6, alpha=0.75)
    ax.legend(frameon=False, fontsize=8.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def metric_rank_stability(rows: list[dict[str, object]]) -> dict[str, object]:
    proposed = p3d.METHODS[-1]
    directions = {
        "HOTA3D_pct": "max",
        "AssA3D_pct": "max",
        "IDF13D_pct": "max",
        "MOTA3D_pct": "max",
        "IDSW": "min",
        "RMSE_mm": "min",
        "Coverage_pct": "max",
    }
    result: dict[str, object] = {}
    for metric, direction in directions.items():
        ranks = []
        best_flags = []
        for threshold in sorted({float(row["Threshold_mm"]) for row in rows}):
            selected = [row for row in rows if float(row["Threshold_mm"]) == threshold]
            ordered = sorted(selected, key=lambda row: float(row[metric]), reverse=direction == "max")
            rank = next(index + 1 for index, row in enumerate(ordered) if row["Method"] == proposed)
            ranks.append(rank)
            best_flags.append(rank == 1)
        result[metric] = {
            "proposed_ranks": ranks,
            "proposed_best_at_all_thresholds": all(best_flags),
        }
    return result


def format_threshold(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def write_reports(
    output_root: Path,
    rows: list[dict[str, object]],
    protocol_rows: list[dict[str, object]],
    assignment_rows: list[dict[str, object]],
    train_summary: dict[str, float],
    rank_stability: dict[str, object],
) -> None:
    thresholds = sorted({float(row["Threshold_mm"]) for row in rows})
    proposed = p3d.METHODS[-1]
    proposed_rows = [row for row in rows if row["Method"] == proposed]
    coverage_monotonic = all(
        float(a["Coverage_pct"]) <= float(b["Coverage_pct"]) + 1e-12
        for a, b in zip(proposed_rows, proposed_rows[1:])
    )
    assignment_differences = sum(int(row["Additional_gate_aware_matches"]) for row in assignment_rows)
    old_match_count = sum(int(row["Legacy_post_gate_matches"]) for row in assignment_rows)
    proposed_by_threshold = {
        float(row["Threshold_mm"]): row for row in rows if row["Method"] == proposed
    }
    central_rows = [row for row in rows if 20.0 <= float(row["Threshold_mm"]) <= 30.0]
    central_stability = metric_rank_stability(central_rows)
    identity_stable_20_30 = all(
        bool(central_stability[key]["proposed_best_at_all_thresholds"])
        for key in ("AssA3D_pct", "IDF13D_pct", "IDSW")
    )
    central_identity_statement = (
        "retained the best AssA3D, IDF13D, and IDSW at every tolerance"
        if identity_stable_20_30
        else "did not retain the best AssA3D, IDF13D, and IDSW at every tolerance"
    )
    protocol_25 = {
        str(row["Protocol"]): row
        for row in protocol_rows
        if row["Method"] == proposed and math.isclose(float(row["Threshold_mm"]), 25.0)
    }
    nearest_idf1_35 = max(
        float(row["IDF13D_pct"])
        for row in rows
        if math.isclose(float(row["Threshold_mm"]), 35.0)
    )
    report = f"""# Audit of the 3D point-distance evaluation protocol

## 1. Existing evaluation protocol

The published comparison code defines `EVAL_DISTANCE_MM = 25.0` at line 54 of
`step5_point_3d_mot_comparison.py`. It is passed to `build_trackeval_data`,
`spatial_error_terms`, `evaluate_one_sequence`, and `evaluate_method`. It is an
evaluation tolerance and is not an FVSA, HRTR, detector, or tracker parameter.
The original comparison script hard-codes it and provides no command-line or
configuration override. Later table-generation scripts expose an independent
`--distance-mm` argument, but that does not make the original comparison value
configurable.

The actual TrackEval similarity is

`S = max(0, 1 - d / (2 * tolerance))`.

Therefore, Identity and CLEAR at TrackEval threshold 0.5 accept pairs with
`d <= tolerance`. HOTA is evaluated by TrackEval at 19 alpha values from 0.05
to 0.95 and then averaged. Consequently, the reported HOTA3D is not a single
binary 25 mm HOTA score. It is the official TrackEval HOTA computation applied
to a task-specific continuous point-distance similarity, with 25 mm serving as
the alpha=0.5 anchor. At HOTA alpha 0.05, the current similarity can admit
distances up to 1.9 times the tolerance; at alpha 0.95, it admits distances up
to 0.1 times the tolerance.

RMSE, MAE, MedAE, and Coverage use a separate matcher: the full Euclidean
distance matrix is assigned first by Hungarian optimization and assignments
greater than the tolerance are rejected afterwards. This post-gating is not
mathematically identical to gate-aware maximum-cardinality assignment.
Jitter is independent of this evaluation tolerance because it is calculated
only from predicted trajectories. HOTA3D, AssA3D, IDF13D, MOTA3D, IDSW,
Coverage, RMSE, MAE, and MedAE are all tolerance-dependent, directly or through
the accepted correspondence set.

Current spatial-error pseudocode:

```text
D = pairwise_euclidean_distances(reference_points, predicted_points)
rows, cols = Hungarian(D)
accepted = [(r, c) for (r, c) in zip(rows, cols) if D[r, c] <= tolerance]
errors = [D[r, c] for (r, c) in accepted]
coverage = len(accepted) / number_of_reference_points
```

## 2. Interpretation of 25 mm

The 25 mm value is **not a universal 3D MOT standard**. For Identity, CLEAR,
and spatial-error reporting, it is a task-specific point-distance tolerance;
for HOTA it is the alpha=0.5 distance anchor. The training references independently
reproduce the recorded rationale: consecutive-motion P95 =
{train_summary['consecutive_motion_p95_mm']:.6f} mm from
{int(train_summary['consecutive_motion_samples'])} samples, and nearest-identity
spacing P01 = {train_summary['nearest_identity_spacing_p01_mm']:.6f} mm from
{int(train_summary['nearest_identity_samples'])} samples. These statistics were
computed only from the eight training sequences.

The old comparison file stored these two values as numeric literals rather
than computing them, so the rationale was reproducible from the data but not
fully traceable from that script alone. This audit supplies the missing
calculation. No code or dated record proves that 25 mm was optimized on a
development set. The manuscript must not claim development-set optimization.
The defensible statement is that it was fixed using training-reference spatial
statistics and then audited on the test set for sensitivity only.

## 3. Sensitivity analysis

The frozen predictions and unchanged pseudo-3D references were reevaluated at
{', '.join(format_threshold(value) for value in thresholds)} mm. No model was
retrained and no FVSA, HRTR, stereo-matcher, or tracker parameter was changed.

Within the central 20-30 mm interval, the proposed framework
{central_identity_statement}. Across the
full 15-35 mm interval, it retained the best AssA3D and lowest IDSW throughout.
At 35 mm its IDF13D ({float(proposed_by_threshold[35.0]['IDF13D_pct']):.3f}%)
was slightly below the best result ({nearest_idf1_35:.3f}%). It did not achieve
the highest HOTA3D or Coverage, and its MOTA advantage was limited to the
strictest tolerance. The robust conclusion is therefore identity preservation,
not universal superiority across all metrics.

For the proposed method, Coverage was {'monotonic' if coverage_monotonic else 'not monotonic'}
over the tested tolerances. Absolute spatial-error values must be interpreted
together with Coverage because each tolerance changes the accepted point set.

The gate-aware assignment audit recovered {assignment_differences} additional
valid matches out of {old_match_count} legacy accepted matches when summed over
all methods and thresholds. Each affected method-threshold combination differed
by one point, so this issue does not alter the ranking, although the two
definitions are not mathematically identical. See
`spatial_assignment_audit.csv`; existing results were not overwritten.

## 4. HOTA protocol audit

Three HOTA inputs were evaluated:

1. Current anchored continuous similarity: `max(0, 1 - d/(2*tolerance))`.
2. Scheme A fixed binary similarity: `1[d <= tolerance]`.
3. Scheme B bounded continuous similarity: `max(0, 1 - d/tolerance)`.

All three call the same TrackEval HOTA implementation and average the same 19
alpha thresholds. They differ only in the localization similarity supplied to
TrackEval. Scheme B treats the stated tolerance as the maximum positive-
similarity distance, whereas the current formulation treats it as the
alpha=0.5 distance. The numerical comparison is in
`hota_protocol_comparison.csv`.

At 25 mm, the proposed framework obtained HOTA3D =
{float(protocol_25[CURRENT_PROTOCOL]['HOTA3D_pct']):.3f}% with the current
anchored-continuous similarity, {float(protocol_25[FIXED_BINARY_PROTOCOL]['HOTA3D_pct']):.3f}%
with Scheme A, and {float(protocol_25[BOUNDED_CONTINUOUS_PROTOCOL]['HOTA3D_pct']):.3f}%
with Scheme B. These values are not interchangeable: absolute HOTA depends
materially on the supplied point-localization similarity.

## 5. Recommended manuscript treatment

- Do not call 25 mm a standard or an optimized model parameter.
- Describe it as a task-specific point-distance matching tolerance fixed from
  training-reference spatial statistics before test reporting.
- Keep the 15-35 mm analysis in Supplementary Material because it is a
  robustness audit, not test-set threshold selection.
- State explicitly that HOTA3D uses TrackEval with a custom point-distance
  similarity. Do not describe the current HOTA3D as one binary 25 mm score.
- Retaining the current TrackEval adaptation is acceptable only if 25 mm is
  described as the alpha=0.5 distance anchor for HOTA, not its maximum
  admissible distance. If the manuscript claims that no match beyond 25 mm
  contributes to HOTA, all HOTA3D and AssA3D tables must be recomputed
  consistently with Scheme A or Scheme B.
- Do not silently replace the original tables. A protocol change requires all
  affected HOTA3D/AssA3D results to be regenerated consistently.
- Consider replacing assign-then-reject RMSE/Coverage matching with gate-aware
  assignment in a future manuscript revision only after explicitly versioning
  the protocol; the present audit preserves both results.

## 6. Direct answers

1. **What is 25 mm?** An evaluation-only, task-specific spatial tolerance and
   the alpha=0.5 anchor of the current TrackEval point similarity.
2. **Is HOTA3D standard?** The HOTA algorithm is the official TrackEval
   implementation, but its localization input is a custom point-distance
   adaptation. It is not an unchanged box-IoU HOTA protocol.
3. **Should 25 mm be changed?** No test-based change is justified. It lies
   between the reproduced 22.32 mm motion P95 and 27.92 mm nearest-identity
   spacing P01 and within a stable 20-30 mm identity-preservation interval.
4. **Does sensitivity support the conclusions?** It supports the AssA3D/IDSW
   advantage and the IDF13D advantage over 20-30 mm. It does not support a
   claim that the proposed method is best in HOTA3D, MOTA3D, or Coverage at
   every tolerance.
5. **Main text or Supplementary?** Supplementary Table S3 and Fig. S1 are the
   appropriate location; the main text needs only the fixed protocol and a
   concise robustness statement if supported.
6. **Section 2.5.2 wording?** Use the supplied `manuscript_methods_english.md`;
   avoid calling 25 mm optimal or universal.
"""
    (output_root / "threshold_analysis_report.md").write_text(report, encoding="utf-8")

    methods_text = f"""# Manuscript-ready evaluation description

For point-based 3D MOT evaluation, each insect was represented by its
reconstructed 3D position rather than a volumetric bounding box. Predicted
positions and annotation-derived pseudo-3D references were compared independently
in each frame using Euclidean distance. A task-specific distance tolerance of
25 mm was fixed using spatial statistics calculated from
the training references: the 95th percentile of consecutive-frame displacement
was {train_summary['consecutive_motion_p95_mm']:.2f} mm, while the first percentile
of nearest-neighbour inter-individual spacing was
{train_summary['nearest_identity_spacing_p01_mm']:.2f} mm. The tolerance was used
to reject spatially inconsistent point correspondences and was not a parameter
of FVSA or HRTR. Identity and CLEAR metrics were computed with TrackEval using a
point-distance similarity whose 0.5 level corresponded to 25 mm. HOTA was
computed over TrackEval's standard set of 19 alpha levels using the same
task-specific continuous point-distance similarity; thus, HOTA3D represents a
point-target adaptation of HOTA rather than an unchanged 3D bounding-box
protocol. Sensitivity to neighbouring matching tolerances was assessed without
changing predictions or model parameters. The proposed method retained the
highest AssA3D and lowest IDSW over 15-35 mm and the highest IDF13D over
20-30 mm (Supplementary Table S3 and Fig. S1).
"""
    (output_root / "manuscript_methods_english.md").write_text(methods_text, encoding="utf-8")

    supplementary = f"""# Supplementary results text

Sensitivity of the point-based 3D evaluation to the matching-distance tolerance
was examined from {format_threshold(thresholds[0])} to {format_threshold(thresholds[-1])} mm while retaining identical predictions,
pseudo-3D references, and model parameters. Absolute tracking and spatial-error
metrics varied with the tolerance because broader gates accepted additional
correspondences and changed the support on which RMSE, MAE, and MedAE were
calculated. The proposed framework retained the highest AssA3D and lowest IDSW
at all five tolerances and the highest IDF13D throughout 20-30 mm. It did not
consistently maximize HOTA3D, MOTA3D, or Coverage. The identity-preservation
conclusion was therefore stable around 25 mm, whereas universal superiority was
not supported. This is a robustness analysis, not threshold optimization;
25 mm remains a task-specific tolerance supported by training-reference
spatial statistics rather than an optimal or universal 3D MOT threshold.
"""
    (output_root / "supplementary_analysis_english.md").write_text(supplementary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit 3D point-matching tolerance sensitivity.")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--thresholds-mm", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS_MM))
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    pipeline_root = Path(args.pipeline_root).resolve()
    data_root = Path(args.data_root).resolve()
    comparison_root = Path(args.comparison_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    thresholds = sorted(set(float(value) for value in args.thresholds_mm))
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("All matching tolerances must be positive.")

    predictions_path = comparison_root / "reports" / "point_3D_MOT_predictions.csv"
    comparison_config_path = comparison_root / "comparison_config.json"
    metadata_path = pipeline_root / "step4_association" / "run_metadata.json"
    for path in (predictions_path, comparison_config_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    comparison_config = json.loads(comparison_config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_sequences = list(metadata["train_sequences"])
    test_sequences = list(metadata["test_sequences"])

    s4.configure_input_roots(
        Path(metadata["step3_train_root"]),
        Path(metadata["step3_test_root"]),
        data_root,
        pipeline_root / "step1_rectification",
    )
    s5.configure_step1_root(pipeline_root / "step1_rectification")

    print("[1/6] Loading frozen predictions")
    predictions = load_frozen_predictions(predictions_path)
    if list(predictions) != p3d.METHODS:
        raise AssertionError(f"Unexpected method order: {list(predictions)}")
    print("[2/6] Rebuilding unchanged pseudo-3D references")
    gt_maps = {sequence: s5.build_gt_reference(sequence)[0] for sequence in test_sequences}
    sequence_lengths = {sequence: int(s4.base.read_seqinfo(sequence)["seqLength"]) for sequence in test_sequences}

    print("[3/6] Recomputing training-only tolerance rationale")
    train_rows, train_summary = train_tolerance_statistics(train_sequences)
    write_csv(
        output_root / "train_tolerance_rationale.csv",
        train_rows + [{"sequence": "POOLED", **train_summary}],
        [
            "sequence",
            "consecutive_motion_samples",
            "consecutive_motion_p95_mm",
            "nearest_identity_samples",
            "nearest_identity_spacing_p01_mm",
        ],
    )

    print("[4/6] Running frozen-prediction tolerance sensitivity")
    all_rows: list[dict[str, object]] = []
    per_sequence_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    for tolerance in thresholds:
        print(f"  tolerance={tolerance:g} mm")
        for method in p3d.METHODS:
            summary, per_sequence = p3d.evaluate_method(
                method,
                predictions[method],
                gt_maps,
                sequence_lengths,
                tolerance,
            )
            current = evaluate_tracking_protocol(
                predictions[method], gt_maps, sequence_lengths, tolerance, CURRENT_PROTOCOL
            )
            row = {
                "Threshold_mm": tolerance,
                "Method": method,
                "HOTA3D_pct": float(summary["HOTA_3D"]),
                "AssA3D_pct": float(summary["AssA_3D"]),
                "HOTA3D_at_alpha_0_5_pct": current["HOTA3D_at_alpha_0_5_pct"],
                "AssA3D_at_alpha_0_5_pct": current["AssA3D_at_alpha_0_5_pct"],
                "IDF13D_pct": float(summary["IDF1_3D"]),
                "MOTA3D_pct": float(summary["MOTA_3D"]),
                "IDSW": int(summary["IDSW"]),
                "RMSE_mm": float(summary["RMSE_mm"]),
                "MAE_mm": float(summary["MAE_mm"]),
                "MedAE_mm": float(summary["MedAE_mm"]),
                "Jitter_mm": float(summary["Jitter_mm"]),
                "Coverage_pct": float(summary["Coverage_pct"]),
                "Matched_observations": int(summary["Matched_points"]),
                "Unmatched_predictions": int(summary["Predicted_points"] - summary["Matched_points"]),
                "Unmatched_references": int(summary["GT_points"] - summary["Matched_points"]),
                "Predicted_observations": int(summary["Predicted_points"]),
                "Reference_observations": int(summary["GT_points"]),
            }
            all_rows.append(row)
            for sequence_row in per_sequence:
                per_sequence_rows.append({"Threshold_mm": tolerance, **sequence_row})

            legacy_errors: list[float] = []
            gate_errors: list[float] = []
            legacy_matches = 0
            gate_matches = 0
            total_gt = 0
            for sequence in test_sequences:
                gt_frames = p3d.gt_by_frame(gt_maps[sequence])
                old_errors, old_matches = p3d.spatial_error_terms(
                    gt_frames, predictions[method].get(sequence, []), tolerance
                )
                new_errors, new_matches = gate_aware_spatial_errors(
                    gt_frames, predictions[method].get(sequence, []), tolerance
                )
                legacy_errors.extend(old_errors)
                gate_errors.extend(new_errors)
                legacy_matches += old_matches
                gate_matches += new_matches
                total_gt += sum(len(items) for items in gt_frames.values())
            assignment_rows.append(
                {
                    "Threshold_mm": tolerance,
                    "Method": method,
                    "Legacy_post_gate_matches": legacy_matches,
                    "Gate_aware_matches": gate_matches,
                    "Additional_gate_aware_matches": gate_matches - legacy_matches,
                    "Legacy_RMSE_mm": float(np.sqrt(np.mean(np.square(legacy_errors)))),
                    "Gate_aware_RMSE_mm": float(np.sqrt(np.mean(np.square(gate_errors)))),
                    "Legacy_Coverage_pct": 100.0 * legacy_matches / total_gt,
                    "Gate_aware_Coverage_pct": 100.0 * gate_matches / total_gt,
                }
            )

            for protocol in PROTOCOLS:
                metrics = evaluate_tracking_protocol(
                    predictions[method], gt_maps, sequence_lengths, tolerance, protocol
                )
                protocol_rows.append(
                    {
                        "Threshold_mm": tolerance,
                        "Method": method,
                        "Protocol": protocol,
                        **metrics,
                    }
                )

    metric_fields = [
        "Threshold_mm",
        "Method",
        "HOTA3D_pct",
        "AssA3D_pct",
        "HOTA3D_at_alpha_0_5_pct",
        "AssA3D_at_alpha_0_5_pct",
        "IDF13D_pct",
        "MOTA3D_pct",
        "IDSW",
        "RMSE_mm",
        "MAE_mm",
        "MedAE_mm",
        "Jitter_mm",
        "Coverage_pct",
        "Matched_observations",
        "Unmatched_predictions",
        "Unmatched_references",
        "Predicted_observations",
        "Reference_observations",
    ]
    protocol_fields = ["Threshold_mm", "Method", "Protocol"] + [
        key for key in protocol_rows[0] if key not in {"Threshold_mm", "Method", "Protocol"}
    ]
    assignment_fields = list(assignment_rows[0])
    per_sequence_fields = list(per_sequence_rows[0])
    write_csv(output_root / "threshold_metrics_all.csv", all_rows, metric_fields)
    write_csv(output_root / "threshold_metrics_per_sequence.csv", per_sequence_rows, per_sequence_fields)
    write_csv(output_root / "hota_protocol_comparison.csv", protocol_rows, protocol_fields)
    write_csv(output_root / "spatial_assignment_audit.csv", assignment_rows, assignment_fields)
    write_csv(output_root / "Table_S3.csv", all_rows, metric_fields)
    table_lines = [
        "# Table S3. Sensitivity of point-based 3D tracking performance to the matching-distance tolerance on the CP3D test set.",
        "",
        "| Tolerance (mm) | Method | HOTA3D (%) | AssA3D (%) | IDF13D (%) | MOTA3D (%) | IDSW | RMSE (mm) | MAE (mm) | MedAE (mm) | Coverage (%) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        table_lines.append(
            f"| {float(row['Threshold_mm']):.0f} | {row['Method']} | "
            f"{float(row['HOTA3D_pct']):.2f} | {float(row['AssA3D_pct']):.2f} | "
            f"{float(row['IDF13D_pct']):.2f} | {float(row['MOTA3D_pct']):.2f} | "
            f"{int(row['IDSW'])} | {float(row['RMSE_mm']):.2f} | "
            f"{float(row['MAE_mm']):.2f} | {float(row['MedAE_mm']):.2f} | "
            f"{float(row['Coverage_pct']):.2f} |"
        )
    table_lines.extend(
        [
            "",
            "Note: Predictions, pseudo-3D references, and all FVSA/HRTR parameters were fixed. "
            "The tolerance is an evaluation protocol variable, not a model parameter.",
        ]
    )
    (output_root / "Table_S3.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    write_simple_xlsx(
        output_root / "threshold_metrics_summary.xlsx",
        [
            ("Table S3", all_rows, metric_fields),
            ("Per sequence", per_sequence_rows, per_sequence_fields),
            ("HOTA protocols", protocol_rows, protocol_fields),
            ("Assignment audit", assignment_rows, assignment_fields),
        ],
    )

    print("[5/6] Rendering supplementary figures")
    make_line_figure(all_rows, "HOTA3D_pct", "HOTA3D (%)", output_root / "Fig_S1a_HOTA3D", args.dpi)
    make_line_figure(all_rows, "IDF13D_pct", "IDF13D (%)", output_root / "Fig_S1b_IDF13D", args.dpi)
    make_line_figure(all_rows, "Coverage_pct", "Coverage (%)", output_root / "Fig_S1c_Coverage", args.dpi)

    print("[6/6] Writing audit report and provenance")
    rank_stability = metric_rank_stability(all_rows)
    write_reports(
        output_root,
        all_rows,
        protocol_rows,
        assignment_rows,
        train_summary,
        rank_stability,
    )
    provenance = {
        "purpose": "Evaluation-only 3D matching-tolerance sensitivity and protocol audit",
        "thresholds_mm": thresholds,
        "frozen_predictions": str(predictions_path),
        "frozen_predictions_sha256": file_sha256(predictions_path),
        "comparison_config": str(comparison_config_path),
        "comparison_config_sha256": file_sha256(comparison_config_path),
        "step4_metadata": str(metadata_path),
        "step4_metadata_sha256": file_sha256(metadata_path),
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "methods": p3d.METHODS,
        "original_evaluation_tolerance_mm": float(comparison_config["evaluation_distance_mm"]),
        "predictions_regenerated": False,
        "references_regenerated_from_unchanged_gt_and_projection_matrices": True,
        "model_or_pipeline_parameters_changed": False,
        "current_similarity": "max(0, 1 - distance / (2 * tolerance))",
        "fixed_binary_similarity": "1 if distance <= tolerance else 0",
        "bounded_continuous_similarity": "max(0, 1 - distance / tolerance)",
        "trackeval_hota_alpha_values": HOTA().array_labels.tolist(),
        "trackeval_identity_clear_threshold": 0.5,
        "spatial_error_legacy_matching": "Hungarian on full Euclidean distance matrix, then reject distance > tolerance",
        "spatial_error_gate_aware_audit": "Set distance > tolerance to a large cost, maximize valid cardinality, then minimize distance",
        "training_tolerance_statistics": train_summary,
        "rank_stability": rank_stability,
    }
    (output_root / "audit_config.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Audit outputs: {output_root}")


if __name__ == "__main__":
    main()
