"""Evaluate trajectory-derived movement descriptors on common temporal support.

This experiment is evaluation-only. It uses the frozen predictions from the
point-based 3D MOT comparison and the unchanged annotation-derived pseudo-3D
references. No detector, tracker, FVSA, HRTR, or evaluation parameter is tuned.

For every test sequence, predictions are matched to pseudo-3D reference points
frame by frame using the existing 25 mm Hungarian point-distance protocol.
Descriptor errors are then computed only on identity-consistent, temporally
continuous intervals that are available for all four methods.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
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
from scipy.optimize import linear_sum_assignment

import step4_stereo_association_eval as s4
import step5_point_3d_mot_comparison as p3d
import step5_reliability_guided_3d_reconstruction as s5


METHOD_LABELS = {
    "Nearest-neighbour 3D tracking": "NN-3D",
    "Euclidean 3D Kalman-Hungarian": "Euclidean KH",
    "Mahalanobis 3D Kalman-Hungarian": "Mahalanobis KH",
    "Proposed reliability-guided framework": "FVSA-HRTR",
}
METHODS = list(METHOD_LABELS)
ERROR_FIELDS = [
    "path_length_relative_error_pct",
    "mean_turning_angle_absolute_error_deg",
    "trajectory_straightness_absolute_error",
]
COVERAGE_FIELD = "longest_identity_consistent_coverage_pct"
DEFAULT_DISTANCE_MM = 25.0
DEFAULT_MIN_INTERVAL_FRAMES = 3
EPSILON = 1e-9


@dataclass(frozen=True)
class MatchedPoint:
    track_id: int
    xyz: np.ndarray
    distance_mm: float


@dataclass(frozen=True)
class DescriptorSet:
    path_length_mm: float
    duration_s: float
    mean_turning_angle_deg: float
    trajectory_straightness: float


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


def load_predictions(path: Path) -> dict[str, dict[str, list[p3d.TrackPoint3D]]]:
    output: dict[str, dict[str, list[p3d.TrackPoint3D]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in METHOD_LABELS:
                continue
            point = p3d.TrackPoint3D(
                seq_name=row["sequence"],
                frame=int(row["frame"]),
                track_id=int(row["track_id"]),
                xyz=np.asarray(
                    [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])],
                    dtype=np.float64,
                ),
                score=float(row["score"]),
            )
            if np.all(np.isfinite(point.xyz)):
                output[method][point.seq_name].append(point)
    for by_sequence in output.values():
        for points in by_sequence.values():
            points.sort(key=lambda item: (item.frame, item.track_id))
    return {method: dict(by_sequence) for method, by_sequence in output.items()}


def framewise_matches(
    gt_frames: dict[int, list[tuple[int, np.ndarray]]],
    predictions: list[p3d.TrackPoint3D],
    distance_gate_mm: float,
) -> dict[int, dict[int, MatchedPoint]]:
    """Reproduce the existing full-Hungarian-then-gate spatial protocol."""
    pred_frames: dict[int, list[p3d.TrackPoint3D]] = defaultdict(list)
    for point in predictions:
        pred_frames[point.frame].append(point)

    output: dict[int, dict[int, MatchedPoint]] = defaultdict(dict)
    for frame, gt_items in gt_frames.items():
        pred_items = pred_frames.get(frame, [])
        if not gt_items or not pred_items:
            continue
        gt_xyz = np.asarray([xyz for _gt_id, xyz in gt_items], dtype=np.float64)
        pred_xyz = np.asarray([point.xyz for point in pred_items], dtype=np.float64)
        distances = np.linalg.norm(gt_xyz[:, None, :] - pred_xyz[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(distances)
        for row, col in zip(rows.tolist(), cols.tolist()):
            distance = float(distances[row, col])
            if distance > distance_gate_mm:
                continue
            gt_id = int(gt_items[row][0])
            point = pred_items[col]
            output[gt_id][frame] = MatchedPoint(
                track_id=int(point.track_id),
                xyz=point.xyz.copy(),
                distance_mm=distance,
            )
    return {gt_id: dict(frames) for gt_id, frames in output.items()}


def common_intervals(
    reference: dict[int, dict[int, np.ndarray]],
    matches: dict[str, dict[int, dict[int, MatchedPoint]]],
    min_interval_frames: int,
) -> list[tuple[int, list[int]]]:
    """Find frame runs shared by all methods without stitching across ID changes."""
    intervals: list[tuple[int, list[int]]] = []
    for gt_id in sorted(reference):
        common = set(reference[gt_id])
        for method in METHODS:
            common &= set(matches[method].get(gt_id, {}))
        ordered = sorted(common)
        if not ordered:
            continue

        run = [ordered[0]]
        for frame in ordered[1:]:
            previous = run[-1]
            contiguous = frame == previous + 1
            same_tracks = all(
                matches[method][gt_id][frame].track_id
                == matches[method][gt_id][previous].track_id
                for method in METHODS
            )
            if contiguous and same_tracks:
                run.append(frame)
                continue
            if len(run) >= min_interval_frames:
                intervals.append((gt_id, run))
            run = [frame]
        if len(run) >= min_interval_frames:
            intervals.append((gt_id, run))
    return intervals


def longest_identity_coverage_rows(
    sequence: str,
    reference: dict[int, dict[int, np.ndarray]],
    matches: dict[str, dict[int, dict[int, MatchedPoint]]],
) -> list[dict[str, object]]:
    """Measure the longest uninterrupted match to one predicted identity."""
    rows: list[dict[str, object]] = []
    for gt_id in sorted(reference):
        reference_frames = sorted(
            frame
            for frame, xyz in reference[gt_id].items()
            if np.all(np.isfinite(xyz)) and float(xyz[2]) > 0.0
        )
        if not reference_frames:
            continue
        for method in METHODS:
            method_matches = matches[method].get(gt_id, {})
            longest = 0
            current = 0
            previous_frame: int | None = None
            previous_track_id: int | None = None
            matched_frames = 0
            for frame in reference_frames:
                matched = method_matches.get(frame)
                if matched is None:
                    current = 0
                    previous_frame = None
                    previous_track_id = None
                    continue
                matched_frames += 1
                if (
                    previous_frame is not None
                    and frame == previous_frame + 1
                    and matched.track_id == previous_track_id
                ):
                    current += 1
                else:
                    current = 1
                longest = max(longest, current)
                previous_frame = frame
                previous_track_id = matched.track_id
            rows.append(
                {
                    "sequence": sequence,
                    "gt_id": gt_id,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "reference_frames": len(reference_frames),
                    "matched_frames": matched_frames,
                    "longest_identity_consistent_frames": longest,
                    COVERAGE_FIELD: 100.0 * longest / len(reference_frames),
                }
            )
    return rows


def movement_descriptors(xyz: np.ndarray, frames: np.ndarray, fps: float) -> DescriptorSet | None:
    """Compute the four movement descriptors on one continuous 3D interval."""
    if len(xyz) < 3 or len(xyz) != len(frames) or fps <= 0:
        return None
    differences = np.diff(xyz, axis=0)
    step_lengths = np.linalg.norm(differences, axis=1)
    path_length = float(np.sum(step_lengths))
    duration = float((frames[-1] - frames[0]) / fps)
    if path_length <= EPSILON or duration <= EPSILON:
        return None

    angles: list[float] = []
    for previous, current in zip(differences[:-1], differences[1:]):
        previous_norm = float(np.linalg.norm(previous))
        current_norm = float(np.linalg.norm(current))
        if previous_norm <= EPSILON or current_norm <= EPSILON:
            continue
        cosine = float(np.dot(previous, current) / (previous_norm * current_norm))
        angles.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
    if not angles:
        return None

    net_displacement = float(np.linalg.norm(xyz[-1] - xyz[0]))
    return DescriptorSet(
        path_length_mm=path_length,
        duration_s=duration,
        mean_turning_angle_deg=float(np.mean(angles)),
        trajectory_straightness=net_displacement / path_length,
    )


def descriptor_errors(predicted: DescriptorSet, reference: DescriptorSet) -> dict[str, float]:
    if reference.path_length_mm <= EPSILON:
        raise ValueError("Relative path-length error requires a positive reference path length.")
    return {
        "path_length_relative_error_pct": 100.0
        * abs(predicted.path_length_mm - reference.path_length_mm)
        / reference.path_length_mm,
        "mean_turning_angle_absolute_error_deg": abs(
            predicted.mean_turning_angle_deg - reference.mean_turning_angle_deg
        ),
        "trajectory_straightness_absolute_error": abs(
            predicted.trajectory_straightness - reference.trajectory_straightness
        ),
    }


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def add_distribution_summary(
    summary: dict[str, object], field: str, values: np.ndarray
) -> None:
    summary[f"{field}_mean"] = float(np.mean(values))
    summary[f"{field}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    summary[f"{field}_median"] = float(np.median(values))
    summary[f"{field}_q1"] = percentile(values, 25.0)
    summary[f"{field}_q3"] = percentile(values, 75.0)


def summarize_errors(
    rows: list[dict[str, object]], coverage_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {
            "method": METHOD_LABELS[method],
            "method_internal": method,
            "n_intervals": len(method_rows),
        }
        for field in ERROR_FIELDS:
            values = np.asarray([float(row[field]) for row in method_rows], dtype=np.float64)
            add_distribution_summary(summary, field, values)
        method_coverage = [row for row in coverage_rows if row["method"] == method]
        coverage_values = np.asarray(
            [float(row[COVERAGE_FIELD]) for row in method_coverage], dtype=np.float64
        )
        summary["n_reference_trajectories"] = len(method_coverage)
        add_distribution_summary(summary, COVERAGE_FIELD, coverage_values)
        summaries.append(summary)
    return summaries


def configure_plot_style() -> None:
    available = {item.name for item in matplotlib.font_manager.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_error_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    field: str,
    ylabel: str,
    panel_label: str | None,
    seed: int,
    method_order: list[str] | None = None,
) -> None:
    displayed_methods = method_order if method_order is not None else METHODS
    grouped = [
        np.asarray([float(row[field]) for row in rows if row["method"] == method], dtype=np.float64)
        for method in displayed_methods
    ]
    box = ax.boxplot(
        grouped,
        positions=np.arange(1, len(METHODS) + 1),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#b42318", "linewidth": 1.3},
        whiskerprops={"color": "#4b5563", "linewidth": 0.9},
        capprops={"color": "#4b5563", "linewidth": 0.9},
        boxprops={"edgecolor": "#374151", "linewidth": 0.9},
    )
    method_colors = dict(zip(METHODS, ["#87b5d8", "#7fc6ad", "#e8b46a", "#d9685a"]))
    for patch, method in zip(box["boxes"], displayed_methods):
        patch.set_facecolor(method_colors[method])
        patch.set_alpha(0.72)

    rng = np.random.default_rng(seed)
    for index, values in enumerate(grouped, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=7,
            color="#4b5563",
            alpha=0.22,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    ax.set_xticks(
        np.arange(1, len(displayed_methods) + 1),
        [METHOD_LABELS[method] for method in displayed_methods],
    )
    ax.tick_params(axis="x", rotation=12)
    ax.set_ylabel(ylabel)
    if field == "path_length_relative_error_pct":
        ax.set_yscale("log")
    if field == COVERAGE_FIELD:
        ax.set_ylim(-3.0, 103.0)
    ax.grid(axis="y", color="#d1d5db", linestyle=":", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    if panel_label:
        ax.text(0.01, 1.02, panel_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)


def save_figures(
    output_root: Path,
    error_rows: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    dpi: int,
) -> None:
    configure_plot_style()
    specifications = [
        (error_rows, ERROR_FIELDS[0], "Path length relative error (%)", "(a)"),
        (coverage_rows, COVERAGE_FIELD, "Longest identity-consistent coverage (%)", "(b)"),
        (error_rows, ERROR_FIELDS[1], "Mean turning-angle absolute error (deg)", "(c)"),
        (error_rows, ERROR_FIELDS[2], "Trajectory straightness absolute error", "(d)"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for index, (ax, (rows, field, ylabel, label)) in enumerate(zip(axes.flat, specifications)):
        draw_error_panel(ax, rows, field, ylabel, label, seed=20260823 + index)
    figure.savefig(output_root / "Figure13_movement_descriptor_errors.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_root / "Figure13_movement_descriptor_errors.pdf", bbox_inches="tight")
    plt.close(figure)

    for index, (rows, field, ylabel, _label) in enumerate(specifications):
        figure, ax = plt.subplots(figsize=(3.55, 2.75), constrained_layout=True)
        draw_error_panel(ax, rows, field, ylabel, None, seed=20260823 + index)
        suffix = chr(ord("a") + index)
        figure.savefig(output_root / f"Figure13{suffix}_clean.png", dpi=dpi, bbox_inches="tight")
        figure.savefig(output_root / f"Figure13{suffix}_clean.pdf", bbox_inches="tight")
        plt.close(figure)


def save_three_descriptor_figures(
    output_root: Path,
    error_rows: list[dict[str, object]],
    dpi: int,
) -> None:
    """Render the requested three-panel descriptor-only publication figure."""
    configure_plot_style()
    specifications = [
        (ERROR_FIELDS[0], "Path length relative error (%)", "(a)"),
        (ERROR_FIELDS[1], "Mean turning-angle absolute error (deg)", "(b)"),
        (ERROR_FIELDS[2], "Trajectory straightness absolute error", "(c)"),
    ]
    method_order = [METHODS[3], METHODS[1], METHODS[2], METHODS[0]]
    order_audit: list[dict[str, object]] = []
    for position, method in enumerate(method_order, start=1):
        method_rows = [row for row in error_rows if row["method"] == method]
        order_audit.append(
            {
                "plot_position": position,
                "method": METHOD_LABELS[method],
                "path_length_relative_error_median_pct": float(
                    np.median([float(row[ERROR_FIELDS[0]]) for row in method_rows])
                ),
                "turning_angle_absolute_error_median_deg": float(
                    np.median([float(row[ERROR_FIELDS[1]]) for row in method_rows])
                ),
                "straightness_absolute_error_median": float(
                    np.median([float(row[ERROR_FIELDS[2]]) for row in method_rows])
                ),
            }
        )
    write_csv(
        output_root / "Figure13_three_descriptor_plot_order_audit.csv",
        order_audit,
        list(order_audit[0]),
    )
    variants = {
        "annotated": True,
        "clean": False,
    }
    for variant, include_panel_labels in variants.items():
        figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.05), constrained_layout=True)
        for index, (ax, (field, ylabel, label)) in enumerate(zip(axes, specifications)):
            draw_error_panel(
                ax,
                error_rows,
                field,
                ylabel,
                label if include_panel_labels else None,
                seed=20260823 + index,
                method_order=method_order,
            )
        stem = f"Figure13_three_descriptor_panels_{variant}"
        figure.savefig(output_root / f"{stem}.png", dpi=dpi, bbox_inches="tight")
        figure.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
        explicit_stem = f"Figure13_three_descriptor_data_and_labels_swapped_{variant}"
        figure.savefig(output_root / f"{explicit_stem}.png", dpi=dpi, bbox_inches="tight")
        figure.savefig(output_root / f"{explicit_stem}.pdf", bbox_inches="tight")
        plt.close(figure)


def save_table_summary_three_panel_figures(
    output_root: Path,
    summaries: list[dict[str, object]],
    dpi: int,
) -> None:
    """Plot the exact Table 11 medians and interquartile ranges."""
    configure_plot_style()
    summary_lookup = {str(row["method"]): row for row in summaries}
    method_order = ["FVSA-HRTR", "Euclidean KH", "Mahalanobis KH", "NN-3D"]
    display_labels = [
        "Proposed\nframework",
        "Euclidean\nKalman-Hungarian",
        "Mahalanobis\nKalman-Hungarian",
        "Nearest-neighbor\n3D",
    ]
    method_colors = ["#d9685a", "#7fc6ad", "#e8b46a", "#87b5d8"]
    specifications = [
        (ERROR_FIELDS[0], "Path length relative error (%)", "(a)"),
        (ERROR_FIELDS[1], "Turning-angle absolute error (deg)", "(b)"),
        (ERROR_FIELDS[2], "Straightness absolute error", "(c)"),
    ]

    for variant, include_panel_labels in {"annotated": True, "clean": False}.items():
        figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)
        for ax, (field, ylabel, panel_label) in zip(axes, specifications):
            medians = np.asarray(
                [float(summary_lookup[method][field + "_median"]) for method in method_order]
            )
            q1 = np.asarray(
                [float(summary_lookup[method][field + "_q1"]) for method in method_order]
            )
            q3 = np.asarray(
                [float(summary_lookup[method][field + "_q3"]) for method in method_order]
            )
            positions = np.arange(1, len(method_order) + 1)
            for position, median, lower, upper, color in zip(
                positions, medians, q1, q3, method_colors
            ):
                ax.errorbar(
                    position,
                    median,
                    yerr=np.asarray([[median - lower], [upper - median]]),
                    fmt="o",
                    markersize=6.0,
                    markerfacecolor=color,
                    markeredgecolor="#263238",
                    markeredgewidth=0.75,
                    ecolor=color,
                    elinewidth=2.0,
                    capsize=5.0,
                    capthick=1.4,
                    zorder=3,
                )
            ax.set_xticks(positions, display_labels)
            ax.tick_params(axis="x", labelsize=7.2, pad=3)
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.55, 4.45)
            ax.set_ylim(bottom=0.0)
            ax.grid(axis="y", color="#d1d5db", linestyle=":", linewidth=0.65, alpha=0.8)
            ax.set_axisbelow(True)
            if include_panel_labels:
                ax.text(
                    0.01,
                    1.02,
                    panel_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                )
        stem = f"Figure13_table_values_median_IQR_{variant}"
        figure.savefig(output_root / f"{stem}.png", dpi=dpi, bbox_inches="tight")
        figure.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
        plt.close(figure)


def save_table_consistent_boxplots(
    output_root: Path,
    error_rows: list[dict[str, object]],
    dpi: int,
) -> None:
    """Draw standard boxplots whose medians and IQRs reproduce Table 11."""
    configure_plot_style()
    method_order = [METHODS[3], METHODS[1], METHODS[2], METHODS[0]]
    display_labels = [
        "Proposed\nframework",
        "Euclidean\nKalman-Hungarian",
        "Mahalanobis\nKalman-Hungarian",
        "Nearest-neighbor\n3D",
    ]
    method_colors = dict(zip(METHODS, ["#87b5d8", "#7fc6ad", "#e8b46a", "#d9685a"]))
    specifications = [
        (ERROR_FIELDS[0], "Path length relative error (%)", "(a)"),
        (ERROR_FIELDS[1], "Turning-angle absolute error (deg)", "(b)"),
        (ERROR_FIELDS[2], "Straightness absolute error", "(c)"),
    ]

    for variant, include_panel_labels in {"annotated": True, "clean": False}.items():
        figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)
        for ax, (field, ylabel, panel_label) in zip(axes, specifications):
            grouped = [
                np.asarray(
                    [float(row[field]) for row in error_rows if row["method"] == method],
                    dtype=np.float64,
                )
                for method in method_order
            ]
            boxes = ax.boxplot(
                grouped,
                positions=np.arange(1, len(method_order) + 1),
                widths=0.58,
                whis=1.5,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#b42318", "linewidth": 1.5},
                whiskerprops={"color": "#4b5563", "linewidth": 1.0},
                capprops={"color": "#4b5563", "linewidth": 1.0},
                boxprops={"edgecolor": "#374151", "linewidth": 1.0},
            )
            for patch, method in zip(boxes["boxes"], method_order):
                patch.set_facecolor(method_colors[method])
                patch.set_alpha(0.78)
            ax.set_xticks(np.arange(1, len(method_order) + 1), display_labels)
            ax.tick_params(axis="x", labelsize=7.2, pad=3)
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.55, 4.45)
            ax.set_ylim(bottom=0.0)
            ax.grid(axis="y", color="#d1d5db", linestyle=":", linewidth=0.65, alpha=0.8)
            ax.set_axisbelow(True)
            if include_panel_labels:
                ax.text(
                    0.01,
                    1.02,
                    panel_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                )
        stem = f"Figure13_table_consistent_boxplots_{variant}"
        figure.savefig(output_root / f"{stem}.png", dpi=dpi, bbox_inches="tight")
        figure.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
        plt.close(figure)


def write_paper_table(output_root: Path, summaries: list[dict[str, object]]) -> None:
    paper_rows: list[dict[str, object]] = []
    for row in summaries:
        paper_rows.append(
            {
                "Method": row["method"],
                "Path length relative error (%), median [IQR]": f"{row[ERROR_FIELDS[0] + '_median']:.2f} [{row[ERROR_FIELDS[0] + '_q1']:.2f}-{row[ERROR_FIELDS[0] + '_q3']:.2f}]",
                "Longest identity-consistent coverage (%), median [IQR]": f"{row[COVERAGE_FIELD + '_median']:.2f} [{row[COVERAGE_FIELD + '_q1']:.2f}-{row[COVERAGE_FIELD + '_q3']:.2f}]",
                "Mean turning-angle absolute error (deg), median [IQR]": f"{row[ERROR_FIELDS[1] + '_median']:.2f} [{row[ERROR_FIELDS[1] + '_q1']:.2f}-{row[ERROR_FIELDS[1] + '_q3']:.2f}]",
                "Trajectory straightness absolute error, median [IQR]": f"{row[ERROR_FIELDS[2] + '_median']:.4f} [{row[ERROR_FIELDS[2] + '_q1']:.4f}-{row[ERROR_FIELDS[2] + '_q3']:.4f}]",
            }
        )
    fields = list(paper_rows[0])
    write_csv(output_root / "Table11_movement_descriptor_accuracy.csv", paper_rows, fields)

    lines = [
        "# Table 11. Accuracy of trajectory-derived movement descriptors relative to annotation-derived pseudo-3D references.",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(fields) - 1)) + "|",
    ]
    for row in paper_rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    lines.extend(
        [
            "",
            f"Path length, turning-angle, and straightness errors are median [Q1-Q3] across {summaries[0]['n_intervals']} identical common-support intervals; lower values indicate better agreement. Longest identity-consistent coverage is median [Q1-Q3] across {summaries[0]['n_reference_trajectories']} complete reference trajectories; higher values indicate better identity continuity. Median and IQR are reported because the distributions are non-Gaussian.",
        ]
    )
    (output_root / "Table11_movement_descriptor_accuracy.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_section_text(output_root: Path, summaries: list[dict[str, object]], interval_count: int) -> None:
    lookup = {str(row["method"]): row for row in summaries}
    ours = lookup["FVSA-HRTR"]
    best_error = {
        field: min(summaries, key=lambda row: float(row[field + "_median"])) for field in ERROR_FIELDS
    }
    best_coverage = max(
        summaries, key=lambda row: float(row[COVERAGE_FIELD + "_median"])
    )
    text = f"""# 3.7. Reliability of trajectory-derived movement descriptors

The reliability of movement descriptors derived from the reconstructed trajectories was evaluated against the annotation-derived pseudo-3D references. To prevent unequal trajectory coverage from favoring a particular method, path length, mean turning angle, and trajectory straightness were calculated over the same {interval_count} temporally continuous intervals. A frame was included in these descriptor comparisons only when all four methods produced a valid 3D point matched to the same reference identity under the fixed 25 mm point-distance criterion. Intervals were split at missing frames and whenever the reconstructed identity of any method changed. Thus, each method was evaluated using identical identities, frame indices, durations, and reference coordinates.

For an interval containing 3D coordinates ${{\\mathbf{{X}}_k}}$, path length was calculated as the sum of Euclidean distances between consecutive points. Mean turning angle was calculated from the angles between successive non-zero displacement vectors, and trajectory straightness was defined as the net displacement divided by the cumulative path length. Relative absolute error was used for path length, whereas absolute errors were used for mean turning angle and trajectory straightness.

Identity continuity was assessed separately on each complete reference trajectory rather than on the common-support intervals. For a reference trajectory containing $N_r$ valid frames, the longest identity-consistent coverage was defined as $C_{{LIC}}=100N_{{max}}/N_r$, where $N_{{max}}$ is the longest sequence of consecutive frames in which the same predicted identity remained validly associated with that reference under the 25 mm criterion. Missing matches, non-consecutive frames, and predicted-ID changes terminated a run. This evaluation used all {ours['n_reference_trajectories']} reference trajectories, including trajectories with no valid predicted match, for which coverage was zero.

Table 11 and Fig. 13 summarize the results. Because the distributions were non-Gaussian and path-length relative errors were strongly right-skewed by near-static reference intervals, the median and interquartile range are reported. FVSA-HRTR yielded a median path-length relative error of {float(ours[ERROR_FIELDS[0] + '_median']):.2f}%, a longest identity-consistent coverage of {float(ours[COVERAGE_FIELD + '_median']):.2f}%, a mean turning-angle absolute error of {float(ours[ERROR_FIELDS[1] + '_median']):.2f} degrees, and a trajectory-straightness absolute error of {float(ours[ERROR_FIELDS[2] + '_median']):.4f}. The lowest median path-length error was obtained by {best_error[ERROR_FIELDS[0]]['method']}, the highest median identity-consistent coverage by {best_coverage['method']}, the lowest median turning-angle error by {best_error[ERROR_FIELDS[1]]['method']}, and the lowest median straightness error by {best_error[ERROR_FIELDS[2]]['method']}. These results quantify descriptor fidelity and identity continuity under the annotation-derived pseudo-3D reference protocol.

**Suggested caption.** Fig. 13. Reliability of trajectory-derived movement descriptors obtained using different 3D reconstruction methods. (a) Path-length relative error; (b) longest identity-consistent continuous coverage; (c) mean turning-angle absolute error; and (d) trajectory-straightness absolute error. Descriptor errors were calculated relative to annotation-derived pseudo-3D references over identical temporally continuous intervals, whereas identity-consistent coverage was calculated over each complete reference trajectory.
"""
    (output_root / "Section_3_7_Reliability_of_trajectory_derived_movement_descriptors.md").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trajectory-derived descriptor reliability.")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--distance-mm", type=float, default=DEFAULT_DISTANCE_MM)
    parser.add_argument("--min-interval-frames", type=int, default=DEFAULT_MIN_INTERVAL_FRAMES)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    pipeline_root = Path(args.pipeline_root).resolve()
    data_root = Path(args.data_root).resolve()
    comparison_root = Path(args.comparison_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    if args.distance_mm <= 0:
        raise ValueError("distance-mm must be positive")
    if args.min_interval_frames < 3:
        raise ValueError("At least three frames are required to calculate turning angles.")

    predictions_path = comparison_root / "reports" / "point_3D_MOT_predictions.csv"
    metadata_path = pipeline_root / "step4_association" / "run_metadata.json"
    for path in (predictions_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test_sequences = list(metadata["test_sequences"])
    s4.configure_input_roots(
        Path(metadata["step3_train_root"]),
        Path(metadata["step3_test_root"]),
        data_root,
        pipeline_root / "step1_rectification",
    )
    s5.configure_step1_root(pipeline_root / "step1_rectification")

    print("[1/5] Loading frozen 3D predictions")
    predictions = load_predictions(predictions_path)
    missing_methods = [method for method in METHODS if method not in predictions]
    if missing_methods:
        raise AssertionError(f"Missing frozen methods: {missing_methods}")

    print("[2/5] Rebuilding unchanged pseudo-3D references")
    references = {sequence: s5.build_gt_reference(sequence)[0] for sequence in test_sequences}
    frame_rates = {
        sequence: float(s4.base.read_seqinfo(sequence)["frameRate"]) for sequence in test_sequences
    }

    print("[3/5] Constructing common identity-consistent temporal intervals")
    error_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    sequence_audit: list[dict[str, object]] = []
    interval_id = 0
    for sequence in test_sequences:
        gt_frames = p3d.gt_by_frame(references[sequence])
        matches = {
            method: framewise_matches(gt_frames, predictions[method].get(sequence, []), args.distance_mm)
            for method in METHODS
        }
        coverage_rows.extend(longest_identity_coverage_rows(sequence, references[sequence], matches))
        intervals = common_intervals(references[sequence], matches, args.min_interval_frames)
        accepted = 0
        for gt_id, frames in intervals:
            frame_array = np.asarray(frames, dtype=np.int64)
            reference_xyz = np.asarray([references[sequence][gt_id][frame] for frame in frames])
            reference_values = movement_descriptors(reference_xyz, frame_array, frame_rates[sequence])
            if reference_values is None:
                continue
            predicted_values: dict[str, DescriptorSet] = {}
            for method in METHODS:
                xyz = np.asarray([matches[method][gt_id][frame].xyz for frame in frames])
                values = movement_descriptors(xyz, frame_array, frame_rates[sequence])
                if values is None:
                    break
                predicted_values[method] = values
            if len(predicted_values) != len(METHODS):
                continue

            interval_id += 1
            accepted += 1
            interval_key = f"{sequence}_GT{gt_id}_{frames[0]:04d}-{frames[-1]:04d}"
            interval_rows.append(
                {
                    "interval_id": interval_id,
                    "interval_key": interval_key,
                    "sequence": sequence,
                    "gt_id": gt_id,
                    "start_frame": frames[0],
                    "end_frame": frames[-1],
                    "n_frames": len(frames),
                    "duration_s": reference_values.duration_s,
                    "reference_path_length_mm": reference_values.path_length_mm,
                    "reference_mean_turning_angle_deg": reference_values.mean_turning_angle_deg,
                    "reference_trajectory_straightness": reference_values.trajectory_straightness,
                }
            )
            for method in METHODS:
                values = predicted_values[method]
                errors = descriptor_errors(values, reference_values)
                common = {
                    "interval_id": interval_id,
                    "interval_key": interval_key,
                    "sequence": sequence,
                    "gt_id": gt_id,
                    "start_frame": frames[0],
                    "end_frame": frames[-1],
                    "n_frames": len(frames),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                }
                error_rows.append({**common, **errors})
                value_rows.append(
                    {
                        **common,
                        "path_length_mm": values.path_length_mm,
                        "duration_s": values.duration_s,
                        "mean_turning_angle_deg": values.mean_turning_angle_deg,
                        "trajectory_straightness": values.trajectory_straightness,
                    }
                )
        sequence_audit.append(
            {
                "sequence": sequence,
                "frame_rate_fps": frame_rates[sequence],
                "candidate_common_runs": len(intervals),
                "evaluated_intervals": accepted,
                "evaluated_frames": sum(len(frames) for _gt_id, frames in intervals),
            }
        )
        print(f"  {sequence}: {accepted} evaluated intervals")

    if not interval_rows:
        raise RuntimeError("No valid common-support intervals were found.")

    print("[4/5] Writing traceable tables")
    error_fields = list(error_rows[0])
    value_fields = list(value_rows[0])
    interval_fields = list(interval_rows[0])
    write_csv(output_root / "movement_descriptor_errors_per_interval.csv", error_rows, error_fields)
    write_csv(
        output_root / "longest_identity_consistent_coverage_per_reference.csv",
        coverage_rows,
        list(coverage_rows[0]),
    )
    write_csv(output_root / "movement_descriptor_values_per_interval.csv", value_rows, value_fields)
    write_csv(output_root / "common_interval_manifest.csv", interval_rows, interval_fields)
    write_csv(output_root / "common_interval_audit_per_sequence.csv", sequence_audit, list(sequence_audit[0]))

    summaries = summarize_errors(error_rows, coverage_rows)
    summary_fields = list(summaries[0])
    write_csv(output_root / "movement_descriptor_error_summary_full.csv", summaries, summary_fields)
    write_paper_table(output_root, summaries)

    config = {
        "experiment": "trajectory-derived movement descriptor reliability",
        "evaluation_only": True,
        "predictions_file": str(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
        "pseudo_3d_reference": "unchanged s5.build_gt_reference from synchronized rectified GT box centers",
        "test_sequences": test_sequences,
        "frame_rates_fps": frame_rates,
        "methods": METHOD_LABELS,
        "point_matching": "framewise full Euclidean Hungarian assignment followed by distance rejection",
        "distance_gate_mm": float(args.distance_mm),
        "common_support": "same GT identity and frame matched by all four methods; split on missing frame or any method track-ID change",
        "minimum_interval_frames": int(args.min_interval_frames),
        "evaluated_interval_count": len(interval_rows),
        "evaluated_interval_frames_total": int(sum(int(row["n_frames"]) for row in interval_rows)),
        "evaluated_reference_trajectory_count_per_method": int(
            len(coverage_rows) / len(METHODS)
        ),
        "descriptor_definitions": {
            "path_length": "sum_k ||X_(k+1)-X_k||_2",
            "duration": "(last_frame-first_frame)/fps",
            "mean_turning_angle": "mean arccos of consecutive non-zero displacement-vector cosine similarities, degrees",
            "trajectory_straightness": "||X_last-X_first||_2/path_length",
            "longest_identity_consistent_coverage": "100 * longest consecutive run matched to one predicted identity / total valid frames in the complete reference trajectory",
        },
        "error_definitions": {
            "path_length": "100*abs(pred-ref)/ref",
            "mean_turning_angle": "abs(pred-ref), degrees",
            "trajectory_straightness": "abs(pred-ref)",
        },
        "random_seed": 20260823,
        "parameter_selection_on_test_errors": False,
    }
    (output_root / "movement_descriptor_experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("[5/5] Rendering publication figures and manuscript assets")
    save_figures(output_root, error_rows, coverage_rows, args.dpi)
    save_three_descriptor_figures(output_root, error_rows, args.dpi)
    save_table_summary_three_panel_figures(output_root, summaries, args.dpi)
    save_table_consistent_boxplots(output_root, error_rows, args.dpi)
    write_section_text(output_root, summaries, len(interval_rows))

    readme = f"""# Trajectory-derived movement descriptor reliability

This folder was generated from frozen 3D predictions and unchanged pseudo-3D references.

- Test sequences: {', '.join(test_sequences)}
- Point-distance gate: {args.distance_mm:g} mm
- Minimum continuous interval: {args.min_interval_frames} frames
- Evaluated common-support intervals: {len(interval_rows)}
- Total interval frames (intervals may belong to different identities): {sum(int(row['n_frames']) for row in interval_rows)}

`Table11_movement_descriptor_accuracy.csv` and `Figure13_movement_descriptor_errors.png` are the paper-facing outputs. Table 11 reports median [Q1-Q3]; the full summary CSV additionally preserves means and standard deviations. The per-interval CSV files and JSON configuration provide full traceability.

Path length, turning angle, and straightness use identical common-support intervals. Longest identity-consistent coverage is evaluated separately over complete reference trajectories so that identity breaks are not removed by common-support filtering.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    print(f"Completed: {len(interval_rows)} common-support intervals")
    for row in summaries:
        print(
            f"  {row['method']}: path={row[ERROR_FIELDS[0] + '_mean']:.3f}%, "
            f"coverage={row[COVERAGE_FIELD + '_mean']:.3f}%, "
            f"turn={row[ERROR_FIELDS[1] + '_mean']:.3f} deg, "
            f"straightness={row[ERROR_FIELDS[2] + '_mean']:.5f}"
        )


if __name__ == "__main__":
    main()
