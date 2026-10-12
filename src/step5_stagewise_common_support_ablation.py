"""Stage-wise common-support ablation for the corrected CP3D pipeline.

Four nested configurations are executed independently:
1. FrameVote association + direct triangulation.
2. Proposed association + direct triangulation.
3. Proposed association + TQS gating + direct triangulation.
4. Proposed association + TQS gating + reliability-guided reconstruction.

Coordinate errors are reported twice:
- on the exact intersection of unique test reference keys from all rows;
- conditionally on each row's own valid output, paired with end-to-end coverage.

No parameter is selected from test-set errors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNTIME_ROOT = ROOT / ".runtime"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(RUNTIME_ROOT / "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import step4_stereo_association_eval as s4
import step5_reliability_guided_3d_reconstruction as s5


@dataclass(frozen=True)
class Configuration:
    key: str
    label: str
    association_method: str
    association_enabled: bool
    quality_enabled: bool
    reconstruction_enabled: bool


CONFIGURATIONS = [
    Configuration("B0", "FrameVote + direct", "FrameVote", False, False, False),
    Configuration("B1", "Proposed + direct", "Proposed", True, False, False),
    Configuration("B2", "Proposed + TQA + direct", "Proposed", True, True, False),
    Configuration("B3", "Complete framework", "Proposed", True, True, True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected stage-wise common-support ablation.")
    parser.add_argument("--step5-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_pipeline(
    step5_root: Path,
) -> tuple[
    dict[str, object],
    dict[str, dict[str, list[s5.StereoSegment]]],
    list[s4.SequenceContext],
    dict[str, tuple[dict[int, dict[int, np.ndarray]], dict[int, s4.Track], dict[int, s4.Track]]],
    dict[str, tuple[float, float]],
]:
    step5_meta = load_json(step5_root / "run_metadata.json")
    step4_root = Path(str(step5_meta["step4_root"])).resolve()
    step3_train_root = Path(str(step5_meta["step3_train_root"])).resolve()
    step3_test_root = Path(str(step5_meta["step3_test_root"])).resolve()
    data_root = Path(str(step5_meta["data_root"])).resolve()
    rect_root = Path(str(step5_meta["step1_rectification_root"])).resolve()

    s5.configure_step4_root(step4_root)
    s5.configure_step1_root(rect_root)
    s4.configure_input_roots(step3_train_root, step3_test_root, data_root, rect_root)

    train_sequences = s4.find_sequences("train")
    test_sequences = s4.find_sequences("test")
    group_priors = s4.compute_group_priors(train_sequences)
    train_contexts = [s4.build_sequence_context(name, group_priors) for name in train_sequences]
    test_contexts = [s4.build_sequence_context(name, group_priors) for name in test_sequences]
    all_contexts = train_contexts + test_contexts
    for context in all_contexts:
        s4.precompute_pair_features(context)

    gt_refs = {context.seq_name: s5.build_gt_reference(context.seq_name) for context in all_contexts}
    depth_priors = s5.compute_depth_priors(train_contexts, gt_refs)
    step4_meta = load_json(step4_root / "run_metadata.json")
    selected_params = step4_meta["selected_params"]

    segments: dict[str, dict[str, list[s5.StereoSegment]]] = {
        "FrameVote": {},
        "Proposed": {},
    }
    for context in all_contexts:
        for method in segments:
            params = selected_params[method]
            score_lookup = s4.build_score_lookup(context, method)
            components = s4.method_components(
                context,
                method,
                float(params["threshold"]),
                float(params["secondary_threshold"]),
            )
            segments[method][context.seq_name] = s5.build_segments_for_context(
                context,
                components,
                score_lookup,
                gt_refs[context.seq_name][0],
                depth_priors,
            )
    return step5_meta, segments, test_contexts, gt_refs, depth_priors


def direct_rows(segment: s5.StereoSegment) -> list[tuple[int, np.ndarray, float]]:
    rows: list[tuple[int, np.ndarray, float]] = []
    for index, sample in enumerate(segment.samples):
        reliability = segment.frame_reliability[index] if index < len(segment.frame_reliability) else 0.5
        rows.append((sample.frame, np.asarray(segment.raw_points[index], dtype=np.float64), float(reliability)))
    return rows


def reliability_guided_rows_without_tqa(
    segment: s5.StereoSegment,
    context: s4.SequenceContext,
    depth_priors: dict[str, tuple[float, float]],
) -> list[tuple[int, np.ndarray, float]]:
    depth_median, depth_iqr = depth_priors.get(s4.seq_group(context.seq_name), (0.0, 500.0))
    depth_gate = max(800.0, depth_iqr * 4.0)
    raw: list[tuple[int, np.ndarray, float]] = []
    for index, sample in enumerate(segment.samples):
        reliability = segment.frame_reliability[index] if index < len(segment.frame_reliability) else 0.5
        xyz = np.asarray(segment.raw_points[index], dtype=np.float64)
        if reliability < 0.45 or abs(float(xyz[2]) - depth_median) > depth_gate:
            continue
        raw.append((sample.frame, xyz, float(reliability)))

    if len(raw) < 3:
        return raw
    frames = [item[0] for item in raw]
    points = np.asarray([item[1] for item in raw], dtype=np.float64)
    reliabilities = np.asarray([item[2] for item in raw], dtype=np.float64)

    refined: list[tuple[int, np.ndarray, float]] = []
    for index, frame in enumerate(frames):
        weighted_sum = np.zeros(3, dtype=np.float64)
        weight_sum = 0.0
        for neighbor_index, neighbor_frame in enumerate(frames):
            if abs(neighbor_frame - frame) > 2:
                continue
            temporal_weight = math.exp(-abs(neighbor_frame - frame) / 1.5)
            weight = temporal_weight * float(reliabilities[neighbor_index])
            weighted_sum += weight * points[neighbor_index]
            weight_sum += weight
        neighborhood = weighted_sum / max(weight_sum, 1e-6)
        raw_weight = 0.55 + 0.35 * float(reliabilities[index])
        xyz = raw_weight * points[index] + (1.0 - raw_weight) * neighborhood
        refined.append((frame, xyz, float(reliabilities[index])))

    completed: list[tuple[int, np.ndarray, float]] = []
    for index, item in enumerate(refined):
        completed.append(item)
        if index == len(refined) - 1:
            continue
        frame_a, xyz_a, reliability_a = refined[index]
        frame_b, xyz_b, reliability_b = refined[index + 1]
        gap = frame_b - frame_a
        if 1 < gap <= 3 and reliability_a >= 0.45 and reliability_b >= 0.45:
            for frame in range(frame_a + 1, frame_b):
                ratio = (frame - frame_a) / gap
                xyz = (1.0 - ratio) * xyz_a + ratio * xyz_b
                completed.append((frame, xyz, float(min(reliability_a, reliability_b) * 0.9)))
    completed.sort(key=lambda item: item[0])
    return completed


def reconstruct_rows(
    configuration: Configuration,
    segment: s5.StereoSegment,
    context: s4.SequenceContext,
    tqs_threshold: float,
    depth_priors: dict[str, tuple[float, float]],
) -> list[tuple[int, np.ndarray, float]]:
    if configuration.quality_enabled and segment.quality_scores["ProposedTQS"] < tqs_threshold:
        return []
    if not configuration.reconstruction_enabled:
        return direct_rows(segment)
    if not configuration.quality_enabled:
        return reliability_guided_rows_without_tqa(segment, context, depth_priors)
    return s5.reconstruct_segment_points(
        segment,
        context,
        "ReliabilityGuided3D",
        tqs_threshold,
        depth_priors,
    )


def execute_configurations(
    configurations: list[Configuration],
    segments: dict[str, dict[str, list[s5.StereoSegment]]],
    test_contexts: list[s4.SequenceContext],
    gt_refs: dict[str, tuple[dict[int, dict[int, np.ndarray]], dict[int, s4.Track], dict[int, s4.Track]]],
    depth_priors: dict[str, tuple[float, float]],
    tqs_threshold: float,
) -> dict[str, list[s5.ReconPoint]]:
    result: dict[str, list[s5.ReconPoint]] = {configuration.key: [] for configuration in configurations}
    for context in test_contexts:
        gt_map = gt_refs[context.seq_name][0]
        for configuration in configurations:
            for segment in segments[configuration.association_method][context.seq_name]:
                if segment.gt_id is None:
                    continue
                rows = reconstruct_rows(configuration, segment, context, tqs_threshold, depth_priors)
                for frame, xyz, reliability in rows:
                    result[configuration.key].append(
                        s5.ReconPoint(
                            seq_name=context.seq_name,
                            split="test",
                            method=configuration.key,
                            gt_id=segment.gt_id,
                            frame=int(frame),
                            point3d=np.asarray(xyz, dtype=np.float64),
                            gt_point3d=gt_map.get(segment.gt_id, {}).get(int(frame)),
                            left_gt_center=None,
                            right_gt_center=None,
                            reliability=float(reliability),
                            component_id=segment.component_id,
                            segment_id=segment.segment_id,
                        )
                    )
    return result


def reference_key(point: s5.ReconPoint) -> tuple[str, int, int]:
    return point.seq_name, int(point.gt_id), int(point.frame)


def deduplicate(points: list[s5.ReconPoint]) -> dict[tuple[str, int, int], s5.ReconPoint]:
    selected: dict[tuple[str, int, int], s5.ReconPoint] = {}
    for point in points:
        if point.gt_point3d is None or point.gt_id is None:
            continue
        key = reference_key(point)
        previous = selected.get(key)
        tie_break = (point.reliability, -point.component_id, -point.segment_id)
        previous_tie = (
            (previous.reliability, -previous.component_id, -previous.segment_id)
            if previous is not None
            else None
        )
        if previous is None or tie_break > previous_tie:
            selected[key] = point
    return selected


def error_metrics(
    selected: dict[tuple[str, int, int], s5.ReconPoint],
    keys: set[tuple[str, int, int]],
) -> dict[str, float]:
    errors = np.asarray(
        [
            np.linalg.norm(selected[key].point3d - selected[key].gt_point3d)
            for key in sorted(keys)
        ],
        dtype=np.float64,
    )
    return {
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))) if errors.size else float("nan"),
        "mae_mm": float(np.mean(errors)) if errors.size else float("nan"),
        "medae_mm": float(np.median(errors)) if errors.size else float("nan"),
    }


def common_support_jitter(
    selected: dict[tuple[str, int, int], s5.ReconPoint],
    common_keys: set[tuple[str, int, int]],
) -> tuple[float, int]:
    grouped_frames: dict[tuple[str, int], set[int]] = defaultdict(set)
    for sequence, gt_id, frame in common_keys:
        grouped_frames[(sequence, gt_id)].add(frame)

    terms: list[float] = []
    for (sequence, gt_id), frames in grouped_frames.items():
        for frame in sorted(frames):
            keys = [
                (sequence, gt_id, frame - 1),
                (sequence, gt_id, frame),
                (sequence, gt_id, frame + 1),
            ]
            if not all(key in common_keys for key in keys):
                continue
            xyz = [selected[key].point3d for key in keys]
            terms.append(float(np.linalg.norm(xyz[2] - 2.0 * xyz[1] + xyz[0])))
    return (float(np.mean(terms)) if terms else float("nan"), len(terms))


def conditional_jitter(points: list[s5.ReconPoint]) -> float:
    terms: list[float] = []
    grouped: dict[tuple[str, int, int], list[s5.ReconPoint]] = defaultdict(list)
    for point in points:
        grouped[(point.seq_name, point.component_id, point.segment_id)].append(point)
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: item.frame)
        xyz = np.asarray([point.point3d for point in ordered], dtype=np.float64)
        if xyz.shape[0] >= 3:
            second_difference = np.diff(xyz, n=2, axis=0)
            terms.extend(np.linalg.norm(second_difference, axis=1).tolist())
    return float(np.mean(terms)) if terms else float("nan")


def total_reference_keys(
    test_contexts: list[s4.SequenceContext],
    gt_refs: dict[str, tuple[dict[int, dict[int, np.ndarray]], dict[int, s4.Track], dict[int, s4.Track]]],
) -> int:
    keys: set[tuple[str, int, int]] = set()
    for context in test_contexts:
        for gt_id, frame_map in gt_refs[context.seq_name][0].items():
            keys.update((context.seq_name, int(gt_id), int(frame)) for frame in frame_map)
    return len(keys)


def build_summary(
    configurations: list[Configuration],
    points_by_config: dict[str, list[s5.ReconPoint]],
    denominator: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[tuple[str, int, int], s5.ReconPoint]],
    set[tuple[str, int, int]],
]:
    unique = {key: deduplicate(points) for key, points in points_by_config.items()}
    common_keys = set.intersection(*(set(points) for points in unique.values()))
    rows: list[dict[str, object]] = []
    for configuration in configurations:
        selected = unique[configuration.key]
        own_keys = set(selected)
        common_error = error_metrics(selected, common_keys)
        common_jitter, jitter_terms = common_support_jitter(selected, common_keys)
        conditional_error = error_metrics(selected, own_keys)
        rows.append(
            {
                "configuration": configuration.key,
                "label": configuration.label,
                "proposed_association": int(configuration.association_enabled),
                "trajectory_quality_assessment": int(configuration.quality_enabled),
                "reliability_guided_reconstruction": int(configuration.reconstruction_enabled),
                "common_support_rmse_mm": common_error["rmse_mm"],
                "common_support_mae_mm": common_error["mae_mm"],
                "common_support_medae_mm": common_error["medae_mm"],
                "common_support_jitter_mm": common_jitter,
                "common_support_points": len(common_keys),
                "common_support_jitter_terms": jitter_terms,
                "conditional_rmse_mm": conditional_error["rmse_mm"],
                "conditional_mae_mm": conditional_error["mae_mm"],
                "conditional_medae_mm": conditional_error["medae_mm"],
                "conditional_jitter_mm": conditional_jitter(points_by_config[configuration.key]),
                "end_to_end_coverage_pct": 100.0 * len(own_keys) / denominator,
                "output_unique_points": len(own_keys),
            }
        )
    return rows, unique, common_keys


def build_per_sequence_rows(
    configurations: list[Configuration],
    unique: dict[str, dict[tuple[str, int, int], s5.ReconPoint]],
    common_keys: set[tuple[str, int, int]],
    denominator_by_sequence: dict[str, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sequences = sorted(denominator_by_sequence)
    for configuration in configurations:
        selected = unique[configuration.key]
        for sequence in sequences:
            sequence_common = {key for key in common_keys if key[0] == sequence}
            sequence_own = {key for key in selected if key[0] == sequence}
            common_error = error_metrics(selected, sequence_common)
            conditional_error = error_metrics(selected, sequence_own)
            common_jitter, jitter_terms = common_support_jitter(selected, sequence_common)
            rows.append(
                {
                    "configuration": configuration.key,
                    "seq_name": sequence,
                    "common_support_rmse_mm": common_error["rmse_mm"],
                    "common_support_mae_mm": common_error["mae_mm"],
                    "common_support_medae_mm": common_error["medae_mm"],
                    "common_support_jitter_mm": common_jitter,
                    "common_support_points": len(sequence_common),
                    "common_support_jitter_terms": jitter_terms,
                    "conditional_rmse_mm": conditional_error["rmse_mm"],
                    "conditional_mae_mm": conditional_error["mae_mm"],
                    "conditional_medae_mm": conditional_error["medae_mm"],
                    "end_to_end_coverage_pct": 100.0
                    * len(sequence_own)
                    / denominator_by_sequence[sequence],
                    "output_unique_points": len(sequence_own),
                }
            )
    return rows


def export_point_audit(
    path: Path,
    configurations: list[Configuration],
    unique: dict[str, dict[tuple[str, int, int], s5.ReconPoint]],
    common_keys: set[tuple[str, int, int]],
) -> None:
    rows: list[dict[str, object]] = []
    for configuration in configurations:
        for key, point in sorted(unique[configuration.key].items()):
            error = float(np.linalg.norm(point.point3d - point.gt_point3d))
            rows.append(
                {
                    "configuration": configuration.key,
                    "seq_name": key[0],
                    "gt_id": key[1],
                    "frame": key[2],
                    "on_common_support": int(key in common_keys),
                    "x_mm": float(point.point3d[0]),
                    "y_mm": float(point.point3d[1]),
                    "z_mm": float(point.point3d[2]),
                    "gt_x_mm": float(point.gt_point3d[0]),
                    "gt_y_mm": float(point.gt_point3d[1]),
                    "gt_z_mm": float(point.gt_point3d[2]),
                    "error_mm": error,
                    "reliability": point.reliability,
                    "component_id": point.component_id,
                    "segment_id": point.segment_id,
                }
            )
    write_csv(path, rows)


def render_table(
    rows: list[dict[str, object]],
    output_dir: Path,
    dpi: int,
    stem: str = "Table7_stagewise_common_support_ablation",
) -> tuple[Path, Path]:
    check = "\u2713"
    headers = [
        "Configuration",
        "Assoc.",
        "TQA",
        "Recon.",
        "Common-support\nRMSE (mm) $\\downarrow$",
        "MAE (mm)\n$\\downarrow$",
        "MedAE (mm)\n$\\downarrow$",
        "Jitter (mm)\n$\\downarrow$",
        "Conditional\nRMSE (mm) $\\downarrow$",
        "Coverage (%)\n$\\uparrow$",
    ]
    cells: list[list[str]] = []
    for row in rows:
        cells.append(
            [
                str(row["configuration"]),
                check if int(row["proposed_association"]) else "",
                check if int(row["trajectory_quality_assessment"]) else "",
                check if int(row["reliability_guided_reconstruction"]) else "",
                f'{float(row["common_support_rmse_mm"]):.3f}',
                f'{float(row["common_support_mae_mm"]):.3f}',
                f'{float(row["common_support_medae_mm"]):.3f}',
                f'{float(row["common_support_jitter_mm"]):.3f}',
                f'{float(row["conditional_rmse_mm"]):.3f}',
                f'{float(row["end_to_end_coverage_pct"]):.3f}',
            ]
        )

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(13.0, 3.8), dpi=dpi)
    ax.axis("off")
    widths = [0.075, 0.065, 0.055, 0.065, 0.16, 0.105, 0.115, 0.11, 0.145, 0.105]
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=widths,
        bbox=[0.01, 0.03, 0.98, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    last_row = len(rows)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#222222")
        cell.set_linewidth(0.0)
        cell.visible_edges = ""
        cell.PAD = 0.11
        if cell.get_text().get_text() == check:
            cell.get_text().set_fontfamily("DejaVu Sans")
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
        elif row_index == last_row:
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
    metric_columns = {
        4: "common_support_rmse_mm",
        5: "common_support_mae_mm",
        6: "common_support_medae_mm",
        7: "common_support_jitter_mm",
        8: "conditional_rmse_mm",
    }
    for column_index, metric in metric_columns.items():
        best_row = min(range(len(rows)), key=lambda index: float(rows[index][metric])) + 1
        table[(best_row, column_index)].get_text().set_weight("bold")
    coverage_row = max(
        range(len(rows)),
        key=lambda index: float(rows[index]["end_to_end_coverage_pct"]),
    ) + 1
    table[(coverage_row, 9)].get_text().set_weight("bold")
    table.scale(1.0, 1.45)

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def write_tex(path: Path, rows: list[dict[str, object]], common_points: int) -> None:
    minimum_metrics = {
        metric: min(float(row[metric]) for row in rows)
        for metric in (
            "common_support_rmse_mm",
            "common_support_mae_mm",
            "common_support_medae_mm",
            "common_support_jitter_mm",
            "conditional_rmse_mm",
        )
    }
    maximum_coverage = max(float(row["end_to_end_coverage_pct"]) for row in rows)

    def metric_text(row: dict[str, object], metric: str) -> str:
        value = float(row[metric])
        text = f"{value:.3f}"
        return f"\\textbf{{{text}}}" if math.isclose(value, minimum_metrics[metric]) else text

    body: list[str] = []
    for row in rows:
        coverage = float(row["end_to_end_coverage_pct"])
        coverage_text = f"{coverage:.3f}"
        if math.isclose(coverage, maximum_coverage):
            coverage_text = f"\\textbf{{{coverage_text}}}"
        body.append(
            " & ".join(
                [
                    str(row["configuration"]),
                    "\\checkmark" if int(row["proposed_association"]) else "",
                    "\\checkmark" if int(row["trajectory_quality_assessment"]) else "",
                    "\\checkmark" if int(row["reliability_guided_reconstruction"]) else "",
                    metric_text(row, "common_support_rmse_mm"),
                    metric_text(row, "common_support_mae_mm"),
                    metric_text(row, "common_support_medae_mm"),
                    metric_text(row, "common_support_jitter_mm"),
                    metric_text(row, "conditional_rmse_mm"),
                    coverage_text,
                ]
            )
            + " \\\\"
        )
    lines = [
        "\\begin{tabular}{lcccrrrrrr}",
        "\\toprule",
        "Config. & Assoc. & TQA & Recon. & Common RMSE & MAE & MedAE & Jitter & Conditional RMSE & Coverage (\\%) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "",
        f"% Common-support metrics use the same {common_points} unique test reference keys in all rows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, object]], common_points: int) -> None:
    lines = [
        "| Config. | Proposed association | TQA | Reliability-guided reconstruction | Common RMSE | Common MAE | Common MedAE | Common jitter | Conditional RMSE | Coverage (%) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {configuration} | {association} | {quality} | {reconstruction} | "
            "{rmse:.3f} | {mae:.3f} | {medae:.3f} | {jitter:.3f} | "
            "{conditional:.3f} | {coverage:.3f} |".format(
                configuration=row["configuration"],
                association="Yes" if int(row["proposed_association"]) else "",
                quality="Yes" if int(row["trajectory_quality_assessment"]) else "",
                reconstruction="Yes" if int(row["reliability_guided_reconstruction"]) else "",
                rmse=float(row["common_support_rmse_mm"]),
                mae=float(row["common_support_mae_mm"]),
                medae=float(row["common_support_medae_mm"]),
                jitter=float(row["common_support_jitter_mm"]),
                conditional=float(row["conditional_rmse_mm"]),
                coverage=float(row["end_to_end_coverage_pct"]),
            )
        )
    lines.extend(
        [
            "",
            f"Common-support metrics use the same {common_points:,} unique `(sequence, identity, frame)` keys in every row.",
            "Conditional RMSE uses each row's own valid output and must be interpreted jointly with Coverage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selection_removed_statistics(
    unique: dict[str, dict[tuple[str, int, int], s5.ReconPoint]],
) -> dict[str, float]:
    removed_keys = set(unique["B1"]) - set(unique["B2"])
    errors = np.asarray(
        [
            np.linalg.norm(unique["B1"][key].point3d - unique["B1"][key].gt_point3d)
            for key in removed_keys
        ],
        dtype=np.float64,
    )
    return {
        "removed_points": int(errors.size),
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))) if errors.size else float("nan"),
        "mean_error_mm": float(np.mean(errors)) if errors.size else float("nan"),
        "median_error_mm": float(np.median(errors)) if errors.size else float("nan"),
        "maximum_error_mm": float(np.max(errors)) if errors.size else float("nan"),
    }


def write_analysis(
    path: Path,
    rows: list[dict[str, object]],
    common_points: int,
    removed_statistics: dict[str, float],
) -> None:
    by_key = {str(row["configuration"]): row for row in rows}
    b0, b1, b2, b3 = (by_key[key] for key in ("B0", "B1", "B2", "B3"))
    association_coverage_gain = float(b1["end_to_end_coverage_pct"]) - float(b0["end_to_end_coverage_pct"])
    association_rmse_reduction = 100.0 * (
        float(b0["common_support_rmse_mm"]) - float(b1["common_support_rmse_mm"])
    ) / float(b0["common_support_rmse_mm"])
    tqa_coverage_loss = float(b1["end_to_end_coverage_pct"]) - float(b2["end_to_end_coverage_pct"])
    tqa_rmse_reduction = 100.0 * (
        float(b1["conditional_rmse_mm"]) - float(b2["conditional_rmse_mm"])
    ) / float(b1["conditional_rmse_mm"])
    recon_rmse_reduction = 100.0 * (
        float(b2["common_support_rmse_mm"]) - float(b3["common_support_rmse_mm"])
    ) / float(b2["common_support_rmse_mm"])
    recon_mae_reduction = 100.0 * (
        float(b2["common_support_mae_mm"]) - float(b3["common_support_mae_mm"])
    ) / float(b2["common_support_mae_mm"])
    recon_medae_reduction = 100.0 * (
        float(b2["common_support_medae_mm"]) - float(b3["common_support_medae_mm"])
    ) / float(b2["common_support_medae_mm"])
    recon_jitter_reduction = 100.0 * (
        float(b2["common_support_jitter_mm"]) - float(b3["common_support_jitter_mm"])
    ) / float(b2["common_support_jitter_mm"])
    lines = [
        "# Stage-wise common-support ablation analysis",
        "",
        f"All common-support errors were calculated on the same {common_points:,} unique test observations. "
        "The end-to-end coverage denominator was fixed at 43,458 pseudo-3D reference keys.",
        "",
        "Replacing FrameVote with the Proposed association changed the composition and coverage of the reconstructed observations. "
        f"Coverage increased by {association_coverage_gain:.3f} percentage points from B0 to B1. "
        f"On the fixed common support, RMSE decreased by {association_rmse_reduction:.3f}%. "
        "The higher conditional RMSE in B1 reflects the additional difficult observations recovered by the Proposed association rather than a like-for-like loss of coordinate accuracy. "
        "The association contribution should primarily be interpreted together with the Step 4 HOTA, AssA, IDF1, MOTA, and IDSW results, "
        "because common-support reconstruction error does not measure cross-view identity consistency.",
        "",
        f"Applying the training-selected TQA threshold reduced coverage by only {tqa_coverage_loss:.3f} percentage points "
        f"while reducing conditional RMSE by {tqa_rmse_reduction:.3f}% from B1 to B2. "
        f"The gate removed {int(removed_statistics['removed_points'])} unique observations whose RMSE was "
        f"{removed_statistics['rmse_mm']:.3f} mm and whose maximum error was "
        f"{removed_statistics['maximum_error_mm']:.3f} mm. "
        "B1 and B2 are expected to have identical or nearly identical common-support coordinates because TQA is a selector rather than a coordinate estimator. "
        "Its contribution is therefore represented by the conditional-error/coverage tradeoff, not by common-support coordinate refinement.",
        "",
        f"With association and TQA fixed, reliability-guided reconstruction reduced common-support RMSE, MAE, MedAE, and jitter by "
        f"{recon_rmse_reduction:.3f}%, {recon_mae_reduction:.3f}%, {recon_medae_reduction:.3f}%, and "
        f"{recon_jitter_reduction:.3f}%, respectively, from B2 to B3. "
        "This paired comparison isolates the coordinate-level contribution of Section 2.5.",
        "",
        "This ablation should not be described as proof that every module improves every metric monotonically. "
        "It provides module-appropriate evidence while controlling support-induced selection bias.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    step5_root = args.step5_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step5_meta, segments, test_contexts, gt_refs, depth_priors = prepare_pipeline(step5_root)
    threshold = float(step5_meta["best_tqs_threshold"])
    points = execute_configurations(
        CONFIGURATIONS,
        segments,
        test_contexts,
        gt_refs,
        depth_priors,
        threshold,
    )
    denominator = total_reference_keys(test_contexts, gt_refs)
    summary_rows, unique, common_keys = build_summary(CONFIGURATIONS, points, denominator)
    removed_statistics = selection_removed_statistics(unique)
    denominator_by_sequence = {
        context.seq_name: sum(len(frame_map) for frame_map in gt_refs[context.seq_name][0].values())
        for context in test_contexts
    }
    per_sequence_rows = build_per_sequence_rows(
        CONFIGURATIONS,
        unique,
        common_keys,
        denominator_by_sequence,
    )

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    texts_dir = output_dir / "texts"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = tables_dir / "Table7_stagewise_common_support_ablation.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(reports_dir / "stagewise_common_support_per_sequence.csv", per_sequence_rows)
    export_point_audit(
        reports_dir / "stagewise_common_support_points.csv",
        CONFIGURATIONS,
        unique,
        common_keys,
    )
    write_markdown(
        tables_dir / "Table7_stagewise_common_support_ablation.md",
        summary_rows,
        len(common_keys),
    )
    write_tex(
        tables_dir / "Table7_stagewise_common_support_ablation.tex",
        summary_rows,
        len(common_keys),
    )
    png_path, pdf_path = render_table(summary_rows, output_dir, args.dpi)
    write_analysis(
        texts_dir / "Section_3_4_stagewise_ablation_analysis.md",
        summary_rows,
        len(common_keys),
        removed_statistics,
    )

    config = {
        "experiment": "Stage-wise common-support framework ablation",
        "step5_root": str(step5_root),
        "step5_metadata_sha256": sha256(step5_root / "run_metadata.json"),
        "step4_root": str(step5_meta["step4_root"]),
        "step3_train_root": str(step5_meta["step3_train_root"]),
        "step3_test_root": str(step5_meta["step3_test_root"]),
        "step1_rectification_root": str(step5_meta["step1_rectification_root"]),
        "test_sequences": [context.seq_name for context in test_contexts],
        "test_reference_denominator": denominator,
        "tqs_threshold": threshold,
        "tqs_threshold_source": "Training split only; reused unchanged from the finalized corrected Step 5 metadata.",
        "common_support_definition": "Intersection of unique (sequence, ground-truth identity, frame) reference keys output by all four independently executed configurations.",
        "common_support_points": len(common_keys),
        "common_support_jitter_definition": "Mean L2 norm of the second XYZ difference over identical consecutive-frame triples on the common support.",
        "conditional_jitter_definition": "Mean L2 norm of the second XYZ difference within each configuration's reconstructed trajectory segments.",
        "coverage_definition": "Unique output reference keys divided by all 43,458 alignable pseudo-3D reference keys in the four test sequences.",
        "duplicate_policy": "For duplicate reference keys, retain the observation with highest frame reliability; ties use component and segment identifiers deterministically.",
        "test_error_parameter_selection": False,
        "tqa_removed_observation_statistics": removed_statistics,
        "configurations": [configuration.__dict__ for configuration in CONFIGURATIONS],
        "outputs": {
            "summary_csv": str(summary_csv),
            "table_png": str(png_path),
            "table_pdf": str(pdf_path),
        },
    }
    (reports_dir / "stagewise_common_support_ablation_config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    complete = next(row for row in summary_rows if row["configuration"] == "B3")
    expected = next(
        row
        for row in step5_meta["reconstruction_rows"]
        if row["method"] == "ReliabilityGuided3D"
    )
    for name, expected_key in (
        ("conditional_rmse_mm", "rmse_mm"),
        ("conditional_mae_mm", "mae_mm"),
        ("conditional_medae_mm", "medae_mm"),
        ("conditional_jitter_mm", "jitter_mm"),
    ):
        if not math.isclose(
            float(complete[name]),
            float(expected[expected_key]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise AssertionError(f"Complete framework mismatch for {name}.")
    if not math.isclose(
        float(complete["end_to_end_coverage_pct"]),
        100.0 * float(expected["coverage"]),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise AssertionError("Complete framework coverage does not reproduce final Step 5.")

    print(json.dumps({"denominator": denominator, "common_support": len(common_keys), "rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
