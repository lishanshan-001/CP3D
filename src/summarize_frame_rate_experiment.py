"""Consolidate per-FPS metrics from the frozen CP3D pipeline run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent.parent / ".runtime" / "matplotlib"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


FPS_VALUES = (20, 60, 120, 240, 480)


def sequence_name(fps: int) -> str:
    return f"CP-I3-FPS{fps:03d}-01"


def save_figure(fig: plt.Figure, root: Path, name: str) -> None:
    fig.savefig(root / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def detection_table(experiment: Path) -> pd.DataFrame:
    source = pd.read_csv(
        experiment / "step2_detection" / "evaluation" / "reports" / "sequence_view_metrics.csv"
    )
    source = source[np.isclose(source["iou_threshold"], 0.5)]
    rows = []
    for fps in FPS_VALUES:
        group = source[source["sequence"].eq(sequence_name(fps))]
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "FPS": fps,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision_percent": 100 * precision,
                "Recall_percent": 100 * recall,
                "F1_percent": 100 * f1,
            }
        )
    return pd.DataFrame(rows)


def tracking_table(experiment: Path) -> pd.DataFrame:
    rows = []
    for fps in FPS_VALUES:
        metadata = json.loads(
            (
                experiment
                / "step3_tracking_by_fps"
                / f"{fps:03d}"
                / "run_metadata.json"
            ).read_text(encoding="utf-8")
        )
        metrics = metadata["metrics"][0]
        rows.append(
            {
                "FPS": fps,
                "HOTA_percent": metrics["HOTA"],
                "DetA_percent": metrics["DetA"],
                "AssA_percent": metrics["AssA"],
                "IDF1_percent": metrics["IDF1"],
                "MOTA_percent": metrics["MOTA"],
                "IDSW": int(metrics["IDSW"]),
                "Frag": int(metrics["Frag"]),
            }
        )
    return pd.DataFrame(rows)


def association_table(experiment: Path) -> pd.DataFrame:
    detailed = pd.read_csv(
        experiment
        / "step4_association"
        / "eval"
        / "CP3D-stereo-test-all"
        / "Proposed"
        / "pedestrian_detailed.csv"
    ).set_index("seq")
    rows = []
    for fps in FPS_VALUES:
        row = detailed.loc[sequence_name(fps)]
        rows.append(
            {
                "FPS": fps,
                "HOTA_percent": 100 * float(row["HOTA___AUC"]),
                "AssA_percent": 100 * float(row["AssA___AUC"]),
                "IDF1_percent": 100 * float(row["IDF1"]),
                "MOTA_percent": 100 * float(row["MOTA"]),
                "IDSW": int(round(float(row["IDSW"]))),
                "Frag": int(round(float(row["Frag"]))),
            }
        )
    return pd.DataFrame(rows)


def gt_reference_count(dataset_root: Path, sequence: str) -> int:
    left = pd.read_csv(
        dataset_root / "test" / sequence / "gt" / "gt_left.txt",
        header=None,
        usecols=[0, 1],
        names=["frame", "gt_id"],
    )
    right = pd.read_csv(
        dataset_root / "test" / sequence / "gt" / "gt_right.txt",
        header=None,
        usecols=[0, 1],
        names=["frame", "gt_id"],
    )
    return len(left.drop_duplicates().merge(right.drop_duplicates(), on=["frame", "gt_id"]))


def exact_jitter(group: pd.DataFrame) -> list[float]:
    coordinates = group.sort_values("frame")[["x_mm", "y_mm", "z_mm"]].to_numpy(
        dtype=np.float64
    )
    if len(coordinates) < 3:
        return []
    return np.linalg.norm(np.diff(coordinates, n=2, axis=0), axis=1).tolist()


def time_normalized_jitter(group: pd.DataFrame, fps: int) -> list[float]:
    ordered = group.sort_values("frame")
    frames = ordered["frame"].to_numpy(dtype=np.int64)
    coordinates = ordered[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
    values: list[float] = []
    for index in range(1, len(ordered) - 1):
        if frames[index] - frames[index - 1] != 1:
            continue
        if frames[index + 1] - frames[index] != 1:
            continue
        second_difference = coordinates[index + 1] - 2 * coordinates[index] + coordinates[index - 1]
        values.append(float(np.linalg.norm(second_difference) * fps * fps))
    return values


def reconstruction_table(experiment: Path) -> pd.DataFrame:
    points = pd.read_csv(
        experiment / "step5_reconstruction" / "reports" / "reconstruction_points.csv"
    )
    points = points[points["method"].eq("ReliabilityGuided3D")].copy()
    dataset_root = experiment / "datasets" / "cp3d_fps"
    rows = []
    for fps in FPS_VALUES:
        sequence = sequence_name(fps)
        sequence_points = points[points["seq_name"].eq(sequence)].copy()
        referenced = sequence_points.dropna(subset=["gt_x_mm", "gt_y_mm", "gt_z_mm"])
        referenced = referenced.sort_values(
            ["gt_id", "frame", "reliability"],
            ascending=[True, True, False],
        ).drop_duplicates(["gt_id", "frame"], keep="first")
        errors = np.linalg.norm(
            referenced[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
            - referenced[["gt_x_mm", "gt_y_mm", "gt_z_mm"]].to_numpy(dtype=np.float64),
            axis=1,
        )

        jitter_terms: list[float] = []
        physical_terms: list[float] = []
        for _, group in sequence_points.groupby(["component_id", "segment_id"], sort=False):
            jitter_terms.extend(exact_jitter(group))
            physical_terms.extend(time_normalized_jitter(group, fps))

        denominator = gt_reference_count(dataset_root, sequence)
        rows.append(
            {
                "FPS": fps,
                "RMSE_mm": float(np.sqrt(np.mean(np.square(errors)))),
                "MAE_mm": float(np.mean(errors)),
                "MedAE_mm": float(np.median(errors)),
                "Jitter_mm": float(np.mean(jitter_terms)),
                "Time_normalized_jitter_mm_per_s2": float(np.mean(physical_terms)),
                "Coverage_percent": 100 * len(referenced) / denominator,
                "Evaluated_points": len(referenced),
                "Alignable_reference_points": denominator,
                "Jitter_terms": len(jitter_terms),
                "Consecutive_time_normalized_jitter_terms": len(physical_terms),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    detection: pd.DataFrame,
    tracking: pd.DataFrame,
    association: pd.DataFrame,
    reconstruction: pd.DataFrame,
    figure_root: Path,
) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    fps = np.asarray(FPS_VALUES)

    def annotate_series(
        axis: plt.Axes,
        x_values: np.ndarray,
        y_values: pd.Series,
        color: str,
        offsets: list[tuple[float, float]],
    ) -> None:
        for x_value, y_value, (x_offset, y_offset) in zip(
            x_values,
            y_values.to_numpy(dtype=np.float64),
            offsets,
        ):
            axis.annotate(
                f"{y_value:.2f}",
                xy=(x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if y_offset >= 0 else "top",
                color=color,
                fontsize=6.5,
                fontfamily="Times New Roman",
                clip_on=False,
            )

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.55), constrained_layout=True)
    detection_color = "#277DA1"
    axes[0].plot(
        fps,
        detection["F1_percent"],
        "-o",
        color=detection_color,
        lw=1.5,
        ms=4,
    )
    axes[0].set_ylabel("Detection F1 (%)")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].margins(x=0.10, y=0.18)
    annotate_series(
        axes[0],
        fps,
        detection["F1_percent"],
        detection_color,
        [(6, -9), (0, 6), (0, -9), (0, 6), (-6, -9)],
    )

    tracking_hota_color = "#4D908E"
    tracking_idf1_color = "#F8961E"
    axes[1].plot(
        fps,
        tracking["HOTA_percent"],
        "-o",
        color=tracking_hota_color,
        lw=1.5,
        ms=4,
        label="HOTA",
    )
    axes[1].plot(
        fps,
        tracking["IDF1_percent"],
        "-s",
        color=tracking_idf1_color,
        lw=1.5,
        ms=4,
        label="IDF1",
    )
    axes[1].set_ylabel("2D tracking score (%)")
    axes[1].margins(x=0.10, y=0.12)
    annotate_series(
        axes[1],
        fps,
        tracking["HOTA_percent"],
        tracking_hota_color,
        [(6, -9), (0, -9), (0, -9), (0, -9), (-6, -9)],
    )
    annotate_series(
        axes[1],
        fps,
        tracking["IDF1_percent"],
        tracking_idf1_color,
        [(6, 6), (0, 6), (0, 6), (0, 6), (-6, 6)],
    )
    axes[1].legend(frameon=False, loc="center right")

    association_hota_color = "#577590"
    association_assa_color = "#F94144"
    axes[2].plot(
        fps,
        association["HOTA_percent"],
        "-o",
        color=association_hota_color,
        lw=1.5,
        ms=4,
        label="HOTA",
    )
    axes[2].plot(
        fps,
        association["AssA_percent"],
        "-s",
        color=association_assa_color,
        lw=1.5,
        ms=4,
        label="AssA",
    )
    axes[2].set_ylabel("Stereo association score (%)")
    axes[2].margins(x=0.10, y=0.14)
    annotate_series(
        axes[2],
        fps,
        association["HOTA_percent"],
        association_hota_color,
        [(6, -9), (0, -9), (0, -9), (0, -9), (-6, -9)],
    )
    annotate_series(
        axes[2],
        fps,
        association["AssA_percent"],
        association_assa_color,
        [(6, 6), (0, 6), (0, 6), (0, 6), (-6, 6)],
    )
    axes[2].legend(frameon=False, loc="lower right")
    for axis in axes:
        axis.set_xlabel(r"Sampling rate (frames s$^{-1}$)")
        axis.set_xticks(fps)
        axis.grid(True, ls=":", lw=0.5, alpha=0.5)
    save_figure(fig, figure_root, "Figure_frame_rate_detection_tracking_association")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    axes[0].plot(fps, reconstruction["RMSE_mm"], "-o", color="#D62828", lw=1.6, ms=4)
    axes[0].set_ylabel("RMSE (mm)")
    axes[1].plot(fps, reconstruction["Jitter_mm"], "-o", color="#6A4C93", lw=1.6, ms=4)
    axes[1].set_ylabel("Jitter (mm)")
    axes[2].plot(fps, reconstruction["Coverage_percent"], "-o", color="#2A9D8F", lw=1.6, ms=4)
    axes[2].set_ylabel("Coverage (%)")
    for axis in axes:
        axis.set_xlabel(r"Sampling rate (frames s$^{-1}$)")
        axis.set_xticks(fps)
        axis.grid(True, ls=":", lw=0.5, alpha=0.5)
    save_figure(fig, figure_root, "Figure_frame_rate_3D_reconstruction")


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(frame.to_markdown(index=False, floatfmt=".3f") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures without rewriting tables that may be open elsewhere.",
    )
    args = parser.parse_args()
    experiment = Path(args.experiment_root).resolve()
    table_root = experiment / "reports" / "tables"
    figure_root = experiment / "reports" / "figures"
    table_root.mkdir(parents=True, exist_ok=True)

    tables = {
        "Table_frame_rate_detection": detection_table(experiment),
        "Table_frame_rate_2D_tracking": tracking_table(experiment),
        "Table_frame_rate_stereo_association": association_table(experiment),
        "Table_frame_rate_3D_reconstruction": reconstruction_table(experiment),
    }
    if not args.figures_only:
        for name, frame in tables.items():
            frame.to_csv(table_root / f"{name}.csv", index=False)
            write_markdown(frame, table_root / f"{name}.md")

    make_figures(
        tables["Table_frame_rate_detection"],
        tables["Table_frame_rate_2D_tracking"],
        tables["Table_frame_rate_stereo_association"],
        tables["Table_frame_rate_3D_reconstruction"],
        figure_root,
    )

    manifest = {
        "fps_values": list(FPS_VALUES),
        "tables": {name: str(table_root / f"{name}.csv") for name in tables},
        "figures": [
            str(figure_root / "Figure_frame_rate_detection_tracking_association.png"),
            str(figure_root / "Figure_frame_rate_3D_reconstruction.png"),
        ],
        "jitter_definition": (
            "Mean Euclidean norm of the second XYZ difference within reconstructed "
            "component/segment trajectories, exactly matching the finalized Step5 implementation."
        ),
        "time_normalized_jitter_definition": (
            "For consecutive-frame triples only, the same second difference is divided "
            "by dt^2 (equivalently multiplied by FPS^2), reported in mm/s^2 as an audit."
        ),
        "test_error_parameter_selection": False,
    }
    if not args.figures_only:
        (experiment / "reports" / "summary_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    for name, frame in tables.items():
        print(f"\n{name}\n{frame.to_string(index=False)}")


if __name__ == "__main__":
    main()
