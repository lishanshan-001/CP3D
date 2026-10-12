"""Add fixed-real-time fluctuation to the frozen sampling-rate experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import summarize_frame_rate_experiment as base
import step5_point_3d_mot_comparison as p3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=0.20)
    return parser.parse_args()


def contiguous_runs(group: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = group.sort_values("frame").drop_duplicates("frame", keep="first")
    if ordered.empty:
        return []
    frame_steps = ordered["frame"].diff().fillna(1).ne(1).cumsum()
    return [run for _, run in ordered.groupby(frame_steps, sort=False)]


def fixed_window_fluctuation(
    group: pd.DataFrame,
    fps: int,
    window_seconds: float,
) -> list[float]:
    """Local linear-detrended XYZ RMS over complete fixed-duration windows."""
    half_window = 0.5 * window_seconds
    tolerance = 0.5 / fps + 1e-12
    values: list[float] = []
    for run in contiguous_runs(group):
        if len(run) < 3:
            continue
        frames = run["frame"].to_numpy(dtype=np.int64)
        times = frames.astype(np.float64) / float(fps)
        xyz = run[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
        for center_time in times:
            indices = np.flatnonzero(np.abs(times - center_time) <= half_window + 1e-12)
            if indices.size < 3:
                continue
            local_times = times[indices]
            if local_times[0] > center_time - half_window + tolerance:
                continue
            if local_times[-1] < center_time + half_window - tolerance:
                continue
            relative = local_times - center_time
            design = np.column_stack([np.ones(indices.size), relative])
            coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
                design, xyz[indices], rcond=None
            )
            residual_xyz = xyz[indices] - design @ coefficients
            values.append(
                float(np.sqrt(np.mean(np.sum(np.square(residual_xyz), axis=1))))
            )
    return values


def main() -> None:
    args = parse_args()
    experiment = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original = base.reconstruction_table(experiment)
    points = pd.read_csv(
        experiment / "step5_reconstruction" / "reports" / "reconstruction_points.csv"
    )
    points = points[points["method"].eq("ReliabilityGuided3D")].copy()

    step4_metadata = json.loads(
        (experiment / "step4_association" / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    train_sequences = list(step4_metadata["train_sequences"])
    test_sequences = list(step4_metadata["test_sequences"])
    p3d.s4.configure_input_roots(
        Path(step4_metadata["step3_train_root"]),
        Path(step4_metadata["step3_test_root"]),
        Path(step4_metadata["data_root"]),
        Path(step4_metadata["rect_root"]),
    )
    p3d.s5.configure_step1_root(Path(step4_metadata["rect_root"]))
    priors = p3d.s4.compute_group_priors(train_sequences)
    contexts = {
        sequence: p3d.s4.build_sequence_context(sequence, priors)
        for sequence in test_sequences
    }
    gt_maps = {
        sequence: p3d.s5.build_gt_reference(sequence)[0] for sequence in test_sequences
    }

    fluctuation_rows: list[dict[str, object]] = []
    for fps in base.FPS_VALUES:
        sequence = base.sequence_name(fps)
        sequence_points = points[points["seq_name"].eq(sequence)]
        values: list[float] = []
        run_count = 0
        for _, group in sequence_points.groupby(
            ["component_id", "segment_id"], sort=False
        ):
            runs = contiguous_runs(group)
            run_count += len(runs)
            values.extend(fixed_window_fluctuation(group, fps, args.window_seconds))
        if not values:
            raise RuntimeError(f"No complete fixed-time windows for {sequence}")
        fluctuation_rows.append(
            {
                "FPS": fps,
                "Fixed_window_fluctuation_mm": float(np.mean(values)),
                "Fixed_window_fluctuation_median_mm": float(np.median(values)),
                "Fixed_window_count": len(values),
                "Consecutive_run_count": run_count,
                "Window_seconds": args.window_seconds,
                "Nominal_samples_per_window": int(round(args.window_seconds * fps)) + 1,
            }
        )

    fluctuation = pd.DataFrame(fluctuation_rows)

    matched_rows: list[dict[str, object]] = []
    for fps in base.FPS_VALUES:
        sequence = base.sequence_name(fps)
        sequence_points = points[points["seq_name"].eq(sequence)].copy()
        sequence_points = sequence_points.sort_values(
            ["component_id", "frame", "reliability"],
            ascending=[True, True, False],
        ).drop_duplicates(["component_id", "frame"], keep="first")
        predictions = [
            p3d.TrackPoint3D(
                seq_name=sequence,
                frame=int(row.frame),
                track_id=int(row.component_id),
                xyz=np.asarray([row.x_mm, row.y_mm, row.z_mm], dtype=np.float64),
                score=float(row.reliability),
            )
            for row in sequence_points.itertuples(index=False)
            if np.isfinite(row.x_mm)
            and np.isfinite(row.y_mm)
            and np.isfinite(row.z_mm)
            and float(row.z_mm) > 0.0
        ]
        metric_row, _raw = p3d.evaluate_one_sequence(
            sequence,
            gt_maps[sequence],
            predictions,
            contexts[sequence].seq_length,
            25.0,
        )
        matched_rows.append(
            {
                "FPS": fps,
                "RMSE_mm": metric_row["RMSE_mm"],
                "MAE_mm": metric_row["MAE_mm"],
                "MedAE_mm": metric_row["MedAE_mm"],
                "Coverage_percent": metric_row["Coverage_pct"],
                "Evaluated_points": int(metric_row["Matched_points"]),
                "Alignable_reference_points": int(metric_row["GT_points"]),
                "Predicted_points": int(metric_row["Predicted_points"]),
                "HOTA_3D": metric_row["HOTA_3D"],
                "AssA_3D": metric_row["AssA_3D"],
                "IDF1_3D": metric_row["IDF1_3D"],
                "MOTA_3D": metric_row["MOTA_3D"],
                "IDSW": int(metric_row["IDSW"]),
            }
        )

    matched = pd.DataFrame(matched_rows)
    original_jitter = original[["FPS", "Jitter_mm", "Jitter_terms"]]
    combined = (
        matched.merge(original_jitter, on="FPS", validate="one_to_one")
        .merge(fluctuation, on="FPS", validate="one_to_one")
        .sort_values("FPS")
    )
    columns = [
        "FPS",
        "RMSE_mm",
        "MAE_mm",
        "MedAE_mm",
        "Jitter_mm",
        "Fixed_window_fluctuation_mm",
        "Coverage_percent",
        "Evaluated_points",
        "Alignable_reference_points",
        "Predicted_points",
        "Jitter_terms",
        "Fixed_window_count",
        "Consecutive_run_count",
        "Window_seconds",
        "Nominal_samples_per_window",
    ]
    combined[columns].to_csv(
        output_dir / "Table_sampling_rate_3D_with_fixed_window.csv", index=False
    )

    lines = [
        "# 3D reconstruction performance across sampling rates",
        "",
        "| Sampling rate (frames s^-1) | RMSE (mm) | MAE (mm) | MedAE (mm) | Per-frame Jitter (mm) | Fixed-window fluctuation (mm) | Coverage (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined.to_dict("records"):
        lines.append(
            f"| {int(row['FPS'])} | {row['RMSE_mm']:.2f} | {row['MAE_mm']:.2f} | "
            f"{row['MedAE_mm']:.2f} | {row['Jitter_mm']:.2f} | "
            f"{row['Fixed_window_fluctuation_mm']:.2f} | {row['Coverage_percent']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Per-frame Jitter is the original unnormalized coordinate second-difference metric. Fixed-window fluctuation is the mean local linear-detrended XYZ RMS residual over complete 0.20 s windows within consecutive HRTR trajectory runs. RMSE, MAE, and MedAE use frame-wise Hungarian matches accepted within 25 mm; Coverage is accepted 25 mm matches divided by all alignable pseudo-3D reference points.",
        ]
    )
    (output_dir / "Table_sampling_rate_3D_with_fixed_window.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    latex = [
        "\\begin{tabular}{rrrrrrr}",
        "\\toprule",
        "Sampling rate & RMSE & MAE & MedAE & Per-frame jitter & Fixed-window fluctuation & Coverage \\\\",
        "(frames s$^{-1}$) & (mm) & (mm) & (mm) & (mm) & (mm) & (\\%) \\\\",
        "\\midrule",
    ]
    for row in combined.to_dict("records"):
        latex.append(
            f"{int(row['FPS'])} & {row['RMSE_mm']:.2f} & {row['MAE_mm']:.2f} & "
            f"{row['MedAE_mm']:.2f} & {row['Jitter_mm']:.2f} & "
            f"{row['Fixed_window_fluctuation_mm']:.2f} & {row['Coverage_percent']:.2f} \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "Table_sampling_rate_3D_with_fixed_window.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )

    config = {
        "experiment_root": str(experiment),
        "sampling_rates_fps": list(base.FPS_VALUES),
        "method": "ReliabilityGuided3D (frozen frame-rate experiment output)",
        "spatial_evaluation_distance_mm": 25.0,
        "spatial_matching": "Frame-wise Hungarian assignment; accept Euclidean distance <= 25 mm",
        "coverage": "Accepted 25 mm matches / all alignable pseudo-3D reference points",
        "fixed_window_seconds": args.window_seconds,
        "fixed_window_definition": "Mean local linear-detrended XYZ RMS residual over complete fixed-duration windows within consecutive component/segment runs",
        "per_frame_jitter_definition": "Mean norm of unnormalized coordinate second differences, as in the original frozen frame-rate summary",
        "window_parameter_source": "Fixed to the 0.20 s physical span of the finalized +/-2-frame HRTR neighborhood at 20 FPS; not selected on test errors",
        "reconstruction_recomputed": False,
        "spatial_metrics_recomputed_under_25mm_protocol": True,
        "test_parameter_tuning": False,
        "all_sampling_rates_present": combined["FPS"].astype(int).tolist()
        == list(base.FPS_VALUES),
        "all_fixed_windows_nonempty": bool((combined["Fixed_window_count"] > 0).all()),
    }
    (output_dir / "sampling_rate_fixed_window_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(combined[columns].to_string(index=False))


if __name__ == "__main__":
    main()
