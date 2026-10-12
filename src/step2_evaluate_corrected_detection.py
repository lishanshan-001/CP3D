"""Evaluate corrected-coordinate Step 2 detections and prepare paper assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = ROOT / ".runtime"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(RUNTIME_ROOT / "matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

from step1_stereo_rectification import load_calibration


DATA_ROOT = ROOT / "CP3D_dataset"


def list_images(path: Path) -> list[Path]:
    images = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        images.extend(path.glob(pattern))
    return sorted(images)


def locate_sequence(seq_name: str) -> tuple[str, Path]:
    for split in ("train", "test"):
        path = DATA_ROOT / split / seq_name
        if path.is_dir():
            return split, path
    raise FileNotFoundError(seq_name)


def read_gt(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "frame": int(parts[0]),
                    "object_id": int(parts[1]),
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "w": float(parts[4]),
                    "h": float(parts[5]),
                }
            )
    return rows


def rectify_gt_rows(
    seq_name: str, view: str, step1_root: Path, calib_file: Path
) -> list[dict[str, float]]:
    _, seq_root = locate_sequence(seq_name)
    k1, d1, k2, d2, _, _ = load_calibration(calib_file)
    rect_root = step1_root / seq_name
    if view == "left":
        k, d = k1, d1
        rotation = np.load(rect_root / "R1.npy")
        projection = np.load(rect_root / "P1.npy")
        gt_path = seq_root / "gt" / "gt_left.txt"
    else:
        k, d = k2, d2
        rotation = np.load(rect_root / "R2.npy")
        projection = np.load(rect_root / "P2.npy")
        gt_path = seq_root / "gt" / "gt_right.txt"

    image = cv2.imread(str(list_images(rect_root / ("img1" if view == "left" else "img2"))[0]))
    height, width = image.shape[:2]
    output = []
    for row in read_gt(gt_path):
        xs = np.linspace(row["x"], row["x"] + row["w"], 3, dtype=np.float32)
        ys = np.linspace(row["y"], row["y"] + row["h"], 3, dtype=np.float32)
        points = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)
        rectified = cv2.undistortPoints(points, k, d, R=rotation, P=projection).reshape(-1, 2)
        x1 = float(np.clip(np.min(rectified[:, 0]), 0, width - 1))
        y1 = float(np.clip(np.min(rectified[:, 1]), 0, height - 1))
        x2 = float(np.clip(np.max(rectified[:, 0]), 0, width - 1))
        y2 = float(np.clip(np.max(rectified[:, 1]), 0, height - 1))
        output.append(
            {
                **row,
                "x": x1,
                "y": y1,
                "w": max(0.0, x2 - x1),
                "h": max(0.0, y2 - y1),
            }
        )
    return output


def read_detections(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "w": float(row["w"]),
                    "h": float(row["h"]),
                    "conf": float(row["conf"]),
                }
            )
    return rows


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - intersection
    return intersection / union if union > 0 else 0.0


def frame_match(gt_rows, det_rows, iou_threshold: float) -> tuple[int, int, int]:
    if not gt_rows:
        return 0, len(det_rows), 0
    if not det_rows:
        return 0, 0, len(gt_rows)
    ious = np.asarray([[box_iou(gt, det) for det in det_rows] for gt in gt_rows], dtype=np.float64)
    cost = 1.0 - ious
    cost[ious < iou_threshold] = 1e6
    gt_idx, det_idx = linear_sum_assignment(cost)
    tp = sum(ious[g, d] >= iou_threshold for g, d in zip(gt_idx, det_idx))
    return int(tp), int(len(det_rows) - tp), int(len(gt_rows) - tp)


def group_frames(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["frame"])].append(row)
    return grouped


def evaluate_view(gt_rows, det_rows, iou_threshold: float) -> dict[str, float]:
    gt_by_frame = group_frames(gt_rows)
    det_by_frame = group_frames(det_rows)
    tp = fp = fn = 0
    for frame in sorted(set(gt_by_frame) | set(det_by_frame)):
        frame_tp, frame_fp, frame_fn = frame_match(
            gt_by_frame.get(frame, []), det_by_frame.get(frame, []), iou_threshold
        )
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pooled_metrics(rows: list[dict]) -> dict[str, float]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def make_detection_figure(sequences, det_root: Path, step1_root: Path, figure_root: Path) -> None:
    selected = [name for name in sequences if name.endswith("-03")]
    if len(selected) != 4:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), constrained_layout=True)
    for ax, seq_name in zip(axes.flat, sorted(selected)):
        frame = 300
        image_path = next(
            path for path in list_images(step1_root / seq_name / "img1")
            if int("".join(filter(str.isdigit, path.stem))) == frame
        )
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        detections = [
            row for row in read_detections(det_root / seq_name / "det_left.csv")
            if int(row["frame"]) == frame
        ]
        ax.imshow(image)
        for row in detections:
            ax.add_patch(
                Rectangle(
                    (row["x"], row["y"]),
                    row["w"],
                    row["h"],
                    fill=False,
                    edgecolor="#39ff6a",
                    linewidth=0.8,
                )
            )
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis("off")
    png_path = figure_root / "Figure_step2_detection_examples_clean.png"
    pdf_path = figure_root / "Figure_step2_detection_examples_clean.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    global DATA_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--det-root", required=True)
    parser.add_argument("--step1-root", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--calib-file", default=str(ROOT / "calibration" / "stereo_params.txt"))
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.5, 0.1])
    parser.add_argument("--primary-iou", type=float, default=0.5)
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_root).resolve()
    calib_file = Path(args.calib_file).resolve()
    det_root = Path(args.det_root).resolve()
    step1_root = Path(args.step1_root).resolve()
    save_root = Path(args.save_root).resolve()
    table_root = save_root / "tables"
    figure_root = save_root / "figures"
    report_root = save_root / "reports"
    for path in (table_root, figure_root, report_root):
        path.mkdir(parents=True, exist_ok=True)

    if args.sequences:
        sequences = args.sequences
    else:
        sequences = []
        for split in args.splits:
            sequences.extend(sorted(path.name for path in (DATA_ROOT / split).glob("CP-*")))
    sequences = sorted(set(sequences))

    rectified_gt = {}
    detections = {}
    for seq_name in sequences:
        for view in ("left", "right"):
            rectified_gt[(seq_name, view)] = rectify_gt_rows(
                seq_name, view, step1_root, calib_file
            )
            detections[(seq_name, view)] = read_detections(det_root / seq_name / f"det_{view}.csv")

    all_threshold_rows = []
    primary_rows = []
    for threshold in args.iou_thresholds:
        for seq_name in sequences:
            for view in ("left", "right"):
                metrics = evaluate_view(
                    rectified_gt[(seq_name, view)], detections[(seq_name, view)], threshold
                )
                row = {
                    "iou_threshold": threshold,
                    "sequence": seq_name,
                    "illumination": seq_name.split("-")[1],
                    "view": view,
                    **metrics,
                }
                all_threshold_rows.append(row)
                if math.isclose(threshold, args.primary_iou):
                    primary_rows.append(row)

    illumination_rows = []
    for illumination in ("I0", "I1", "I2", "I3"):
        selected = [row for row in primary_rows if row["illumination"] == illumination]
        if selected:
            illumination_rows.append({"Illumination": illumination, **pooled_metrics(selected)})

    write_csv(
        report_root / "threshold_sensitivity.csv",
        all_threshold_rows,
        ["iou_threshold", "sequence", "illumination", "view", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    write_csv(
        report_root / "sequence_view_metrics.csv",
        primary_rows,
        ["iou_threshold", "sequence", "illumination", "view", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    write_csv(
        table_root / "Table3_detection_stability.csv",
        illumination_rows,
        ["Illumination", "tp", "fp", "fn", "precision", "recall", "f1"],
    )

    md_lines = [
        "Table 3. Detection stability of YOLOv12-L under different illumination conditions.",
        "",
        "| Illumination | Precision (%) | Recall (%) | F1 (%) |",
        "|---|---:|---:|---:|",
    ]
    tex_lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Illumination & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\",
        "\\midrule",
    ]
    for row in illumination_rows:
        md_lines.append(
            f"| {row['Illumination']} | {100 * row['precision']:.2f} | "
            f"{100 * row['recall']:.2f} | {100 * row['f1']:.2f} |"
        )
        tex_lines.append(
            f"{row['Illumination']} & {100 * row['precision']:.2f} & "
            f"{100 * row['recall']:.2f} & {100 * row['f1']:.2f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (table_root / "Table3_detection_stability.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    (table_root / "Table3_detection_stability.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )

    make_detection_figure(sequences, det_root, step1_root, figure_root)
    summary = {
        "det_root": str(det_root),
        "step1_root": str(step1_root),
        "splits": args.splits,
        "sequences": sequences,
        "primary_iou": args.primary_iou,
        "iou_thresholds": args.iou_thresholds,
        "overall_primary": pooled_metrics(primary_rows),
        "illumination_metrics": illumination_rows,
        "metric_definition": "One-to-one Hungarian matching at fixed IoU; pooled over both camera views.",
    }
    (report_root / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
