# -*- coding: utf-8 -*-

"""
Step 3
Build MOTChallenge-style left/right datasets, run multiple non-appearance MOT
trackers, evaluate with official TrackEval, and export paper-ready comparison tables.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DATA_ROOT = ROOT / "CP3D_dataset"
RECT_ROOT = ROOT / "results" / "step1_rectification"
CALIB_FILE = ROOT / "calibration" / "stereo_params.txt"
DEFAULT_DET_ROOT = ROOT / "results" / "step2_detection"
STEP3_ROOT = ROOT / "results" / "step3_motchallenge"

GT_MOT_ROOT = STEP3_ROOT / "gt" / "mot_challenge"
TRACKERS_MOT_ROOT = STEP3_ROOT / "trackers" / "mot_challenge"
EVAL_ROOT = STEP3_ROOT / "eval"
REPORT_ROOT = STEP3_ROOT / "reports"
RUNTIME_ROOT = ROOT / ".runtime"

RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(RUNTIME_ROOT / "matplotlib")

import trackeval

from _mot_trackers import ByteTrackTracker
from _mot_trackers import Detection
from _mot_trackers import IOUTracker
from _mot_trackers import OCSortTracker
from _mot_trackers import SortTracker
from step1_stereo_rectification import load_calibration

TRACKER_NAMES = ["SORT", "ByteTrack", "OCSort", "IOUTrack"]
VIEWS = ["left", "right"]


def natural_sort(items: list[str]) -> list[str]:
    import re

    def key(value: str):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", value)]

    return sorted(items, key=key)


def find_sequences() -> list[str]:
    seqs = []
    for split in ("train", "test"):
        split_dir = DATA_ROOT / split
        if not split_dir.is_dir():
            continue
        seqs.extend([p.name for p in split_dir.glob("CP-*") if p.is_dir()])
    return natural_sort(sorted(set(seqs)))


def locate_split(seq_name: str) -> str:
    for split in ("train", "test"):
        if (DATA_ROOT / split / seq_name).is_dir():
            return split
    raise FileNotFoundError(f"Sequence not found in train/test: {seq_name}")


def read_seqinfo(seq_name: str) -> dict[str, str]:
    split = locate_split(seq_name)
    ini_path = DATA_ROOT / split / seq_name / "gt" / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    seq_section = parser["Sequence"]
    return {
        "name": seq_name,
        "frameRate": seq_section.get("frameRate", "20"),
        "seqLength": seq_section.get("seqLength", "0"),
        "imWidth": seq_section.get("imWidth", "1920"),
        "imHeight": seq_section.get("imHeight", "1088"),
        "imExt": seq_section.get("imExt", ".jpg"),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_img_link(dst_img_dir: Path, src_img_dir: Path) -> None:
    if dst_img_dir.exists():
        return

    try:
        os.symlink(src_img_dir, dst_img_dir, target_is_directory=True)
        return
    except OSError:
        pass

    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst_img_dir), str(src_img_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return
    except Exception:
        pass

    ensure_dir(dst_img_dir)
    note_path = dst_img_dir / "_SOURCE_DIR.txt"
    note_path.write_text(str(src_img_dir), encoding="utf-8")


def write_seqinfo(dst_path: Path, seq_name: str, seqinfo: dict[str, str]) -> None:
    content = "\n".join(
        [
            "[Sequence]",
            f"name={seq_name}",
            "imDir=img1",
            f"frameRate={seqinfo['frameRate']}",
            f"seqLength={seqinfo['seqLength']}",
            f"imWidth={seqinfo['imWidth']}",
            f"imHeight={seqinfo['imHeight']}",
            f"imExt={seqinfo['imExt']}",
            "",
        ]
    )
    dst_path.write_text(content, encoding="utf-8")


def convert_det_to_mot(src_det_path: Path, dst_det_path: Path) -> None:
    ensure_dir(dst_det_path.parent)
    with open(src_det_path, "r", encoding="utf-8") as src, open(dst_det_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame, track_id, x, y, w, h, score = parts[:7]
            dst.write(f"{frame},{track_id},{x},{y},{w},{h},{score},-1,-1,-1\n")


def copy_gt(src_gt_path: Path, dst_gt_path: Path) -> None:
    ensure_dir(dst_gt_path.parent)
    dst_gt_path.write_text(src_gt_path.read_text(encoding="utf-8"), encoding="utf-8")


def load_rectify_meta(seq_name: str, view: str) -> dict[str, np.ndarray]:
    k1, d1, k2, d2, _, _ = load_calibration(CALIB_FILE)
    seq_root = RECT_ROOT / seq_name
    if view == "left":
        return {
            "K": k1,
            "D": d1,
            "R": np.load(seq_root / "R1.npy"),
            "P": np.load(seq_root / "P1.npy"),
        }
    return {
        "K": k2,
        "D": d2,
        "R": np.load(seq_root / "R2.npy"),
        "P": np.load(seq_root / "P2.npy"),
    }


def rectify_box(x: float, y: float, w: float, h: float, meta: dict[str, np.ndarray], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xs = np.linspace(x, x + w, 3, dtype=np.float32)
    ys = np.linspace(y, y + h, 3, dtype=np.float32)
    points = np.array([[px, py] for py in ys for px in xs], dtype=np.float32).reshape(-1, 1, 2)
    rect_points = cv2.undistortPoints(points, meta["K"], meta["D"], R=meta["R"], P=meta["P"]).reshape(-1, 2)

    rx1 = float(np.clip(np.min(rect_points[:, 0]), 0, img_w - 1))
    ry1 = float(np.clip(np.min(rect_points[:, 1]), 0, img_h - 1))
    rx2 = float(np.clip(np.max(rect_points[:, 0]), 0, img_w - 1))
    ry2 = float(np.clip(np.max(rect_points[:, 1]), 0, img_h - 1))
    return rx1, ry1, max(0.0, rx2 - rx1), max(0.0, ry2 - ry1)


def rectify_gt(src_gt_path: Path, dst_gt_path: Path, seq_name: str, view: str, seqinfo: dict[str, str]) -> None:
    ensure_dir(dst_gt_path.parent)
    meta = load_rectify_meta(seq_name, view)
    img_w = int(seqinfo["imWidth"])
    img_h = int(seqinfo["imHeight"])

    grouped: dict[tuple[str, str], dict[str, object]] = {}
    with open(src_gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            frame, track_id = parts[0], parts[1]
            x, y, w, h = map(float, parts[2:6])
            mark, cls, vis = parts[6:9]
            rx, ry, rw, rh = rectify_box(x, y, w, h, meta, img_w, img_h)
            key = (frame, track_id)
            x1, y1, x2, y2 = rx, ry, rx + rw, ry + rh
            if key not in grouped:
                grouped[key] = {
                    "frame": frame,
                    "track_id": track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "mark": mark,
                    "cls": cls,
                    "vis": vis,
                }
            else:
                grouped[key]["x1"] = min(grouped[key]["x1"], x1)
                grouped[key]["y1"] = min(grouped[key]["y1"], y1)
                grouped[key]["x2"] = max(grouped[key]["x2"], x2)
                grouped[key]["y2"] = max(grouped[key]["y2"], y2)

    with open(dst_gt_path, "w", encoding="utf-8", newline="") as f:
        sorted_items = sorted(grouped.values(), key=lambda row: (int(row["frame"]), int(row["track_id"])))
        for item in sorted_items:
            rw = max(0.0, item["x2"] - item["x1"])
            rh = max(0.0, item["y2"] - item["y1"])
            row = [
                item["frame"],
                item["track_id"],
                f"{item['x1']:.2f}",
                f"{item['y1']:.2f}",
                f"{rw:.2f}",
                f"{rh:.2f}",
                item["mark"],
                item["cls"],
                item["vis"],
            ]
            f.write(",".join(row) + "\n")


def build_view_dataset(view: str, sequences: list[str], det_root: Path) -> dict[str, int]:
    dataset_name = f"CP3D-{view}-all"
    dataset_root = GT_MOT_ROOT / dataset_name
    ensure_dir(dataset_root)
    ensure_dir(GT_MOT_ROOT / "seqmaps")

    seq_lengths: dict[str, int] = {}
    for seq_name in sequences:
        seqinfo = read_seqinfo(seq_name)
        seq_lengths[seq_name] = int(seqinfo["seqLength"])
        seq_root = dataset_root / seq_name
        ensure_dir(seq_root)

        if view == "left":
            src_img_dir = RECT_ROOT / seq_name / "img1"
            src_gt_path = DATA_ROOT / locate_split(seq_name) / seq_name / "gt" / "gt_left.txt"
            src_det_path = det_root / seq_name / "det_left.txt"
        else:
            src_img_dir = RECT_ROOT / seq_name / "img2"
            src_gt_path = DATA_ROOT / locate_split(seq_name) / seq_name / "gt" / "gt_right.txt"
            src_det_path = det_root / seq_name / "det_right.txt"

        ensure_img_link(seq_root / "img1", src_img_dir)
        write_seqinfo(seq_root / "seqinfo.ini", seq_name, seqinfo)
        rectify_gt(src_gt_path, seq_root / "gt" / "gt.txt", seq_name, view, seqinfo)
        convert_det_to_mot(src_det_path, seq_root / "det" / "det.txt")

    seqmap_path = GT_MOT_ROOT / "seqmaps" / f"{dataset_name}.txt"
    with open(seqmap_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for seq_name in sequences:
            writer.writerow([seq_name])

    return seq_lengths


def load_detections(det_path: Path) -> dict[int, list[Detection]]:
    frame_map: dict[int, list[Detection]] = {}
    with open(det_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame, _, x, y, w, h, score, *_ = line.split(",")
            det = Detection(
                tlwh=np.array([float(x), float(y), float(w), float(h)], dtype=np.float64),
                score=float(score),
            )
            frame_map.setdefault(int(frame), []).append(det)
    return frame_map


def make_tracker(tracker_name: str):
    if tracker_name == "SORT":
        return SortTracker()
    if tracker_name == "ByteTrack":
        return ByteTrackTracker()
    if tracker_name == "OCSort":
        return OCSortTracker()
    if tracker_name == "IOUTrack":
        return IOUTracker()
    raise ValueError(f"Unsupported tracker: {tracker_name}")


def run_tracker_on_sequence(tracker_name: str, det_path: Path, seq_length: int) -> list[list[object]]:
    tracker = make_tracker(tracker_name)
    dets_by_frame = load_detections(det_path)
    rows: list[list[object]] = []

    for frame_id in range(1, seq_length + 1):
        outputs = tracker.update(dets_by_frame.get(frame_id, []))
        for output in outputs:
            x, y, w, h = output.tlwh.tolist()
            rows.append(
                [
                    frame_id,
                    output.track_id,
                    round(float(x), 2),
                    round(float(y), 2),
                    round(float(w), 2),
                    round(float(h), 2),
                    round(float(output.score), 6),
                    -1,
                    -1,
                    -1,
                ]
            )

    return rows


def save_tracker_rows(rows: list[list[object]], dst_path: Path) -> None:
    ensure_dir(dst_path.parent)
    with open(dst_path, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")


def run_tracking_for_view(view: str, sequences: list[str], seq_lengths: dict[str, int], tracker_names: list[str]) -> None:
    dataset_name = f"CP3D-{view}-all"
    dataset_root = GT_MOT_ROOT / dataset_name
    tracker_root = TRACKERS_MOT_ROOT / dataset_name
    ensure_dir(tracker_root)

    for tracker_name in tracker_names:
        print(f"[Track] {view} -> {tracker_name}")
        for seq_name in sequences:
            det_path = dataset_root / seq_name / "det" / "det.txt"
            out_path = tracker_root / tracker_name / "data" / f"{seq_name}.txt"
            rows = run_tracker_on_sequence(tracker_name, det_path, seq_lengths[seq_name])
            save_tracker_rows(rows, out_path)


def run_trackeval_for_view(view: str, sequences: list[str], tracker_names: list[str]) -> tuple[dict, Path]:
    dataset_name = f"CP3D-{view}-all"
    output_folder = EVAL_ROOT / dataset_name
    ensure_dir(output_folder)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_CONFIG"] = False
    eval_config["TIME_PROGRESS"] = False
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["LOG_ON_ERROR"] = str(STEP3_ROOT / "trackeval_error_log.txt")
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_DETAILED"] = True
    eval_config["PLOT_CURVES"] = False
    eval_config["PRINT_RESULTS"] = False
    eval_config["PRINT_ONLY_COMBINED"] = True

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(GT_MOT_ROOT)
    dataset_config["TRACKERS_FOLDER"] = str(TRACKERS_MOT_ROOT)
    dataset_config["OUTPUT_FOLDER"] = str(output_folder)
    dataset_config["BENCHMARK"] = f"CP3D-{view}"
    dataset_config["SPLIT_TO_EVAL"] = "all"
    dataset_config["SEQMAP_FOLDER"] = str(GT_MOT_ROOT / "seqmaps")
    dataset_config["TRACKERS_TO_EVAL"] = tracker_names
    dataset_config["TRACKER_SUB_FOLDER"] = "data"
    dataset_config["OUTPUT_SUB_FOLDER"] = ""
    dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
    dataset_config["DO_PREPROC"] = False
    dataset_config["PRINT_CONFIG"] = False
    dataset_config["SEQ_INFO"] = {seq_name: None for seq_name in sequences}

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics_list = [
        trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    ]
    results, _ = evaluator.evaluate([dataset], metrics_list, show_progressbar=False)
    return results, output_folder


def extract_metrics_from_summary(output_folder: Path, tracker_name: str) -> dict[str, float]:
    summary_path = output_folder / tracker_name / "pedestrian_summary.txt"
    lines = [line.strip() for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = lines[0].split()
    values = lines[1].split()
    metrics = dict(zip(header, values))
    return {
        "HOTA": float(metrics["HOTA"]),
        "IDF1": float(metrics["IDF1"]),
        "MOTA": float(metrics["MOTA"]),
        "IDSW": float(metrics["IDSW"]),
    }


def write_reports(metrics_rows: list[dict[str, object]]) -> None:
    ensure_dir(REPORT_ROOT)

    long_csv = REPORT_ROOT / "tracker_metrics_long.csv"
    with open(long_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["view", "tracker", "HOTA", "IDF1", "MOTA", "IDSW"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    wide_rows = {}
    for row in metrics_rows:
        tracker = row["tracker"]
        wide_rows.setdefault(tracker, {"tracker": tracker})
        prefix = row["view"]
        for key in ["HOTA", "IDF1", "MOTA", "IDSW"]:
            wide_rows[tracker][f"{prefix}_{key}"] = row[key]

    wide_csv = REPORT_ROOT / "tracker_metrics_wide.csv"
    wide_fieldnames = [
        "tracker",
        "left_HOTA",
        "left_IDF1",
        "left_MOTA",
        "left_IDSW",
        "right_HOTA",
        "right_IDF1",
        "right_MOTA",
        "right_IDSW",
    ]
    with open(wide_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=wide_fieldnames)
        writer.writeheader()
        for tracker_name in TRACKER_NAMES:
            if tracker_name in wide_rows:
                writer.writerow(wide_rows[tracker_name])

    md_path = REPORT_ROOT / "tracker_metrics_paper.md"
    with open(md_path, "w", encoding="utf-8") as f:
        for view in VIEWS:
            f.write(f"## {view.capitalize()} View\n\n")
            f.write("| Tracker | HOTA | IDF1 | MOTA | IDSW |\n")
            f.write("| --- | ---: | ---: | ---: | ---: |\n")
            for row in [r for r in metrics_rows if r["view"] == view]:
                f.write(
                    f"| {row['tracker']} | {row['HOTA']:.3f} | {row['IDF1']:.3f} | "
                    f"{row['MOTA']:.3f} | {int(row['IDSW'])} |\n"
                )
            f.write("\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", nargs="+", default=VIEWS, choices=VIEWS)
    parser.add_argument("--trackers", nargs="+", default=TRACKER_NAMES, choices=TRACKER_NAMES)
    parser.add_argument("--seq", nargs="*", default=[])
    parser.add_argument("--det-root", default=str(DEFAULT_DET_ROOT))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-track", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    sequences = args.seq if args.seq else find_sequences()
    sequences = natural_sort(sequences)
    if not sequences:
        raise FileNotFoundError("No sequences found for step3")
    det_root = Path(args.det_root).resolve()

    ensure_dir(STEP3_ROOT)
    ensure_dir(GT_MOT_ROOT)
    ensure_dir(TRACKERS_MOT_ROOT)
    ensure_dir(EVAL_ROOT)

    metrics_rows: list[dict[str, object]] = []
    run_meta = {
        "views": args.views,
        "trackers": args.trackers,
        "sequences": sequences,
        "det_root": str(det_root),
    }

    for view in args.views:
        if not args.skip_build:
            print(f"[Build] {view}")
            seq_lengths = build_view_dataset(view, sequences, det_root)
        else:
            seq_lengths = {seq: int(read_seqinfo(seq)["seqLength"]) for seq in sequences}

        if not args.skip_track:
            run_tracking_for_view(view, sequences, seq_lengths, args.trackers)

        if not args.skip_eval:
            print(f"[Eval] {view}")
            _, output_folder = run_trackeval_for_view(view, sequences, args.trackers)
            for tracker_name in args.trackers:
                metrics = extract_metrics_from_summary(output_folder, tracker_name)
                metrics_rows.append({"view": view, "tracker": tracker_name, **metrics})
            run_meta.setdefault("eval_outputs", {})[view] = str(output_folder)

    if metrics_rows:
        write_reports(metrics_rows)
        run_meta["reports"] = str(REPORT_ROOT)
        run_meta["metrics_rows"] = metrics_rows

    with open(STEP3_ROOT / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print("DONE")


if __name__ == "__main__":
    main()
