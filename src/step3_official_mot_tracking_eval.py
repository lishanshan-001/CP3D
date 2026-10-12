# -*- coding: utf-8 -*-

"""
Step 3 (official trackers)
Build MOTChallenge-style left/right datasets, run official tracker
implementations, evaluate with TrackEval, and export paper-ready tables.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import step3_mot_tracking_eval as base


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
THIRD_PARTY_ROOT = ROOT / "third_party"
DEFAULT_DET_ROOT = ROOT / "results" / "step2_rawproj_conf010_all"
DEFAULT_SAVE_ROOT = ROOT / "results" / "step3_motchallenge_official"

TRACKER_NAMES = ["SORT", "ByteTrack", "OCSort", "IOUTrack"]
VIEWS = ["left", "right"]


def configure_base_paths(
    save_root: Path,
    data_root: Path,
    rect_root: Path,
    calib_file: Path,
    runtime_root: Path,
) -> None:
    base.DATA_ROOT = data_root
    base.RECT_ROOT = rect_root
    base.CALIB_FILE = calib_file
    base.STEP3_ROOT = save_root
    base.GT_MOT_ROOT = save_root / "gt" / "mot_challenge"
    base.TRACKERS_MOT_ROOT = save_root / "trackers" / "mot_challenge"
    base.EVAL_ROOT = save_root / "eval"
    base.REPORT_ROOT = save_root / "reports"
    base.RUNTIME_ROOT = runtime_root
    base.RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(base.RUNTIME_ROOT / "matplotlib")


def patch_legacy_numpy() -> None:
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]


def ensure_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_sort_module():
    patch_legacy_numpy()
    ensure_sys_path(ROOT)
    import matplotlib

    original_use = matplotlib.use

    def force_agg(_backend, *args, **kwargs):
        return original_use("Agg", force=True)

    matplotlib.use = force_agg  # type: ignore[assignment]
    try:
        return load_module_from_file("sort_official_module", THIRD_PARTY_ROOT / "sort_official" / "sort.py")
    finally:
        matplotlib.use = original_use  # type: ignore[assignment]


@lru_cache(maxsize=1)
def load_bytetrack_symbols():
    patch_legacy_numpy()
    ensure_sys_path(ROOT)
    ensure_sys_path(THIRD_PARTY_ROOT / "bytetrack_official")
    module = importlib.import_module("yolox.tracker.byte_tracker")
    basetrack_module = importlib.import_module("yolox.tracker.basetrack")
    return module.BYTETracker, basetrack_module.BaseTrack


@lru_cache(maxsize=1)
def load_ocsort_class():
    patch_legacy_numpy()
    ensure_sys_path(ROOT)
    ensure_sys_path(THIRD_PARTY_ROOT / "ocsort_official")
    module = importlib.import_module("trackers.ocsort_tracker.ocsort")
    return module.OCSort


@lru_cache(maxsize=1)
def load_iou_modules():
    patch_legacy_numpy()
    ensure_sys_path(THIRD_PARTY_ROOT / "iou_tracker_official")
    tracker_module = importlib.import_module("iou_tracker")
    util_module = importlib.import_module("util")
    return tracker_module, util_module


def load_frame_dets_xyxy(det_path: Path) -> dict[int, np.ndarray]:
    frame_map: dict[int, list[list[float]]] = {}
    with open(det_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame, _, x, y, w, h, score, *_ = line.split(",")
            x1 = float(x)
            y1 = float(y)
            x2 = x1 + float(w)
            y2 = y1 + float(h)
            frame_map.setdefault(int(frame), []).append([x1, y1, x2, y2, float(score)])
    return {frame_id: np.asarray(rows, dtype=np.float64) for frame_id, rows in frame_map.items()}


def load_frame_dets_mot(det_path: Path) -> np.ndarray:
    rows = []
    with open(det_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame, track_id, x, y, w, h, score, *_ = line.split(",")
            rows.append(
                [
                    float(frame),
                    float(track_id),
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                    float(score),
                ]
            )
    if not rows:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def sort_outputs_to_rows(outputs: np.ndarray, frame_id: int, default_score: float = 1.0) -> list[list[object]]:
    rows: list[list[object]] = []
    for x1, y1, x2, y2, track_id in outputs.tolist():
        rows.append(
            [
                frame_id,
                int(track_id),
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2 - x1), 2),
                round(float(y2 - y1), 2),
                round(float(default_score), 6),
                -1,
                -1,
                -1,
            ]
        )
    return rows


def run_sort_official(det_path: Path, seq_length: int) -> list[list[object]]:
    sort_module = load_sort_module()
    tracker = sort_module.Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    dets_by_frame = load_frame_dets_xyxy(det_path)
    rows: list[list[object]] = []

    for frame_id in range(1, seq_length + 1):
        dets = dets_by_frame.get(frame_id, np.empty((0, 5), dtype=np.float64))
        rows.extend(sort_outputs_to_rows(tracker.update(dets), frame_id))
    return rows


def run_bytetrack_official(
    det_path: Path,
    seq_length: int,
    img_h: int,
    img_w: int,
    frame_rate: int,
    track_thresh: float,
    match_thresh: float,
    track_buffer: int,
) -> list[list[object]]:
    tracker_cls, basetrack_cls = load_bytetrack_symbols()
    basetrack_cls._count = 0
    tracker = tracker_cls(
        SimpleNamespace(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            mot20=False,
        ),
        frame_rate=frame_rate,
    )
    dets_by_frame = load_frame_dets_xyxy(det_path)
    rows: list[list[object]] = []

    for frame_id in range(1, seq_length + 1):
        dets = dets_by_frame.get(frame_id, np.empty((0, 5), dtype=np.float64))
        outputs = tracker.update(dets, (img_h, img_w), (img_h, img_w))
        for track in outputs:
            x, y, w, h = track.tlwh.tolist()
            rows.append(
                [
                    frame_id,
                    int(track.track_id),
                    round(float(x), 2),
                    round(float(y), 2),
                    round(float(w), 2),
                    round(float(h), 2),
                    round(float(getattr(track, "score", 1.0)), 6),
                    -1,
                    -1,
                    -1,
                ]
            )
    return rows


def run_ocsort_official(
    det_path: Path,
    seq_length: int,
    img_h: int,
    img_w: int,
    det_thresh: float,
    iou_thresh: float,
    max_age: int,
    min_hits: int,
    delta_t: int,
    inertia: float,
    use_byte: bool,
) -> list[list[object]]:
    tracker_cls = load_ocsort_class()
    tracker = tracker_cls(
        det_thresh=det_thresh,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_thresh,
        delta_t=delta_t,
        asso_func="iou",
        inertia=inertia,
        use_byte=use_byte,
    )
    dets_by_frame = load_frame_dets_xyxy(det_path)
    rows: list[list[object]] = []

    for frame_id in range(1, seq_length + 1):
        dets = dets_by_frame.get(frame_id, np.empty((0, 5), dtype=np.float64))
        outputs = tracker.update(dets, (img_h, img_w), (img_h, img_w))
        rows.extend(sort_outputs_to_rows(outputs, frame_id))
    return rows


def run_iou_official(
    det_path: Path,
    sigma_l: float,
    sigma_h: float,
    sigma_iou: float,
    t_min: int,
) -> list[list[object]]:
    tracker_module, util_module = load_iou_modules()
    detections = util_module.load_mot(load_frame_dets_mot(det_path), with_classes=False)
    tracks = tracker_module.track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min)

    rows: list[list[object]] = []
    for track_id, track in enumerate(tracks, start=1):
        for offset, bbox in enumerate(track["bboxes"]):
            x1, y1, x2, y2 = bbox
            rows.append(
                [
                    int(track["start_frame"] + offset),
                    track_id,
                    round(float(x1 + 1.0), 2),
                    round(float(y1 + 1.0), 2),
                    round(float(x2 - x1), 2),
                    round(float(y2 - y1), 2),
                    round(float(track["max_score"]), 6),
                    -1,
                    -1,
                    -1,
                ]
            )
    rows.sort(key=lambda row: (int(row[0]), int(row[1])))
    return rows


def run_official_tracker_on_sequence(
    tracker_name: str,
    det_path: Path,
    seqinfo: dict[str, str],
    byte_track_thresh: float,
    byte_match_thresh: float,
    byte_track_buffer: int,
    ocsort_det_thresh: float,
    ocsort_iou_thresh: float,
    ocsort_max_age: int,
    ocsort_min_hits: int,
    ocsort_delta_t: int,
    ocsort_inertia: float,
    ocsort_use_byte: bool,
    iou_sigma_l: float,
    iou_sigma_h: float,
    iou_sigma: float,
    iou_t_min: int,
) -> list[list[object]]:
    seq_length = int(seqinfo["seqLength"])
    img_w = int(seqinfo["imWidth"])
    img_h = int(seqinfo["imHeight"])
    frame_rate = int(seqinfo["frameRate"])

    if tracker_name == "SORT":
        return run_sort_official(det_path, seq_length)
    if tracker_name == "ByteTrack":
        return run_bytetrack_official(
            det_path,
            seq_length,
            img_h,
            img_w,
            frame_rate,
            byte_track_thresh,
            byte_match_thresh,
            byte_track_buffer,
        )
    if tracker_name == "OCSort":
        return run_ocsort_official(
            det_path,
            seq_length,
            img_h,
            img_w,
            ocsort_det_thresh,
            ocsort_iou_thresh,
            ocsort_max_age,
            ocsort_min_hits,
            ocsort_delta_t,
            ocsort_inertia,
            ocsort_use_byte,
        )
    if tracker_name == "IOUTrack":
        return run_iou_official(det_path, iou_sigma_l, iou_sigma_h, iou_sigma, iou_t_min)
    raise ValueError(f"Unsupported tracker: {tracker_name}")


def run_tracking_for_view(
    view: str,
    sequences: list[str],
    tracker_names: list[str],
    args: argparse.Namespace,
) -> None:
    dataset_name = f"CP3D-{view}-all"
    dataset_root = base.GT_MOT_ROOT / dataset_name
    tracker_root = base.TRACKERS_MOT_ROOT / dataset_name
    base.ensure_dir(tracker_root)

    for tracker_name in tracker_names:
        print(f"[Official Track] {view} -> {tracker_name}")
        for seq_name in sequences:
            det_path = dataset_root / seq_name / "det" / "det.txt"
            out_path = tracker_root / tracker_name / "data" / f"{seq_name}.txt"
            seqinfo = base.read_seqinfo(seq_name)
            rows = run_official_tracker_on_sequence(
                tracker_name,
                det_path,
                seqinfo,
                args.bytetrack_track_thresh,
                args.bytetrack_match_thresh,
                args.bytetrack_track_buffer,
                args.ocsort_det_thresh,
                args.ocsort_iou_thresh,
                args.ocsort_max_age,
                args.ocsort_min_hits,
                args.ocsort_delta_t,
                args.ocsort_inertia,
                args.ocsort_use_byte,
                args.iou_sigma_l,
                args.iou_sigma_h,
                args.iou_sigma_iou,
                args.iou_t_min,
            )
            base.save_tracker_rows(rows, out_path)


def write_latex_table(metrics_rows: list[dict[str, object]], output_path: Path) -> None:
    by_view = {view: [row for row in metrics_rows if row["view"] == view] for view in VIEWS}
    lines = []
    for view in VIEWS:
        lines.append(f"% {view} view")
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\hline")
        lines.append("Tracker & HOTA & IDF1 & MOTA & IDSW \\\\")
        lines.append("\\hline")
        for row in by_view[view]:
            lines.append(
                f"{row['tracker']} & {row['HOTA']:.3f} & {row['IDF1']:.3f} & "
                f"{row['MOTA']:.3f} & {int(row['IDSW'])} \\\\"
            )
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", nargs="+", default=VIEWS, choices=VIEWS)
    parser.add_argument("--trackers", nargs="+", default=TRACKER_NAMES, choices=TRACKER_NAMES)
    parser.add_argument("--seq", nargs="*", default=[])
    parser.add_argument("--data-root", default=str(ROOT / "CP3D_dataset"))
    parser.add_argument("--rect-root", default=str(ROOT / "results" / "step1_rectification"))
    parser.add_argument("--calib-file", default=str(ROOT / "calibration" / "stereo_params.txt"))
    parser.add_argument("--third-party-root", default=str(THIRD_PARTY_ROOT))
    parser.add_argument("--runtime-root", default=str(ROOT / ".runtime"))
    parser.add_argument("--det-root", default=str(DEFAULT_DET_ROOT))
    parser.add_argument("--save-root", default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-track", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--bytetrack-track-thresh", type=float, default=0.5)
    parser.add_argument("--bytetrack-match-thresh", type=float, default=0.8)
    parser.add_argument("--bytetrack-track-buffer", type=int, default=30)
    parser.add_argument("--ocsort-det-thresh", type=float, default=0.6)
    parser.add_argument("--ocsort-iou-thresh", type=float, default=0.3)
    parser.add_argument("--ocsort-max-age", type=int, default=30)
    parser.add_argument("--ocsort-min-hits", type=int, default=3)
    parser.add_argument("--ocsort-delta-t", type=int, default=3)
    parser.add_argument("--ocsort-inertia", type=float, default=0.2)
    parser.add_argument("--ocsort-use-byte", action="store_true")
    parser.add_argument("--iou-sigma-l", type=float, default=0.3)
    parser.add_argument("--iou-sigma-h", type=float, default=0.5)
    parser.add_argument("--iou-sigma-iou", type=float, default=0.3)
    parser.add_argument("--iou-t-min", type=int, default=5)
    return parser


def main() -> None:
    global THIRD_PARTY_ROOT

    parser = build_argparser()
    args = parser.parse_args()

    save_root = Path(args.save_root).resolve()
    det_root = Path(args.det_root).resolve()
    data_root = Path(args.data_root).resolve()
    rect_root = Path(args.rect_root).resolve()
    calib_file = Path(args.calib_file).resolve()
    THIRD_PARTY_ROOT = Path(args.third_party_root).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    configure_base_paths(save_root, data_root, rect_root, calib_file, runtime_root)

    sequences = args.seq if args.seq else base.find_sequences()
    sequences = base.natural_sort(sequences)
    if not sequences:
        raise FileNotFoundError("No sequences found for official step3")

    base.ensure_dir(base.STEP3_ROOT)
    base.ensure_dir(base.GT_MOT_ROOT)
    base.ensure_dir(base.TRACKERS_MOT_ROOT)
    base.ensure_dir(base.EVAL_ROOT)

    metrics_rows: list[dict[str, object]] = []
    run_meta = {
        "implementation": "official_repos_with_runtime_compat_shim",
        "views": args.views,
        "trackers": args.trackers,
        "sequences": sequences,
        "data_root": str(data_root),
        "rect_root": str(rect_root),
        "calibration_file": str(calib_file),
        "third_party_root": str(THIRD_PARTY_ROOT),
        "det_root": str(det_root),
        "save_root": str(save_root),
        "params": {
            "bytetrack_track_thresh": args.bytetrack_track_thresh,
            "bytetrack_match_thresh": args.bytetrack_match_thresh,
            "bytetrack_track_buffer": args.bytetrack_track_buffer,
            "ocsort_det_thresh": args.ocsort_det_thresh,
            "ocsort_iou_thresh": args.ocsort_iou_thresh,
            "ocsort_max_age": args.ocsort_max_age,
            "ocsort_min_hits": args.ocsort_min_hits,
            "ocsort_delta_t": args.ocsort_delta_t,
            "ocsort_inertia": args.ocsort_inertia,
            "ocsort_use_byte": args.ocsort_use_byte,
            "iou_sigma_l": args.iou_sigma_l,
            "iou_sigma_h": args.iou_sigma_h,
            "iou_sigma_iou": args.iou_sigma_iou,
            "iou_t_min": args.iou_t_min,
        },
    }

    for view in args.views:
        if not args.skip_build:
            print(f"[Build] {view}")
            base.build_view_dataset(view, sequences, det_root)
        if not args.skip_track:
            run_tracking_for_view(view, sequences, args.trackers, args)
        if not args.skip_eval:
            print(f"[Eval] {view}")
            _, output_folder = base.run_trackeval_for_view(view, sequences, args.trackers)
            for tracker_name in args.trackers:
                metrics = base.extract_metrics_from_summary(output_folder, tracker_name)
                metrics_rows.append({"view": view, "tracker": tracker_name, **metrics})
            run_meta.setdefault("eval_outputs", {})[view] = str(output_folder)

    if metrics_rows:
        base.write_reports(metrics_rows)
        write_latex_table(metrics_rows, base.REPORT_ROOT / "tracker_metrics_paper.tex")
        run_meta["reports"] = str(base.REPORT_ROOT)
        run_meta["metrics_rows"] = metrics_rows

    with open(base.STEP3_ROOT / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print("DONE")


if __name__ == "__main__":
    main()
