"""
Point-target 3D MOT comparison for the CP3D stereo pipeline.

The three classical baselines share the same frame-wise stereo observations:
1. Nearest-neighbour 3D tracking.
2. Euclidean 3D Kalman-Hungarian tracking (AB3DMOT-style core).
3. Mahalanobis 3D Kalman-Hungarian tracking.

The proposed method is loaded from the final trajectory-level association,
trajectory-quality assessment, and reliability-guided reconstruction output.
All matcher and tracker parameters are selected on the eight training
sequences. The four test sequences are evaluated once with fixed parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
RUNTIME_ROOT = CODE_ROOT / ".runtime"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_ROOT / "matplotlib"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.metrics import CLEAR, HOTA, Identity

import step4_stereo_association_eval as s4
import step5_reliability_guided_3d_reconstruction as s5


METHODS = [
    "Nearest-neighbour 3D tracking",
    "Euclidean 3D Kalman-Hungarian",
    "Mahalanobis 3D Kalman-Hungarian",
    "Proposed reliability-guided framework",
]
CLASSICAL_METHODS = METHODS[:3]
EVAL_DISTANCE_MM = 25.0


@dataclass
class Observation3D:
    seq_name: str
    frame: int
    xyz: np.ndarray
    score: float
    left_track_id: int
    right_track_id: int


@dataclass
class TrackPoint3D:
    seq_name: str
    frame: int
    track_id: int
    xyz: np.ndarray
    score: float = 1.0


@dataclass
class PointTrack:
    track_id: int
    xyz: np.ndarray
    last_observed_frame: int
    state_frame: int
    covariance: np.ndarray | None = None
    state: np.ndarray | None = None
    score: float = 1.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def triangulate(P1: np.ndarray, P2: np.ndarray, left: s4.Detection, right: s4.Detection) -> np.ndarray:
    points_left = np.asarray([[left.cx], [left.cy]], dtype=np.float64)
    points_right = np.asarray([[right.cx], [right.cy]], dtype=np.float64)
    homogeneous = cv2.triangulatePoints(P1, P2, points_left, points_right)
    if abs(float(homogeneous[3, 0])) <= 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return (homogeneous[:3, 0] / homogeneous[3, 0]).astype(np.float64)


def framewise_pairs(
    ctx: s4.SequenceContext,
    frame: int,
    params: dict[str, float],
) -> list[tuple[s4.Detection, s4.Detection, float]]:
    left_dets = ctx.left_frame_index.get(frame, [])
    right_dets = ctx.right_frame_index.get(frame, [])
    if not left_dets or not right_dets:
        return []

    prior = ctx.group_prior
    v_gate = max(3.0, prior.v_p95 * params["v_multiplier"])
    disp_margin = prior.disp_iqr * params["disp_margin_iqr"]
    disp_low = prior.disp_p05 - disp_margin
    disp_high = prior.disp_p95 + disp_margin
    disp_scale = max(8.0, 0.5 * (disp_high - disp_low))
    cost = np.full((len(left_dets), len(right_dets)), 1e6, dtype=np.float64)

    for row, left in enumerate(left_dets):
        for col, right in enumerate(right_dets):
            vdiff = abs(left.cy - right.cy)
            disparity = left.cx - right.cx
            if vdiff > v_gate or disparity < disp_low or disparity > disp_high:
                continue
            disp_dev = abs(disparity - prior.disp_median)
            value = (
                params["vertical_weight"] * vdiff / v_gate
                + (1.0 - params["vertical_weight"]) * disp_dev / disp_scale
            )
            cost[row, col] = value

    rows, cols = linear_sum_assignment(cost)
    pairs: list[tuple[s4.Detection, s4.Detection, float]] = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        value = float(cost[row, col])
        if value > params["cost_threshold"]:
            continue
        confidence = math.sqrt(max(0.0, left_dets[row].score * right_dets[col].score))
        pairs.append((left_dets[row], right_dets[col], confidence))
    return pairs


def active_dominant_ids(
    ctx: s4.SequenceContext,
    frame: int,
    view: str,
) -> dict[int, int]:
    dets = ctx.left_frame_index.get(frame, []) if view == "left" else ctx.right_frame_index.get(frame, [])
    tracks = ctx.left_tracks if view == "left" else ctx.right_tracks
    result: dict[int, int] = {}
    for det in dets:
        gt_id = tracks[det.track_id].dominant_gt_id
        if gt_id is not None:
            result[det.track_id] = int(gt_id)
    return result


def evaluate_stereo_pairing(
    contexts: list[s4.SequenceContext],
    params: dict[str, float],
) -> dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for ctx in contexts:
        for frame in range(1, ctx.seq_length + 1):
            left_labels = active_dominant_ids(ctx, frame, "left")
            right_labels = active_dominant_ids(ctx, frame, "right")
            available_gt = set(left_labels.values()) & set(right_labels.values())
            recovered_gt: set[int] = set()
            for left, right, _score in framewise_pairs(ctx, frame, params):
                left_gt = left_labels.get(left.track_id)
                right_gt = right_labels.get(right.track_id)
                if left_gt is not None and left_gt == right_gt:
                    true_positive += 1
                    recovered_gt.add(left_gt)
                else:
                    false_positive += 1
            false_negative += len(available_gt - recovered_gt)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tune_stereo_matcher(contexts: list[s4.SequenceContext]) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for v_multiplier in [0.75, 1.0, 1.25, 1.5]:
        for disp_margin_iqr in [0.0, 0.5, 1.0]:
            for cost_threshold in [0.75, 1.0, 1.25]:
                params = {
                    "v_multiplier": v_multiplier,
                    "disp_margin_iqr": disp_margin_iqr,
                    "cost_threshold": cost_threshold,
                    "vertical_weight": 0.65,
                }
                metrics = evaluate_stereo_pairing(contexts, params)
                rows.append({**params, **metrics})
    best = max(
        rows,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            -float(row["disp_margin_iqr"]),
            -float(row["v_multiplier"]),
        ),
    )
    params = {
        key: float(best[key])
        for key in ["v_multiplier", "disp_margin_iqr", "cost_threshold", "vertical_weight"]
    }
    return params, rows


def load_complete_stereo_tuning(path: Path) -> tuple[dict[str, float], list[dict[str, object]]] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36:
        return None
    required = {
        "v_multiplier",
        "disp_margin_iqr",
        "cost_threshold",
        "vertical_weight",
        "precision",
        "recall",
        "f1",
    }
    if not rows or not required.issubset(rows[0]):
        return None
    converted: list[dict[str, object]] = []
    for row in rows:
        try:
            converted.append({key: float(value) for key, value in row.items()})
        except (TypeError, ValueError):
            return None
    best = max(
        converted,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            -float(row["disp_margin_iqr"]),
            -float(row["v_multiplier"]),
        ),
    )
    params = {
        key: float(best[key])
        for key in ["v_multiplier", "disp_margin_iqr", "cost_threshold", "vertical_weight"]
    }
    return params, converted


def build_observations(
    contexts: list[s4.SequenceContext],
    stereo_params: dict[str, float],
) -> dict[str, dict[int, list[Observation3D]]]:
    output: dict[str, dict[int, list[Observation3D]]] = {}
    for ctx in contexts:
        P1, P2 = s5.load_projection(ctx.seq_name)
        frame_map: dict[int, list[Observation3D]] = defaultdict(list)
        for frame in range(1, ctx.seq_length + 1):
            for left, right, score in framewise_pairs(ctx, frame, stereo_params):
                xyz = triangulate(P1, P2, left, right)
                if not np.all(np.isfinite(xyz)) or float(xyz[2]) <= 0.0:
                    continue
                frame_map[frame].append(
                    Observation3D(
                        seq_name=ctx.seq_name,
                        frame=frame,
                        xyz=xyz,
                        score=score,
                        left_track_id=left.track_id,
                        right_track_id=right.track_id,
                    )
                )
        output[ctx.seq_name] = dict(frame_map)
    return output


def transition(dt: float) -> np.ndarray:
    matrix = np.eye(6, dtype=np.float64)
    matrix[0, 3] = dt
    matrix[1, 4] = dt
    matrix[2, 5] = dt
    return matrix


def process_noise(dt: float, variance: float) -> np.ndarray:
    block = np.asarray(
        [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]],
        dtype=np.float64,
    ) * variance
    matrix = np.zeros((6, 6), dtype=np.float64)
    for axis in range(3):
        matrix[axis, axis] = block[0, 0]
        matrix[axis, axis + 3] = block[0, 1]
        matrix[axis + 3, axis] = block[1, 0]
        matrix[axis + 3, axis + 3] = block[1, 1]
    return matrix


def initialize_kalman(track_id: int, observation: Observation3D, measurement_variance: float) -> PointTrack:
    state = np.zeros(6, dtype=np.float64)
    state[:3] = observation.xyz
    covariance = np.diag(
        [measurement_variance, measurement_variance, measurement_variance, 400.0, 400.0, 400.0]
    ).astype(np.float64)
    return PointTrack(
        track_id=track_id,
        xyz=observation.xyz.copy(),
        last_observed_frame=observation.frame,
        state_frame=observation.frame,
        covariance=covariance,
        state=state,
        score=observation.score,
    )


def predict_kalman(track: PointTrack, frame: int, process_variance: float) -> None:
    if track.state is None or track.covariance is None:
        raise ValueError("Kalman state is not initialized")
    dt = float(max(0, frame - track.state_frame))
    if dt <= 0:
        return
    F = transition(dt)
    track.state = F @ track.state
    track.covariance = F @ track.covariance @ F.T + process_noise(dt, process_variance)
    track.xyz = track.state[:3].copy()
    track.state_frame = frame


def update_kalman(track: PointTrack, observation: Observation3D, measurement_variance: float) -> None:
    if track.state is None or track.covariance is None:
        raise ValueError("Kalman state is not initialized")
    H = np.zeros((3, 6), dtype=np.float64)
    H[:, :3] = np.eye(3)
    R = np.eye(3, dtype=np.float64) * measurement_variance
    innovation = observation.xyz - H @ track.state
    S = H @ track.covariance @ H.T + R
    gain = track.covariance @ H.T @ np.linalg.pinv(S)
    track.state = track.state + gain @ innovation
    identity = np.eye(6, dtype=np.float64)
    track.covariance = (identity - gain @ H) @ track.covariance
    track.xyz = track.state[:3].copy()
    track.last_observed_frame = observation.frame
    track.score = observation.score


def run_tracker(
    seq_name: str,
    observations: dict[int, list[Observation3D]],
    method: str,
    params: dict[str, float],
    seq_length: int,
) -> list[TrackPoint3D]:
    active: dict[int, PointTrack] = {}
    output: list[TrackPoint3D] = []
    next_id = 1
    max_age = int(params["max_age"])
    measurement_variance = float(params.get("measurement_variance", 100.0))
    process_variance = float(params.get("process_variance", 10.0))

    for frame in range(1, seq_length + 1):
        detections = observations.get(frame, [])
        stale = [
            track_id
            for track_id, track in active.items()
            if frame - track.last_observed_frame > max_age
        ]
        for track_id in stale:
            del active[track_id]

        track_ids = sorted(active)
        if method != "Nearest-neighbour 3D tracking":
            for track_id in track_ids:
                predict_kalman(active[track_id], frame, process_variance)

        cost = np.full((len(track_ids), len(detections)), 1e9, dtype=np.float64)
        if track_ids and detections:
            track_xyz = np.asarray([active[track_id].xyz for track_id in track_ids], dtype=np.float64)
            detection_xyz = np.asarray([detection.xyz for detection in detections], dtype=np.float64)
            deltas = detection_xyz[np.newaxis, :, :] - track_xyz[:, np.newaxis, :]
            if method == "Mahalanobis 3D Kalman-Hungarian":
                for row, track_id in enumerate(track_ids):
                    track = active[track_id]
                    assert track.covariance is not None
                    innovation_cov = track.covariance[:3, :3] + np.eye(3) * measurement_variance
                    inverse_covariance = np.linalg.pinv(innovation_cov)
                    cost[row, :] = np.einsum(
                        "ni,ij,nj->n",
                        deltas[row],
                        inverse_covariance,
                        deltas[row],
                    )
            else:
                cost = np.linalg.norm(deltas, axis=2)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            gate = float(
                params["mahalanobis_gate"]
                if method == "Mahalanobis 3D Kalman-Hungarian"
                else params["distance_gate_mm"]
            )
            for row, col in zip(rows.tolist(), cols.tolist()):
                if float(cost[row, col]) > gate:
                    continue
                track_id = track_ids[row]
                detection = detections[col]
                track = active[track_id]
                if method == "Nearest-neighbour 3D tracking":
                    track.xyz = detection.xyz.copy()
                    track.last_observed_frame = frame
                    track.state_frame = frame
                    track.score = detection.score
                else:
                    update_kalman(track, detection, measurement_variance)
                output.append(
                    TrackPoint3D(
                        seq_name=seq_name,
                        frame=frame,
                        track_id=track_id,
                        xyz=track.xyz.copy(),
                        score=detection.score,
                    )
                )
                matched_tracks.add(track_id)
                matched_detections.add(col)

        for col, detection in enumerate(detections):
            if col in matched_detections:
                continue
            if method == "Nearest-neighbour 3D tracking":
                active[next_id] = PointTrack(
                    track_id=next_id,
                    xyz=detection.xyz.copy(),
                    last_observed_frame=frame,
                    state_frame=frame,
                    score=detection.score,
                )
            else:
                active[next_id] = initialize_kalman(next_id, detection, measurement_variance)
            output.append(
                TrackPoint3D(
                    seq_name=seq_name,
                    frame=frame,
                    track_id=next_id,
                    xyz=active[next_id].xyz.copy(),
                    score=detection.score,
                )
            )
            next_id += 1
    return output


def load_proposed_points(path: Path, sequences: Iterable[str]) -> dict[str, list[TrackPoint3D]]:
    wanted = set(sequences)
    best: dict[tuple[str, int, int], tuple[float, TrackPoint3D]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "Reliability-guided refinement":
                continue
            seq_name = row["seq_name"]
            if seq_name not in wanted:
                continue
            frame = int(row["frame"])
            track_id = int(row["component_id"])
            reliability = float(row["reliability"])
            point = TrackPoint3D(
                seq_name=seq_name,
                frame=frame,
                track_id=track_id,
                xyz=np.asarray(
                    [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])],
                    dtype=np.float64,
                ),
                score=reliability,
            )
            key = (seq_name, track_id, frame)
            previous = best.get(key)
            if previous is None or reliability > previous[0]:
                best[key] = (reliability, point)
    output: dict[str, list[TrackPoint3D]] = {seq: [] for seq in wanted}
    for _reliability, point in best.values():
        if np.all(np.isfinite(point.xyz)) and float(point.xyz[2]) > 0.0:
            output[point.seq_name].append(point)
    for points in output.values():
        points.sort(key=lambda item: (item.frame, item.track_id))
    return output


def gt_by_frame(gt_map: dict[int, dict[int, np.ndarray]]) -> dict[int, list[tuple[int, np.ndarray]]]:
    output: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for gt_id, frames in gt_map.items():
        for frame, xyz in frames.items():
            if np.all(np.isfinite(xyz)) and float(xyz[2]) > 0.0:
                output[frame].append((int(gt_id), xyz))
    return dict(output)


def build_trackeval_data(
    gt_frames: dict[int, list[tuple[int, np.ndarray]]],
    predictions: list[TrackPoint3D],
    seq_length: int,
    distance_gate_mm: float,
) -> dict[str, object]:
    pred_frames: dict[int, list[TrackPoint3D]] = defaultdict(list)
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
        similarities.append(np.maximum(0.0, 1.0 - distances / (2.0 * distance_gate_mm)))

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


def spatial_error_terms(
    gt_frames: dict[int, list[tuple[int, np.ndarray]]],
    predictions: list[TrackPoint3D],
    distance_gate_mm: float,
) -> tuple[list[float], int]:
    pred_frames: dict[int, list[TrackPoint3D]] = defaultdict(list)
    for point in predictions:
        pred_frames[point.frame].append(point)
    errors: list[float] = []
    matches = 0
    for frame, gt_items in gt_frames.items():
        pred_items = pred_frames.get(frame, [])
        if not gt_items or not pred_items:
            continue
        gt_xyz = np.asarray([item[1] for item in gt_items], dtype=np.float64)
        pred_xyz = np.asarray([item.xyz for item in pred_items], dtype=np.float64)
        distances = np.linalg.norm(gt_xyz[:, None, :] - pred_xyz[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(distances)
        for row, col in zip(rows.tolist(), cols.tolist()):
            distance = float(distances[row, col])
            if distance <= distance_gate_mm:
                errors.append(distance)
                matches += 1
    return errors, matches


def jitter_terms(predictions: list[TrackPoint3D]) -> list[float]:
    grouped: dict[int, list[TrackPoint3D]] = defaultdict(list)
    for point in predictions:
        grouped[point.track_id].append(point)
    values: list[float] = []
    for points in grouped.values():
        ordered = sorted(points, key=lambda item: item.frame)
        for index in range(1, len(ordered) - 1):
            if not (
                ordered[index].frame == ordered[index - 1].frame + 1
                and ordered[index + 1].frame == ordered[index].frame + 1
            ):
                continue
            acceleration = ordered[index + 1].xyz - 2.0 * ordered[index].xyz + ordered[index - 1].xyz
            values.append(float(np.linalg.norm(acceleration)))
    return values


def evaluate_one_sequence(
    seq_name: str,
    gt_map: dict[int, dict[int, np.ndarray]],
    predictions: list[TrackPoint3D],
    seq_length: int,
    distance_gate_mm: float,
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    gt_frames = gt_by_frame(gt_map)
    data = build_trackeval_data(gt_frames, predictions, seq_length, distance_gate_mm)
    hota_metric = HOTA()
    identity_metric = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    clear_metric = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    hota = hota_metric.eval_sequence(data)
    identity = identity_metric.eval_sequence(data)
    clear = clear_metric.eval_sequence(data)
    errors, matched_points = spatial_error_terms(gt_frames, predictions, distance_gate_mm)
    jitters = jitter_terms(predictions)
    gt_count = int(data["num_gt_dets"])
    row = {
        "sequence": seq_name,
        "HOTA_3D": 100.0 * float(np.mean(hota["HOTA"])),
        "AssA_3D": 100.0 * float(np.mean(hota["AssA"])),
        "IDF1_3D": 100.0 * float(identity["IDF1"]),
        "MOTA_3D": 100.0 * float(clear["MOTA"]),
        "IDSW": float(clear["IDSW"]),
        "RMSE_mm": float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan"),
        "MAE_mm": float(np.mean(errors)) if errors else float("nan"),
        "MedAE_mm": float(np.median(errors)) if errors else float("nan"),
        "Jitter_mm": float(np.mean(jitters)) if jitters else float("nan"),
        "Coverage_pct": 100.0 * matched_points / max(1, gt_count),
        "GT_points": float(gt_count),
        "Predicted_points": float(data["num_tracker_dets"]),
        "Matched_points": float(matched_points),
    }
    raw = {"hota": hota, "identity": identity, "clear": clear}
    return row, raw


def evaluate_method(
    method: str,
    predictions_by_seq: dict[str, list[TrackPoint3D]],
    gt_maps: dict[str, dict[int, dict[int, np.ndarray]]],
    seq_lengths: dict[str, int],
    distance_gate_mm: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    per_sequence: list[dict[str, float]] = []
    raw_results: dict[str, dict[str, dict[str, object]]] = {}
    pooled_errors: list[float] = []
    pooled_jitter: list[float] = []
    total_matched = 0
    total_gt = 0

    for seq_name in sorted(gt_maps):
        row, raw = evaluate_one_sequence(
            seq_name,
            gt_maps[seq_name],
            predictions_by_seq.get(seq_name, []),
            seq_lengths[seq_name],
            distance_gate_mm,
        )
        row["method"] = method
        per_sequence.append(row)
        raw_results[seq_name] = raw
        errors, matched = spatial_error_terms(
            gt_by_frame(gt_maps[seq_name]),
            predictions_by_seq.get(seq_name, []),
            distance_gate_mm,
        )
        pooled_errors.extend(errors)
        pooled_jitter.extend(jitter_terms(predictions_by_seq.get(seq_name, [])))
        total_matched += matched
        total_gt += int(row["GT_points"])

    hota_metric = HOTA()
    identity_metric = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    clear_metric = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    combined_hota = hota_metric.combine_sequences({seq: raw["hota"] for seq, raw in raw_results.items()})
    combined_identity = identity_metric.combine_sequences(
        {seq: raw["identity"] for seq, raw in raw_results.items()}
    )
    combined_clear = clear_metric.combine_sequences({seq: raw["clear"] for seq, raw in raw_results.items()})
    summary = {
        "method": method,
        "HOTA_3D": 100.0 * float(np.mean(combined_hota["HOTA"])),
        "AssA_3D": 100.0 * float(np.mean(combined_hota["AssA"])),
        "IDF1_3D": 100.0 * float(combined_identity["IDF1"]),
        "MOTA_3D": 100.0 * float(combined_clear["MOTA"]),
        "IDSW": float(combined_clear["IDSW"]),
        "RMSE_mm": float(np.sqrt(np.mean(np.square(pooled_errors)))) if pooled_errors else float("nan"),
        "MAE_mm": float(np.mean(pooled_errors)) if pooled_errors else float("nan"),
        "MedAE_mm": float(np.median(pooled_errors)) if pooled_errors else float("nan"),
        "Jitter_mm": float(np.mean(pooled_jitter)) if pooled_jitter else float("nan"),
        "Coverage_pct": 100.0 * total_matched / max(1, total_gt),
        "GT_points": float(total_gt),
        "Predicted_points": float(sum(row["Predicted_points"] for row in per_sequence)),
        "Matched_points": float(total_matched),
    }
    return summary, per_sequence


def evaluate_training_selection(
    predictions_by_seq: dict[str, list[TrackPoint3D]],
    gt_maps: dict[str, dict[int, dict[int, np.ndarray]]],
    seq_lengths: dict[str, int],
    distance_gate_mm: float,
) -> dict[str, float]:
    """Compute only the metrics needed for train-split parameter selection."""
    clear_metric = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    clear_results = {}
    predicted_points = 0
    for seq_name in sorted(gt_maps):
        data = build_trackeval_data(
            gt_by_frame(gt_maps[seq_name]),
            predictions_by_seq.get(seq_name, []),
            seq_lengths[seq_name],
            distance_gate_mm,
        )
        clear_results[seq_name] = clear_metric.eval_sequence(data)
        predicted_points += int(data["num_tracker_dets"])
    clear = clear_metric.combine_sequences(clear_results)
    return {
        "MOTA_3D": 100.0 * float(clear["MOTA"]),
        "IDSW": float(clear["IDSW"]),
        "Predicted_points": float(predicted_points),
    }


def tune_tracker(
    method: str,
    observations: dict[str, dict[int, list[Observation3D]]],
    contexts: list[s4.SequenceContext],
    gt_maps: dict[str, dict[int, dict[int, np.ndarray]]],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    seq_lengths = {ctx.seq_name: ctx.seq_length for ctx in contexts}
    candidates: list[dict[str, float]] = []
    if method == "Nearest-neighbour 3D tracking":
        for distance_gate in [50.0, 75.0, 100.0, 125.0]:
            for max_age in [3.0, 5.0]:
                candidates.append({"distance_gate_mm": distance_gate, "max_age": max_age})
    elif method == "Euclidean 3D Kalman-Hungarian":
        for distance_gate in [50.0, 75.0, 100.0, 125.0]:
            for max_age in [3.0, 5.0]:
                candidates.append(
                    {
                        "distance_gate_mm": distance_gate,
                        "max_age": max_age,
                        "measurement_variance": 100.0,
                        "process_variance": 10.0,
                    }
                )
    else:
        for mahalanobis_gate in [7.815, 11.345, 16.266]:
            for max_age in [3.0, 5.0]:
                candidates.append(
                    {
                        "mahalanobis_gate": mahalanobis_gate,
                        "max_age": max_age,
                        "measurement_variance": 100.0,
                        "process_variance": 10.0,
                    }
                )

    rows: list[dict[str, object]] = []
    for index, params in enumerate(candidates, start=1):
        predictions = {
            ctx.seq_name: run_tracker(
                ctx.seq_name,
                observations[ctx.seq_name],
                method,
                params,
                ctx.seq_length,
            )
            for ctx in contexts
        }
        summary = evaluate_training_selection(
            predictions,
            gt_maps,
            seq_lengths,
            EVAL_DISTANCE_MM,
        )
        rows.append({"candidate": index, "method": method, **params, **summary})
    best = max(
        rows,
        key=lambda row: (
            float(row["MOTA_3D"]),
            -float(row["IDSW"]),
            -float(row["Predicted_points"]),
        ),
    )
    selected = {
        key: float(best[key])
        for key in [
            "distance_gate_mm",
            "mahalanobis_gate",
            "max_age",
            "measurement_variance",
            "process_variance",
        ]
        if key in best and best[key] != ""
    }
    return selected, rows


def consecutive_segments(frame_map: dict[int, np.ndarray], minimum_points: int = 30) -> list[dict[int, np.ndarray]]:
    segments: list[dict[int, np.ndarray]] = []
    current: dict[int, np.ndarray] = {}
    previous = None
    for frame in sorted(frame_map):
        if previous is not None and frame != previous + 1:
            if len(current) >= minimum_points:
                segments.append(current)
            current = {}
        current[frame] = frame_map[frame]
        previous = frame
    if len(current) >= minimum_points:
        segments.append(current)
    return segments


def select_visualization_segment(
    gt_map: dict[int, dict[int, np.ndarray]],
    method_predictions: dict[str, list[TrackPoint3D]],
) -> tuple[int, dict[int, np.ndarray]]:
    candidates: list[tuple[float, int, float, int, dict[int, np.ndarray]]] = []
    moving_candidates: list[tuple[float, int, float, int, dict[int, np.ndarray]]] = []
    for gt_id, frame_map in gt_map.items():
        for segment in consecutive_segments(frame_map):
            support_ratios = []
            for method in METHODS:
                track = best_predicted_track(segment, method_predictions[method], EVAL_DISTANCE_MM)
                matched = 0
                for point in track:
                    gt_point = segment.get(point.frame)
                    if gt_point is not None and np.linalg.norm(point.xyz - gt_point) <= EVAL_DISTANCE_MM:
                        matched += 1
                support_ratios.append(matched / max(1, len(segment)))
            points = np.asarray(list(segment.values()), dtype=np.float64)
            extent = float(np.linalg.norm(np.ptp(points, axis=0)))
            candidate = (min(support_ratios), len(segment), extent, int(gt_id), segment)
            candidates.append(candidate)
            if extent >= 100.0:
                moving_candidates.append(candidate)
    if not candidates:
        gt_id = max(gt_map, key=lambda item: len(gt_map[item]))
        return int(gt_id), gt_map[gt_id]
    selection_pool = moving_candidates if moving_candidates else candidates
    selected = max(selection_pool, key=lambda item: (item[0], item[1], item[2]))
    return selected[3], selected[4]


def best_predicted_track(
    gt_frames: dict[int, np.ndarray],
    predictions: list[TrackPoint3D],
    gate_mm: float,
) -> list[TrackPoint3D]:
    grouped: dict[int, list[TrackPoint3D]] = defaultdict(list)
    for point in predictions:
        grouped[point.track_id].append(point)
    scored: list[tuple[int, float, int]] = []
    for track_id, points in grouped.items():
        distances = []
        for point in points:
            gt_point = gt_frames.get(point.frame)
            if gt_point is not None:
                distances.append(float(np.linalg.norm(point.xyz - gt_point)))
        matched = sum(distance <= gate_mm for distance in distances)
        median = float(np.median(distances)) if distances else float("inf")
        scored.append((matched, -median, track_id))
    if not scored:
        return []
    selected_id = max(scored)[2]
    return sorted(grouped[selected_id], key=lambda item: item.frame)


def equalize_3d_axes(ax: plt.Axes, limits: tuple[np.ndarray, np.ndarray]) -> None:
    lower, upper = limits
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_zlim(float(lower[2]), float(upper[2]))
    ax.view_init(elev=22, azim=-58)
    relative_span = (upper - lower) / max(float(np.max(upper - lower)), 1e-9)
    ax.set_box_aspect(np.maximum(relative_span, 0.42).tolist())
    ax.set_xlabel("X (mm)", labelpad=5)
    ax.set_ylabel("Y (mm)", labelpad=5)
    ax.set_zlabel("Z (mm)", labelpad=5)
    ax.tick_params(pad=1)
    ax.grid(True, alpha=0.18, linewidth=0.5)


def make_clean_figure(
    output_root: Path,
    sequence: str,
    gt_map: dict[int, dict[int, np.ndarray]],
    predictions: dict[str, dict[str, list[TrackPoint3D]]],
) -> dict[str, object]:
    gt_id, gt_frames = select_visualization_segment(
        gt_map,
        {method: predictions[method][sequence] for method in METHODS},
    )
    gt_order = sorted(gt_frames)
    gt_points = np.asarray([gt_frames[frame] for frame in gt_order], dtype=np.float64)
    selected: dict[str, list[TrackPoint3D]] = {}
    all_points = [gt_points]
    for method in METHODS:
        track = best_predicted_track(gt_frames, predictions[method][sequence], EVAL_DISTANCE_MM)
        selected[method] = track
        if track:
            all_points.append(np.asarray([point.xyz for point in track], dtype=np.float64))

    stacked = np.vstack(all_points)
    lower = np.percentile(stacked, 1, axis=0)
    upper = np.percentile(stacked, 99, axis=0)
    span = np.maximum(upper - lower, np.asarray([20.0, 20.0, 20.0]))
    lower = lower - 0.08 * span
    upper = upper + 0.08 * span

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    figure = plt.figure(figsize=(12.0, 9.5))
    for index, method in enumerate(METHODS, start=1):
        ax = figure.add_subplot(2, 2, index, projection="3d")
        ax.plot(
            gt_points[:, 0],
            gt_points[:, 1],
            gt_points[:, 2],
            color="#6f6f6f",
            linewidth=1.4,
            linestyle=(0, (4, 2)),
            alpha=0.9,
        )
        track = selected[method]
        if track:
            points = np.asarray([point.xyz for point in track], dtype=np.float64)
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color="#c43c2f",
                linewidth=2.0,
                alpha=0.95,
            )
        equalize_3d_axes(ax, (lower, upper))
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.97, wspace=0.12, hspace=0.22)
    figure_root = output_root / "figures"
    ensure_dir(figure_root)
    png_path = figure_root / "Figure_point_3D_MOT_comparison_clean.png"
    pdf_path = figure_root / "Figure_point_3D_MOT_comparison_clean.pdf"
    figure.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.20, facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.20, facecolor="white")
    plt.close(figure)

    panel_lines = [
        "# Clean figure panel mapping",
        "",
        "- Top left: Nearest-neighbour 3D tracking",
        "- Top right: Euclidean 3D Kalman-Hungarian",
        "- Bottom left: Mahalanobis 3D Kalman-Hungarian",
        "- Bottom right: Proposed reliability-guided framework",
        "- Gray dashed curve: pseudo-3D reference",
        "- Red solid curve: reconstructed trajectory",
        f"- Sequence: {sequence}",
        f"- Reference identity: {gt_id}",
        f"- Frame range: {min(gt_frames)}-{max(gt_frames)}",
        "- All panels use the same view and axis limits.",
    ]
    (figure_root / "figure_panel_mapping.md").write_text("\n".join(panel_lines) + "\n", encoding="utf-8")
    return {
        "sequence": sequence,
        "gt_id": gt_id,
        "frame_start": min(gt_frames),
        "frame_end": max(gt_frames),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "axis_lower_mm": lower.tolist(),
        "axis_upper_mm": upper.tolist(),
    }


def write_paper_table(path: Path, rows: list[dict[str, float]]) -> None:
    lines = [
        "| Method | HOTA-3D | AssA-3D | IDF1-3D | MOTA-3D | IDSW | RMSE (mm) | Jitter (mm) | Coverage (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['HOTA_3D']:.2f} | {row['AssA_3D']:.2f} | "
            f"{row['IDF1_3D']:.2f} | {row['MOTA_3D']:.2f} | {int(row['IDSW'])} | "
            f"{row['RMSE_mm']:.2f} | {row['Jitter_mm']:.2f} | {row['Coverage_pct']:.2f} |"
        )
    lines.extend(
        [
            "",
            "RMSE is computed over accepted frame-wise Hungarian point matches within 25 mm. Coverage is the number of accepted 25 mm matches divided by all alignable pseudo-3D reference points. The word 'matched' may be used in explanatory text, but it does not denote a separate metric.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predictions(path: Path, predictions: dict[str, dict[str, list[TrackPoint3D]]]) -> None:
    rows = []
    for method, by_sequence in predictions.items():
        for seq_name, points in by_sequence.items():
            for point in points:
                rows.append(
                    {
                        "method": method,
                        "sequence": seq_name,
                        "frame": point.frame,
                        "track_id": point.track_id,
                        "x_mm": float(point.xyz[0]),
                        "y_mm": float(point.xyz[1]),
                        "z_mm": float(point.xyz[2]),
                        "score": point.score,
                    }
                )
    write_csv(
        path,
        rows,
        ["method", "sequence", "frame", "track_id", "x_mm", "y_mm", "z_mm", "score"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--visualization-sequence", default="CP-I2-03")
    args = parser.parse_args()

    pipeline_root = Path(args.pipeline_root).resolve()
    data_root = Path(args.data_root).resolve()
    save_root = Path(args.save_root).resolve()
    ensure_dir(save_root)
    ensure_dir(save_root / "tables")
    ensure_dir(save_root / "reports")
    ensure_dir(save_root / "texts")

    step4_root = pipeline_root / "step4_association"
    step1_root = pipeline_root / "step1_rectification"
    metadata = json.loads((step4_root / "run_metadata.json").read_text(encoding="utf-8"))
    train_sequences = list(metadata["train_sequences"])
    test_sequences = list(metadata["test_sequences"])
    s4.configure_input_roots(
        Path(metadata["step3_train_root"]),
        Path(metadata["step3_test_root"]),
        data_root,
        step1_root,
    )
    s5.configure_step1_root(step1_root)

    priors = s4.compute_group_priors(train_sequences)
    print("[1/7] Building train and test sequence contexts")
    train_contexts = [s4.build_sequence_context(sequence, priors) for sequence in train_sequences]
    test_contexts = [s4.build_sequence_context(sequence, priors) for sequence in test_sequences]

    print("[2/7] Selecting frame-wise stereo matcher on training split")
    stereo_tuning_path = save_root / "reports" / "framewise_stereo_matcher_tuning.csv"
    cached_tuning = load_complete_stereo_tuning(stereo_tuning_path)
    if cached_tuning is None:
        stereo_params, stereo_tuning_rows = tune_stereo_matcher(train_contexts)
    else:
        stereo_params, stereo_tuning_rows = cached_tuning
        print("  Reused validated complete 36-candidate deterministic tuning cache")
    write_csv(
        stereo_tuning_path,
        stereo_tuning_rows,
        list(stereo_tuning_rows[0]),
    )

    print("[3/7] Building frame-wise 3D observations")
    train_observations = build_observations(train_contexts, stereo_params)
    test_observations = build_observations(test_contexts, stereo_params)

    print("[4/7] Building unchanged pseudo-3D references")
    train_gt = {sequence: s5.build_gt_reference(sequence)[0] for sequence in train_sequences}
    test_gt = {sequence: s5.build_gt_reference(sequence)[0] for sequence in test_sequences}

    print("[5/7] Selecting 3D tracker parameters on training split")
    selected_params: dict[str, dict[str, float]] = {}
    tracker_tuning_rows: list[dict[str, object]] = []
    for method in CLASSICAL_METHODS:
        params, rows = tune_tracker(method, train_observations, train_contexts, train_gt)
        selected_params[method] = params
        tracker_tuning_rows.extend(rows)
        print(f"  {method}: {params}")
    write_csv(
        save_root / "reports" / "tracker_training_selection.csv",
        tracker_tuning_rows,
        sorted({key for row in tracker_tuning_rows for key in row}),
    )

    print("[6/7] Running fixed methods on held-out test split")
    test_predictions: dict[str, dict[str, list[TrackPoint3D]]] = {}
    for method in CLASSICAL_METHODS:
        test_predictions[method] = {
            ctx.seq_name: run_tracker(
                ctx.seq_name,
                test_observations[ctx.seq_name],
                method,
                selected_params[method],
                ctx.seq_length,
            )
            for ctx in test_contexts
        }
    proposed_source = (
        pipeline_root
        / "step5_reconstruction"
        / "paper_assets"
        / "reports"
        / "reconstruction_points_compag.csv"
    )
    test_predictions[METHODS[-1]] = load_proposed_points(proposed_source, test_sequences)

    seq_lengths = {ctx.seq_name: ctx.seq_length for ctx in test_contexts}
    summaries: list[dict[str, float]] = []
    per_sequence_rows: list[dict[str, float]] = []
    for method in METHODS:
        summary, per_sequence = evaluate_method(
            method,
            test_predictions[method],
            test_gt,
            seq_lengths,
            EVAL_DISTANCE_MM,
        )
        summaries.append(summary)
        per_sequence_rows.extend(per_sequence)

    summary_fields = [
        "method",
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
    write_csv(
        save_root / "tables" / "Table_point_3D_MOT_comparison.csv",
        summaries,
        summary_fields,
    )
    write_paper_table(save_root / "tables" / "Table_point_3D_MOT_comparison.md", summaries)
    write_csv(
        save_root / "reports" / "point_3D_MOT_per_sequence.csv",
        per_sequence_rows,
        ["method"] + [key for key in per_sequence_rows[0] if key != "method"],
    )
    write_predictions(save_root / "reports" / "point_3D_MOT_predictions.csv", test_predictions)

    print("[7/7] Rendering clean paper figure and validating")
    figure_info = make_clean_figure(
        save_root,
        args.visualization_sequence,
        test_gt[args.visualization_sequence],
        test_predictions,
    )
    finite_metrics = all(
        np.isfinite(float(row[key]))
        for row in summaries
        for key in [
            "HOTA_3D",
            "AssA_3D",
            "IDF1_3D",
            "MOTA_3D",
            "RMSE_mm",
            "Jitter_mm",
            "Coverage_pct",
        ]
    )
    all_positive_depth = all(
        float(point.xyz[2]) > 0.0
        for by_sequence in test_predictions.values()
        for points in by_sequence.values()
        for point in points
    )
    validation = {
        "status": "PASS" if finite_metrics and all_positive_depth else "FAIL",
        "finite_summary_metrics": finite_metrics,
        "positive_depth_all_predictions": all_positive_depth,
        "methods_complete": [row["method"] for row in summaries] == METHODS,
        "test_sequences_complete": sorted(test_gt) == sorted(test_sequences),
        "test_split_used_for_parameter_selection": False,
    }
    (save_root / "reports" / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = {
        "experiment": "Point-target 3D MOT framework comparison",
        "protocol_warning": (
            "This is a point-distance adaptation for stereo insect trajectories, "
            "not the unchanged 3D-box-IoU AB3DMOT protocol."
        ),
        "pipeline_root": str(pipeline_root),
        "data_root": str(data_root),
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "input_2d_tracker": "ByteTrack observations; track IDs ignored by classical 3D baselines",
        "pseudo_3d_reference": "Unchanged synchronized rectified GT-centre triangulation",
        "evaluation_distance_mm": EVAL_DISTANCE_MM,
        "distance_rationale": {
            "train_gt_consecutive_motion_p95_mm": 22.31789215,
            "train_gt_nearest_identity_spacing_p01_mm": 27.9209449,
        },
        "point_similarity": "max(0, 1 - EuclideanDistance / (2 * 25 mm))",
        "trackeval_threshold_for_ID_and_CLEAR": 0.5,
        "stereo_matcher_selected_on_train": stereo_params,
        "tracker_parameters_selected_on_train": selected_params,
        "tracker_selection_objective": "Maximum train-split CLEAR MOTA, then minimum IDSW",
        "proposed_input": str(proposed_source),
        "metrics": {
            "HOTA_3D": "TrackEval HOTA using point-distance similarity",
            "IDF1_3D": "TrackEval Identity metric at 25 mm",
            "MOTA_3D_IDSW": "TrackEval CLEAR metrics at 25 mm",
            "RMSE_MAE_MedAE": "Euclidean errors of frame-wise Hungarian matches within 25 mm",
            "Jitter": "Mean norm of second differences over consecutive predicted track points",
            "Coverage": "Matched pseudo-3D GT points divided by all pseudo-3D GT points",
        },
        "figure": figure_info,
        "validation": validation,
    }
    (save_root / "comparison_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = f"""# Point-target 3D MOT comparison

This supplement compares three classical point-target 3D tracking baselines
with the final CP3D reliability-guided stereo framework.

## Fair comparison protocol

- Same YOLOv12-L detections and ByteTrack observations.
- Classical baselines ignore ByteTrack identity labels.
- Same train-only frame-wise stereo matcher for all classical baselines.
- All baseline parameters are selected on the eight training sequences.
- The four test sequences are evaluated once.
- Same unchanged pseudo-3D references and a fixed 25 mm point-distance gate.
- HOTA, Identity, and CLEAR are calculated by TrackEval.

This is a point-target adaptation. It must not be described as the unchanged
official AB3DMOT 3D-box-IoU evaluation.

## Outputs

- `tables/Table_point_3D_MOT_comparison.csv`
- `tables/Table_point_3D_MOT_comparison.md`
- `reports/point_3D_MOT_per_sequence.csv`
- `reports/framewise_stereo_matcher_tuning.csv`
- `reports/tracker_training_selection.csv`
- `reports/point_3D_MOT_predictions.csv`
- `reports/validation_report.json`
- `figures/Figure_point_3D_MOT_comparison_clean.png`
- `figures/Figure_point_3D_MOT_comparison_clean.pdf`
- `figures/figure_panel_mapping.md`
- `comparison_config.json`
"""
    (save_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"summaries": summaries, "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
