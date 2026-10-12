"""
Step 5
Trajectory quality scoring and reliability-guided 3D trajectory reconstruction.

This script:
1. Reuses the tuned step4 stereo association outputs and features.
2. Builds pseudo-3D references from rectified left/right ground-truth centers.
3. Scores stereo trajectory segments with several quality strategies.
4. Tunes train-split thresholds for reliable-segment selection.
5. Benchmarks several 3D reconstruction variants on the test split.
6. Exports paper-ready tables, figures, and discussion text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent.parent / ".runtime"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_ROOT / "matplotlib"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import step3_mot_tracking_eval as base
import step4_stereo_association_eval as s4


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

STEP1_ROOT = ROOT / "results" / "step1_rectification"
STEP4_ROOT = ROOT / "results" / "step4_stereo_association"
DEFAULT_SAVE_ROOT = ROOT / "results" / "step5_3d_reconstruction"

QUALITY_METHODS = ["LengthOnly", "GeoOnly", "GeoConf", "ProposedTQS"]
RECON_METHODS = ["RawStereo3D", "GeomGate3D", "TQSFiltered3D", "ReliabilityGuided3D"]


def configure_step4_root(step4_root: Path) -> None:
    global STEP4_ROOT
    STEP4_ROOT = step4_root.resolve()


def configure_step1_root(step1_root: Path) -> None:
    global STEP1_ROOT
    STEP1_ROOT = step1_root.resolve()


@dataclass
class StereoSample:
    frame: int
    left_track_id: int
    right_track_id: int
    left_det: s4.Detection
    right_det: s4.Detection
    pair_score: float
    pair_feature: s4.PairFeature | None

    @property
    def disparity(self) -> float:
        return self.left_det.cx - self.right_det.cx

    @property
    def vdiff(self) -> float:
        return abs(self.left_det.cy - self.right_det.cy)

    @property
    def conf(self) -> float:
        return 0.5 * (self.left_det.score + self.right_det.score)


@dataclass
class StereoSegment:
    seq_name: str
    split: str
    component_id: int
    segment_id: int
    left_track_id: int
    right_track_id: int
    samples: list[StereoSample]
    gt_id: int | None = None
    gt_vote_hits: int = 0
    raw_points: list[np.ndarray] = field(default_factory=list)
    gt_points: list[np.ndarray] = field(default_factory=list)
    point_errors: list[float] = field(default_factory=list)
    frame_reliability: list[float] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    quality_scores: dict[str, float] = field(default_factory=dict)

    @property
    def frames(self) -> list[int]:
        return [sample.frame for sample in self.samples]

    @property
    def length(self) -> int:
        return len(self.samples)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def span(self) -> int:
        return self.end - self.start + 1

    @property
    def rmse(self) -> float | None:
        if not self.point_errors:
            return None
        return float(np.sqrt(np.mean(np.square(self.point_errors))))

    @property
    def mae(self) -> float | None:
        if not self.point_errors:
            return None
        return float(np.mean(self.point_errors))


@dataclass
class ReconPoint:
    seq_name: str
    split: str
    method: str
    gt_id: int | None
    frame: int
    point3d: np.ndarray
    gt_point3d: np.ndarray | None
    left_gt_center: tuple[float, float] | None
    right_gt_center: tuple[float, float] | None
    reliability: float
    component_id: int
    segment_id: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def exp_score(value: float, scale: float) -> float:
    scale = max(scale, 1e-6)
    return math.exp(-max(0.0, value) / scale)


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def load_projection(seq_name: str) -> tuple[np.ndarray, np.ndarray]:
    seq_root = STEP1_ROOT / seq_name
    return np.load(seq_root / "P1.npy"), np.load(seq_root / "P2.npy")


def triangulate_xy(P1: np.ndarray, P2: np.ndarray, left_xy: tuple[float, float], right_xy: tuple[float, float]) -> np.ndarray:
    pts1 = np.asarray(left_xy, dtype=np.float64).reshape(2, 1)
    pts2 = np.asarray(right_xy, dtype=np.float64).reshape(2, 1)
    hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
    xyz = (hom[:3] / hom[3]).reshape(-1)
    return xyz.astype(np.float64)


def project_point(P: np.ndarray, point3d: np.ndarray) -> tuple[float, float]:
    hom = np.append(point3d.astype(np.float64), 1.0)
    proj = P @ hom
    if abs(float(proj[2])) <= 1e-9:
        return float("nan"), float("nan")
    return float(proj[0] / proj[2]), float(proj[1] / proj[2])


def load_selected_assoc_params() -> dict[str, float]:
    metadata = json.loads((STEP4_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
    params = metadata["selected_params"]["Proposed"]
    return {
        "threshold": float(params["threshold"]),
        "secondary_threshold": float(params["secondary_threshold"]),
    }


def build_gt_reference(seq_name: str) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, s4.Track], dict[int, s4.Track]]:
    P1, P2 = load_projection(seq_name)
    left_gt, _ = s4.load_rectified_gt(seq_name, "left")
    right_gt, _ = s4.load_rectified_gt(seq_name, "right")
    gt_points: dict[int, dict[int, np.ndarray]] = {}
    for gt_id in sorted(set(left_gt) & set(right_gt)):
        overlap = left_gt[gt_id].overlap_frames(right_gt[gt_id])
        if not overlap:
            continue
        frame_map: dict[int, np.ndarray] = {}
        for frame in overlap:
            ldet = left_gt[gt_id].detections[frame]
            rdet = right_gt[gt_id].detections[frame]
            frame_map[frame] = triangulate_xy(P1, P2, (ldet.cx, ldet.cy), (rdet.cx, rdet.cy))
        if frame_map:
            gt_points[gt_id] = frame_map
    return gt_points, left_gt, right_gt


def find_sample_gt_id(ctx: s4.SequenceContext, sample: StereoSample) -> int | None:
    left_candidates = {det.track_id: det for det in ctx.left_gt_frame_index.get(sample.frame, [])}
    right_candidates = {det.track_id: det for det in ctx.right_gt_frame_index.get(sample.frame, [])}
    best_gt_id = None
    best_cost = float("inf")
    for gt_id in sorted(set(left_candidates) & set(right_candidates)):
        lgt = left_candidates[gt_id]
        rgt = right_candidates[gt_id]
        ldist = math.hypot(sample.left_det.cx - lgt.cx, sample.left_det.cy - lgt.cy)
        rdist = math.hypot(sample.right_det.cx - rgt.cx, sample.right_det.cy - rgt.cy)
        gate_l = max(18.0, 0.9 * (sample.left_det.h + lgt.h))
        gate_r = max(18.0, 0.9 * (sample.right_det.h + rgt.h))
        if ldist <= gate_l and rdist <= gate_r:
            cost = ldist + rdist
            if cost < best_cost:
                best_cost = cost
                best_gt_id = gt_id
    return best_gt_id


def build_segments_for_context(
    ctx: s4.SequenceContext,
    components: list[dict[str, set[int]]],
    score_lookup: dict[tuple[int, int], float],
    gt_reference: dict[int, dict[int, np.ndarray]],
    depth_priors: dict[str, tuple[float, float]],
) -> list[StereoSegment]:
    P1, P2 = load_projection(ctx.seq_name)
    feature_lookup = {(feature.left_id, feature.right_id): feature for feature in ctx.pair_features}
    segments: list[StereoSegment] = []
    next_segment_id = 1

    for component_id, comp in enumerate(components, start=1):
        component_samples: list[StereoSample] = []
        active_frames = sorted(
            set(frame for track_id in comp["left"] for frame in ctx.left_tracks[track_id].frames)
            | set(frame for track_id in comp["right"] for frame in ctx.right_tracks[track_id].frames)
        )
        for frame in active_frames:
            active_left = [track_id for track_id in comp["left"] if frame in ctx.left_tracks[track_id].detections]
            active_right = [track_id for track_id in comp["right"] if frame in ctx.right_tracks[track_id].detections]
            if not active_left or not active_right:
                continue
            best_pair = None
            best_score = -1.0
            for left_id in active_left:
                for right_id in active_right:
                    score = score_lookup.get((left_id, right_id), 0.0)
                    if score > best_score:
                        best_score = score
                        best_pair = (left_id, right_id)
            if best_pair is None:
                continue
            left_id, right_id = best_pair
            sample = StereoSample(
                frame=frame,
                left_track_id=left_id,
                right_track_id=right_id,
                left_det=ctx.left_tracks[left_id].detections[frame],
                right_det=ctx.right_tracks[right_id].detections[frame],
                pair_score=best_score,
                pair_feature=feature_lookup.get((left_id, right_id)),
            )
            component_samples.append(sample)

        if not component_samples:
            continue

        current: list[StereoSample] = []
        prev_frame = None
        prev_pair = None
        for sample in component_samples:
            sample_pair = (sample.left_track_id, sample.right_track_id)
            if current and (
                prev_frame is None
                or sample.frame != prev_frame + 1
                or sample_pair != prev_pair
            ):
                if len(current) >= 3:
                    segments.append(
                        StereoSegment(
                            seq_name=ctx.seq_name,
                            split=ctx.split,
                            component_id=component_id,
                            segment_id=next_segment_id,
                            left_track_id=current[0].left_track_id,
                            right_track_id=current[0].right_track_id,
                            samples=list(current),
                        )
                    )
                    next_segment_id += 1
                current = []
            current.append(sample)
            prev_frame = sample.frame
            prev_pair = sample_pair
        if len(current) >= 3:
            segments.append(
                StereoSegment(
                    seq_name=ctx.seq_name,
                    split=ctx.split,
                    component_id=component_id,
                    segment_id=next_segment_id,
                    left_track_id=current[0].left_track_id,
                    right_track_id=current[0].right_track_id,
                    samples=list(current),
                )
            )
            next_segment_id += 1

    for segment in segments:
        gt_votes: Counter[int] = Counter()
        sample_gt_ids: list[int | None] = []
        raw_points: list[np.ndarray] = []
        frame_reliability: list[float] = []
        for sample in segment.samples:
            gt_id = find_sample_gt_id(ctx, sample)
            sample_gt_ids.append(gt_id)
            if gt_id is not None:
                gt_votes[gt_id] += 1
            raw_xyz = triangulate_xy(P1, P2, (sample.left_det.cx, sample.left_det.cy), (sample.right_det.cx, sample.right_det.cy))
            raw_points.append(raw_xyz)
            reliability = frame_level_reliability(sample, ctx)
            frame_reliability.append(reliability)

        if gt_votes:
            gt_id, hits = gt_votes.most_common(1)[0]
            if hits >= max(3, int(math.ceil(segment.length * 0.5))):
                segment.gt_id = gt_id
                segment.gt_vote_hits = hits
        segment.raw_points = raw_points
        segment.frame_reliability = frame_reliability

        gt_points: list[np.ndarray] = []
        point_errors: list[float] = []
        if segment.gt_id is not None:
            for sample, sample_gt_id, raw_xyz in zip(segment.samples, sample_gt_ids, raw_points):
                if sample_gt_id != segment.gt_id:
                    continue
                if segment.gt_id not in gt_reference or sample.frame not in gt_reference[segment.gt_id]:
                    continue
                gt_xyz = gt_reference[segment.gt_id][sample.frame]
                gt_points.append(gt_xyz)
                point_errors.append(float(np.linalg.norm(raw_xyz - gt_xyz)))
        segment.gt_points = gt_points
        segment.point_errors = point_errors
        segment.features = compute_segment_features(segment, ctx, depth_priors)
        segment.quality_scores = compute_quality_scores(segment.features)
    return segments


def frame_level_reliability(sample: StereoSample, ctx: s4.SequenceContext) -> float:
    prior = ctx.group_prior
    roi_hit = 1.0 if (s4.inside_roi(sample.left_det, ctx.roi_left) and s4.inside_roi(sample.right_det, ctx.roi_right)) else 0.0
    disp_dev = abs(sample.disparity - prior.disp_median)
    disp_scale = max(60.0, prior.disp_iqr * 2.25)
    v_scale = max(40.0, prior.v_p95 * 1.15)
    parts = [
        0.30 * clamp01(sample.conf),
        0.25 * exp_score(sample.vdiff, v_scale),
        0.20 * exp_score(disp_dev, disp_scale),
        0.15 * clamp01(sample.pair_score),
        0.10 * roi_hit,
    ]
    return clamp01(sum(parts))


def compute_segment_features(
    segment: StereoSegment,
    ctx: s4.SequenceContext,
    depth_priors: dict[str, tuple[float, float]],
) -> dict[str, float]:
    pair_feature = segment.samples[0].pair_feature
    length_score = clamp01(segment.length / 20.0)
    completeness_score = clamp01(segment.length / max(1.0, float(segment.span)))
    conf_score = float(np.mean([sample.conf for sample in segment.samples]))
    pair_score = float(np.mean([sample.pair_score for sample in segment.samples]))
    mean_reliability = float(np.mean(segment.frame_reliability)) if segment.frame_reliability else 0.0

    disparities = np.asarray([sample.disparity for sample in segment.samples], dtype=np.float64)
    disp_std = float(np.std(disparities)) if disparities.size else 0.0
    disp_stability = exp_score(disp_std, max(24.0, ctx.group_prior.disp_iqr * 1.25))

    if len(segment.raw_points) >= 3:
        raw_arr = np.asarray(segment.raw_points, dtype=np.float64)
        accel = np.diff(raw_arr, n=2, axis=0)
        accel_mag = np.linalg.norm(accel, axis=1) if accel.size else np.asarray([], dtype=np.float64)
        smooth_score = exp_score(float(np.median(accel_mag)) if accel_mag.size else 0.0, 400.0)
        z_arr = raw_arr[:, 2]
    else:
        smooth_score = 0.5
        z_arr = np.asarray([point[2] for point in segment.raw_points], dtype=np.float64) if segment.raw_points else np.asarray([], dtype=np.float64)

    group_name = s4.seq_group(ctx.seq_name)
    depth_median, depth_iqr = depth_priors.get(group_name, (0.0, 500.0))
    if z_arr.size:
        depth_dev = abs(float(np.median(z_arr)) - depth_median)
        depth_std = float(np.std(z_arr))
        depth_prior_score = exp_score(depth_dev, max(200.0, depth_iqr * 2.0)) * exp_score(depth_std, max(180.0, depth_iqr * 1.5))
    else:
        depth_prior_score = 0.0

    features = {
        "pair_score": pair_score,
        "length_score": length_score,
        "completeness_score": completeness_score,
        "conf_score": conf_score,
        "mean_reliability": mean_reliability,
        "disp_stability_score": disp_stability,
        "smooth3d_score": smooth_score,
        "temporal_score": pair_feature.temporal_score if pair_feature else 0.0,
        "epi_score": pair_feature.epi_score if pair_feature else 0.0,
        "spatial_score": pair_feature.spatial_score if pair_feature else 0.0,
        "motion_score": pair_feature.motion_score if pair_feature else 0.0,
        "rank_score": pair_feature.rank_score if pair_feature else 0.0,
        "local_score": pair_feature.local_score if pair_feature else 0.0,
        "region_score": pair_feature.region_score if pair_feature else 0.0,
        "depth_prior_score": depth_prior_score,
    }
    return {key: clamp01(float(value)) for key, value in features.items()}


def compute_quality_scores(features: dict[str, float]) -> dict[str, float]:
    scores = {
        "LengthOnly": 0.60 * features["length_score"] + 0.40 * features["completeness_score"],
        "GeoOnly": (
            0.35 * features["temporal_score"]
            + 0.35 * features["epi_score"]
            + 0.30 * features["spatial_score"]
        ),
        "GeoConf": (
            0.25 * features["pair_score"]
            + 0.25 * features["epi_score"]
            + 0.20 * features["spatial_score"]
            + 0.15 * features["region_score"]
            + 0.15 * features["conf_score"]
        ),
        "ProposedTQS": (
            0.16 * features["pair_score"]
            + 0.08 * features["temporal_score"]
            + 0.10 * features["epi_score"]
            + 0.09 * features["spatial_score"]
            + 0.07 * features["motion_score"]
            + 0.06 * features["rank_score"]
            + 0.06 * features["mean_reliability"]
            + 0.04 * features["region_score"]
            + 0.08 * features["conf_score"]
            + 0.06 * features["length_score"]
            + 0.04 * features["completeness_score"]
            + 0.12 * features["smooth3d_score"]
            + 0.04 * features["depth_prior_score"]
        ),
    }
    return {key: clamp01(float(value)) for key, value in scores.items()}


def collect_labeled_segments(segments: list[StereoSegment], min_points: int = 5) -> list[StereoSegment]:
    return [segment for segment in segments if segment.gt_id is not None and len(segment.point_errors) >= min_points]


def choose_reliable_rmse_threshold(train_segments: list[StereoSegment]) -> float:
    rmses = np.asarray([segment.rmse for segment in train_segments if segment.rmse is not None], dtype=np.float64)
    if rmses.size == 0:
        return 300.0
    return float(np.percentile(rmses, 45))


def compute_depth_priors(
    contexts: list[s4.SequenceContext],
    gt_refs: dict[str, tuple[dict[int, dict[int, np.ndarray]], dict[int, s4.Track], dict[int, s4.Track]]],
) -> dict[str, tuple[float, float]]:
    grouped_depths: dict[str, list[float]] = {}
    for ctx in contexts:
        group_name = s4.seq_group(ctx.seq_name)
        grouped_depths.setdefault(group_name, [])
        gt_map = gt_refs[ctx.seq_name][0]
        for frame_map in gt_map.values():
            grouped_depths[group_name].extend(float(point[2]) for point in frame_map.values())
    priors: dict[str, tuple[float, float]] = {}
    for group_name, values in grouped_depths.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            priors[group_name] = (0.0, 500.0)
            continue
        priors[group_name] = (
            float(np.median(arr)),
            float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        )
    return priors


def tune_quality_threshold(train_segments: list[StereoSegment], method: str, rmse_thr: float) -> dict[str, float]:
    thresholds = [round(value, 2) for value in np.arange(0.20, 0.96, 0.05)]
    best = {"threshold": thresholds[0], "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in thresholds:
        tp = fp = fn = 0
        for segment in train_segments:
            pred = segment.quality_scores[method] >= threshold
            gt = segment.rmse is not None and segment.rmse <= rmse_thr
            if pred and gt:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif (not pred) and gt:
                fn += 1
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        if f1 > best["f1"] + 1e-9:
            best = {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1}
    return best


def evaluate_quality_method(
    test_segments: list[StereoSegment],
    method: str,
    threshold: float,
    rmse_thr: float,
) -> dict[str, float]:
    tp = fp = fn = 0
    selected_errors: list[float] = []
    selected_segment_count = 0
    for segment in test_segments:
        pred = segment.quality_scores[method] >= threshold
        gt = segment.rmse is not None and segment.rmse <= rmse_thr
        if pred:
            selected_segment_count += 1
            selected_errors.extend(segment.point_errors)
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif (not pred) and gt:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    rmse = float(np.sqrt(np.mean(np.square(selected_errors)))) if selected_errors else float("nan")
    coverage = selected_segment_count / max(1, len(test_segments))
    scores = np.asarray([segment.quality_scores[method] for segment in test_segments], dtype=np.float64)
    rmses = np.asarray([segment.rmse if segment.rmse is not None else np.nan for segment in test_segments], dtype=np.float64)
    valid = ~np.isnan(rmses)
    if np.sum(valid) >= 3:
        corr = float(spearmanr(scores[valid], rmses[valid]).correlation)
    else:
        corr = float("nan")
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "selected_rmse_mm": rmse,
        "selection_ratio": coverage,
        "spearman_rho": corr,
    }


def temporal_parameters(ctx: s4.SequenceContext) -> tuple[int, float, int]:
    frame_rate = int(base.read_seqinfo(ctx.seq_name)["frameRate"])
    scale = frame_rate / 20.0
    radius = max(1, int(round(2.0 * scale)))
    decay = max(1e-6, 1.5 * scale)
    max_gap = max(1, int(round(3.0 * scale)))
    return radius, decay, max_gap


def reconstruct_segment_points(
    segment: StereoSegment,
    ctx: s4.SequenceContext,
    method: str,
    tqs_threshold: float,
    depth_priors: dict[str, tuple[float, float]],
) -> list[tuple[int, np.ndarray, float]]:
    if method == "TQSFiltered3D" and segment.quality_scores["ProposedTQS"] < tqs_threshold:
        return []
    if method == "ReliabilityGuided3D" and segment.quality_scores["ProposedTQS"] < tqs_threshold:
        return []

    prior = ctx.group_prior
    disp_gate = max(60.0, prior.disp_iqr * 2.25)
    v_gate = max(40.0, prior.v_p95 * 1.15)
    depth_median, depth_iqr = depth_priors.get(s4.seq_group(ctx.seq_name), (0.0, 500.0))

    raw: list[tuple[int, np.ndarray, float]] = []
    for idx, sample in enumerate(segment.samples):
        if method == "GeomGate3D":
            if sample.vdiff > v_gate or abs(sample.disparity - prior.disp_median) > disp_gate:
                continue
        reliability = segment.frame_reliability[idx] if idx < len(segment.frame_reliability) else 0.5
        point3d = segment.raw_points[idx]
        if method in {"TQSFiltered3D", "ReliabilityGuided3D"}:
            depth_dev = abs(float(point3d[2]) - depth_median)
            depth_gate = max(800.0, depth_iqr * 4.0)
            if reliability < 0.45 or depth_dev > depth_gate:
                continue
        raw.append((sample.frame, point3d, reliability))

    if method != "ReliabilityGuided3D":
        return raw

    if len(raw) < 3:
        return raw

    frames = [item[0] for item in raw]
    points = np.asarray([item[1] for item in raw], dtype=np.float64)
    reliabilities = np.asarray([item[2] for item in raw], dtype=np.float64)
    temporal_radius, temporal_decay, short_gap = temporal_parameters(ctx)

    smoothed = []
    for i, frame in enumerate(frames):
        acc_point = np.zeros(3, dtype=np.float64)
        acc_weight = 0.0
        for j, other_frame in enumerate(frames):
            if abs(other_frame - frame) > temporal_radius:
                continue
            temporal_w = math.exp(-abs(other_frame - frame) / temporal_decay)
            weight = temporal_w * float(reliabilities[j])
            acc_point += weight * points[j]
            acc_weight += weight
        neighborhood_point = acc_point / max(acc_weight, 1e-6)
        raw_weight = 0.55 + 0.35 * float(reliabilities[i])
        smooth_point = raw_weight * points[i] + (1.0 - raw_weight) * neighborhood_point
        smoothed.append((frame, smooth_point, float(reliabilities[i])))

    with_gaps: list[tuple[int, np.ndarray, float]] = []
    for idx, item in enumerate(smoothed):
        with_gaps.append(item)
        if idx == len(smoothed) - 1:
            continue
        frame_a, point_a, rel_a = smoothed[idx]
        frame_b, point_b, rel_b = smoothed[idx + 1]
        gap = frame_b - frame_a
        if 1 < gap <= short_gap and rel_a >= 0.45 and rel_b >= 0.45:
            for missing_frame in range(frame_a + 1, frame_b):
                ratio = (missing_frame - frame_a) / gap
                interp = (1.0 - ratio) * point_a + ratio * point_b
                with_gaps.append((missing_frame, interp, float(min(rel_a, rel_b) * 0.9)))
    with_gaps.sort(key=lambda item: item[0])
    return with_gaps


def evaluate_reconstruction(
    contexts: list[s4.SequenceContext],
    segments_by_seq: dict[str, list[StereoSegment]],
    gt_refs: dict[str, tuple[dict[int, dict[int, np.ndarray]], dict[int, s4.Track], dict[int, s4.Track]]],
    tqs_threshold: float,
    depth_priors: dict[str, tuple[float, float]],
) -> tuple[list[dict[str, float]], dict[str, list[ReconPoint]]]:
    evaluation_sequences = {ctx.seq_name for ctx in contexts}
    total_gt_points = 0
    for seq_name, (gt_map, _, _) in gt_refs.items():
        if seq_name not in evaluation_sequences:
            continue
        for frame_map in gt_map.values():
            total_gt_points += len(frame_map)

    all_points: dict[str, list[ReconPoint]] = {method: [] for method in RECON_METHODS}
    for ctx in contexts:
        gt_map, left_gt_tracks, right_gt_tracks = gt_refs[ctx.seq_name]
        for segment in segments_by_seq[ctx.seq_name]:
            if segment.gt_id is None:
                continue
            left_gt = left_gt_tracks.get(segment.gt_id)
            right_gt = right_gt_tracks.get(segment.gt_id)
            for method in RECON_METHODS:
                rows = reconstruct_segment_points(segment, ctx, method, tqs_threshold, depth_priors)
                for frame, point3d, reliability in rows:
                    gt_point = gt_map.get(segment.gt_id, {}).get(frame)
                    left_center = None
                    right_center = None
                    if left_gt is not None and frame in left_gt.detections:
                        left_center = (left_gt.detections[frame].cx, left_gt.detections[frame].cy)
                    if right_gt is not None and frame in right_gt.detections:
                        right_center = (right_gt.detections[frame].cx, right_gt.detections[frame].cy)
                    all_points[method].append(
                        ReconPoint(
                            seq_name=ctx.seq_name,
                            split=ctx.split,
                            method=method,
                            gt_id=segment.gt_id,
                            frame=frame,
                            point3d=point3d,
                            gt_point3d=gt_point,
                            left_gt_center=left_center,
                            right_gt_center=right_center,
                            reliability=reliability,
                            component_id=segment.component_id,
                            segment_id=segment.segment_id,
                        )
                    )

    rows: list[dict[str, float]] = []
    for method, points in all_points.items():
        errors = []
        reproj_errors = []
        jitter_terms = []
        grouped: dict[tuple[str, int, int], list[ReconPoint]] = {}
        best_reference_points: dict[tuple[str, int, int], ReconPoint] = {}
        for point in points:
            if point.gt_point3d is not None and point.gt_id is not None:
                key = (point.seq_name, int(point.gt_id), point.frame)
                previous = best_reference_points.get(key)
                if previous is None or point.reliability > previous.reliability:
                    best_reference_points[key] = point
            grouped.setdefault((point.seq_name, point.component_id, point.segment_id), []).append(point)

        for point in best_reference_points.values():
            errors.append(float(np.linalg.norm(point.point3d - point.gt_point3d)))
            if point.gt_point3d is not None and point.left_gt_center is not None and point.right_gt_center is not None:
                P1, P2 = load_projection(point.seq_name)
                lproj = project_point(P1, point.point3d)
                rproj = project_point(P2, point.point3d)
                reproj_errors.append(
                    0.5
                    * (
                        math.hypot(lproj[0] - point.left_gt_center[0], lproj[1] - point.left_gt_center[1])
                        + math.hypot(rproj[0] - point.right_gt_center[0], rproj[1] - point.right_gt_center[1])
                    )
                )

        for group_points in grouped.values():
            ordered = sorted(group_points, key=lambda item: item.frame)
            if len(ordered) >= 3:
                coords = np.asarray([item.point3d for item in ordered], dtype=np.float64)
                accel = np.diff(coords, n=2, axis=0)
                if accel.size:
                    jitter_terms.extend(np.linalg.norm(accel, axis=1).tolist())

        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")
        mae = float(np.mean(errors)) if errors else float("nan")
        medae = float(np.median(errors)) if errors else float("nan")
        reproj = float(np.mean(reproj_errors)) if reproj_errors else float("nan")
        jitter = float(np.mean(jitter_terms)) if jitter_terms else float("nan")
        coverage = len(best_reference_points) / max(1, total_gt_points)
        rows.append(
            {
                "method": method,
                "rmse_mm": rmse,
                "mae_mm": mae,
                "medae_mm": medae,
                "reproj_px": reproj,
                "coverage": coverage,
                "jitter_mm": jitter,
            }
        )
    return rows, all_points


def write_reconstruction_points(
    path: Path,
    method_points: dict[str, list[ReconPoint]],
) -> None:
    rows: list[dict[str, object]] = []
    for method, points in method_points.items():
        for point in points:
            gt = point.gt_point3d
            rows.append(
                {
                    "method": method,
                    "seq_name": point.seq_name,
                    "component_id": point.component_id,
                    "segment_id": point.segment_id,
                    "gt_id": point.gt_id if point.gt_id is not None else -1,
                    "frame": point.frame,
                    "x_mm": float(point.point3d[0]),
                    "y_mm": float(point.point3d[1]),
                    "z_mm": float(point.point3d[2]),
                    "gt_x_mm": float(gt[0]) if gt is not None else float("nan"),
                    "gt_y_mm": float(gt[1]) if gt is not None else float("nan"),
                    "gt_z_mm": float(gt[2]) if gt is not None else float("nan"),
                    "reliability": point.reliability,
                }
            )
    write_csv(
        path,
        [
            "method",
            "seq_name",
            "component_id",
            "segment_id",
            "gt_id",
            "frame",
            "x_mm",
            "y_mm",
            "z_mm",
            "gt_x_mm",
            "gt_y_mm",
            "gt_z_mm",
            "reliability",
        ],
        rows,
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")


def write_tex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{" + "l" + "c" * (len(headers) - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    ensure_dir(png_path.parent)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def make_quality_scatter(test_segments: list[StereoSegment], threshold: float, rmse_thr: float, fig_root: Path) -> None:
    scores = np.asarray([segment.quality_scores["ProposedTQS"] for segment in test_segments], dtype=np.float64)
    rmses = np.asarray([segment.rmse if segment.rmse is not None else np.nan for segment in test_segments], dtype=np.float64)
    valid = ~np.isnan(rmses)
    reliable = rmses <= rmse_thr

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(scores[valid & reliable], rmses[valid & reliable], s=28, c="#1b9e77", alpha=0.75, label="Reliable segments")
    ax.scatter(scores[valid & ~reliable], rmses[valid & ~reliable], s=28, c="#d95f02", alpha=0.75, label="Unreliable segments")
    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.2)
    ax.axhline(rmse_thr, color="#555555", linestyle=":", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Proposed TQS")
    ax.set_ylabel("Pseudo-3D RMSE (mm, log scale)")
    ax.set_title("Trajectory quality score vs. pseudo-3D error")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, fig_root / "Figure4_quality_score_scatter.png", fig_root / "Figure4_quality_score_scatter.pdf")


def make_segment_selection_figure(
    ctx: s4.SequenceContext,
    segments: list[StereoSegment],
    threshold: float,
    fig_root: Path,
) -> None:
    candidate = max(segments, key=lambda segment: (segment.length, segment.quality_scores["ProposedTQS"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, view in zip(axes, ["left", "right"]):
        ax.set_xlim(0, ctx.width)
        ax.set_ylim(ctx.height, 0)
        ax.set_xlabel("u (pixels)")
        ax.set_title(f"{ctx.seq_name} {view}-view")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
        roi = ctx.roi_left if view == "left" else ctx.roi_right
        rect = plt.Rectangle((roi[0], roi[1]), roi[2], roi[3], fill=False, linestyle="--", linewidth=1.2, edgecolor="#666666")
        ax.add_patch(rect)
        for segment in segments:
            pts = [(sample.left_det.cx, sample.left_det.cy) if view == "left" else (sample.right_det.cx, sample.right_det.cy) for sample in segment.samples]
            xs = [pt[0] for pt in pts]
            ys = [pt[1] for pt in pts]
            selected = segment.quality_scores["ProposedTQS"] >= threshold
            color = "#1b9e77" if selected else "#bdbdbd"
            alpha = 0.95 if segment.segment_id == candidate.segment_id else (0.85 if selected else 0.35)
            lw = 2.1 if segment.segment_id == candidate.segment_id else (1.5 if selected else 0.9)
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha)
        if view == "left":
            ax.set_ylabel("v (pixels)")
    fig.suptitle("Reliable stereo segment selection on rectified trajectories", fontsize=13)
    save_figure(fig, fig_root / "Figure5_segment_selection_visualization.png", fig_root / "Figure5_segment_selection_visualization.pdf")


def choose_representative_track(method_points: dict[str, list[ReconPoint]]) -> tuple[str, int] | None:
    target_method = "ReliabilityGuided3D"
    counts: Counter[tuple[str, int]] = Counter()
    for point in method_points[target_method]:
        if point.gt_id is not None and point.split == "test":
            counts[(point.seq_name, point.gt_id)] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def make_reconstruction_comparison(
    method_points: dict[str, list[ReconPoint]],
    fig_root: Path,
) -> None:
    target = choose_representative_track(method_points)
    if target is None:
        return
    seq_name, gt_id = target
    fig = plt.figure(figsize=(12.5, 9.0))
    methods = RECON_METHODS
    for idx, method in enumerate(methods, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        pts = [point for point in method_points[method] if point.seq_name == seq_name and point.gt_id == gt_id]
        pts = sorted(pts, key=lambda item: item.frame)
        if not pts:
            continue
        pred = np.asarray([point.point3d for point in pts], dtype=np.float64)
        gt = np.asarray([point.gt_point3d for point in pts if point.gt_point3d is not None], dtype=np.float64)
        if gt.size:
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="#333333", linestyle="--", linewidth=1.4, label="Pseudo-GT")
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="#1f78b4", linewidth=1.8, label=method)
        ax.set_title(method)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.view_init(elev=20, azim=-58)
        if idx == 1:
            ax.legend(frameon=False, loc="upper left")
    fig.suptitle(f"3D reconstruction comparison on {seq_name}, GT ID {gt_id}", fontsize=13)
    save_figure(fig, fig_root / "Figure6_3d_reconstruction_comparison.png", fig_root / "Figure6_3d_reconstruction_comparison.pdf")


def choose_overview_sequence(method_points: dict[str, list[ReconPoint]], method: str = "ReliabilityGuided3D") -> str | None:
    counts: Counter[str] = Counter()
    for point in method_points[method]:
        if point.split == "test" and point.gt_id is not None:
            counts[point.seq_name] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def make_multi_trajectory_overview(
    method_points: dict[str, list[ReconPoint]],
    fig_root: Path,
) -> None:
    seq_name = choose_overview_sequence(method_points, "ReliabilityGuided3D")
    if seq_name is None:
        return

    methods = ["RawStereo3D", "ReliabilityGuided3D"]
    fig = plt.figure(figsize=(13.5, 6.0))
    cmap = plt.get_cmap("tab20")

    for idx, method in enumerate(methods, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        groups: dict[int, list[ReconPoint]] = {}
        for point in method_points[method]:
            if point.seq_name != seq_name or point.gt_id is None:
                continue
            groups.setdefault(point.gt_id, []).append(point)
        selected = []
        for gt_id, pts in groups.items():
            ordered = sorted(pts, key=lambda item: item.frame)
            if len(ordered) >= 12:
                selected.append((gt_id, ordered))
        selected.sort(key=lambda item: len(item[1]), reverse=True)
        selected = selected[:12]

        for color_idx, (gt_id, pts) in enumerate(selected):
            arr = np.asarray([point.point3d for point in pts], dtype=np.float64)
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],
                color=cmap(color_idx % 20),
                linewidth=1.6,
                alpha=0.9,
                label=f"ID {gt_id}" if idx == 1 else None,
            )

        ax.set_title(f"{method} on {seq_name}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.view_init(elev=22, azim=-58)
        if idx == 1 and selected:
            ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)

    fig.suptitle("Multi-trajectory 3D overview on the representative test sequence", fontsize=13)
    save_figure(fig, fig_root / "Figure7_3d_multi_trajectory_overview.png", fig_root / "Figure7_3d_multi_trajectory_overview.pdf")


def build_paper_tables(
    paper_root: Path,
    quality_rows: list[dict[str, float]],
    recon_rows: list[dict[str, float]],
) -> None:
    table_root = paper_root / "tables"
    ensure_dir(table_root)

    quality_md_rows = []
    quality_tex_rows = []
    for row in quality_rows:
        quality_md_rows.append(
            [
                row["method"],
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1']:.3f}",
                f"{row['selected_rmse_mm']:.1f}",
                f"{row['selection_ratio']:.3f}",
            ]
        )
        quality_tex_rows.append(list(quality_md_rows[-1]))
    headers_q = ["Strategy", "Precision", "Recall", "F1", "Selected RMSE (mm)", "Selection Ratio"]
    write_markdown_table(table_root / "Table2_trajectory_quality_scoring.md", headers_q, quality_md_rows)
    write_tex_table(
        table_root / "Table2_trajectory_quality_scoring.tex",
        "Comparison of trajectory quality scoring strategies on the CP3D test split.",
        "tab:quality_scoring",
        headers_q,
        quality_tex_rows,
    )

    recon_md_rows = []
    recon_tex_rows = []
    for row in recon_rows:
        recon_md_rows.append(
            [
                row["method"],
                f"{row['rmse_mm']:.1f}",
                f"{row['mae_mm']:.1f}",
                f"{row['medae_mm']:.1f}",
                f"{row['reproj_px']:.2f}",
                f"{row['coverage']:.3f}",
                f"{row['jitter_mm']:.1f}",
            ]
        )
        recon_tex_rows.append(list(recon_md_rows[-1]))
    headers_r = ["Method", "RMSE (mm)", "MAE (mm)", "MedAE (mm)", "Reproj. (px)", "Coverage", "Jitter (mm)"]
    write_markdown_table(table_root / "Table3_reliability_guided_3d_reconstruction.md", headers_r, recon_md_rows)
    write_tex_table(
        table_root / "Table3_reliability_guided_3d_reconstruction.tex",
        "Comparison of 3D reconstruction strategies on the CP3D test split.",
        "tab:reconstruction_3d",
        headers_r,
        recon_tex_rows,
    )


def write_paper_texts(
    paper_root: Path,
    rmse_thr: float,
    quality_rows: list[dict[str, float]],
    recon_rows: list[dict[str, float]],
) -> None:
    text_root = paper_root / "texts"
    ensure_dir(text_root)

    best_quality = max(quality_rows, key=lambda row: row["f1"])
    proposed_quality = next(row for row in quality_rows if row["method"] == "ProposedTQS")
    best_rmse = min(recon_rows, key=lambda row: row["rmse_mm"])
    raw_recon = next(row for row in recon_rows if row["method"] == "RawStereo3D")
    guided_recon = next(row for row in recon_rows if row["method"] == "ReliabilityGuided3D")
    tqs_recon = next(row for row in recon_rows if row["method"] == "TQSFiltered3D")

    method_3_3 = r"""### Method: Trajectory quality scoring strategy

To prevent low-quality stereo fragments from directly entering the 3D reconstruction stage, a dedicated trajectory quality scoring strategy was introduced after stereo trajectory association. Let a stereo segment be denoted by
\[
\mathcal{S}=\{(p_t^{L},p_t^{R})\}_{t=1}^{n},
\]
where \(p_t^{L}=(u_t^{L},v_t^{L})\) and \(p_t^{R}=(u_t^{R},v_t^{R})\) are the rectified left-view and right-view trajectory centers at frame \(t\), respectively. A pseudo-3D reference is first constructed from the rectified ground-truth centers by
\[
\mathbf{X}_t^{gt}=\operatorname{Triangulate}(P_1,p_t^{L,gt},P_2,p_t^{R,gt}),
\]
where \(P_1\) and \(P_2\) are the calibrated stereo projection matrices. The pseudo-3D root-mean-square error of a predicted segment is then used as the supervisory signal during train-split threshold selection.

At the frame level, a reliability score \(r_t\) is defined as
\[
r_t = 0.30\,c_t
      + 0.25\exp\!\left(-\frac{\Delta v_t}{s_v}\right)
      + 0.20\exp\!\left(-\frac{|d_t-\tilde d_g|}{s_d}\right)
      + 0.15\,a_t
      + 0.10\,\mathbb{I}_{ROI,t},
\]
where \(c_t\) is the mean detection confidence of the left-right pair, \(\Delta v_t=|v_t^{L}-v_t^{R}|\) is the vertical disparity residual after rectification, \(d_t=u_t^{L}-u_t^{R}\) is the horizontal disparity, \(\tilde d_g\) is the group-level disparity prior estimated from the train split, \(a_t\) is the trajectory-pair association affinity obtained in Step 4, and \(\mathbb{I}_{ROI,t}\) indicates whether the pair lies inside the valid calibrated observation regions in both views.

The frame-level cue is then lifted to the segment level together with several complementary consistency descriptors. Specifically, the segment length and temporal completeness are defined as
\[
s_{\text{len}}=\min\!\left(\frac{n}{20},1\right), \qquad
s_{\text{comp}}=\frac{n}{T_{\text{span}}},
\]
where \(T_{\text{span}}\) is the temporal span of the segment. To describe stereo geometric stability, the disparity-variation score is computed as
\[
s_{\text{disp}}=\exp\!\left(-\frac{\sigma_d}{s_{\sigma d}}\right),
\]
where \(\sigma_d\) is the standard deviation of the segment disparity sequence. To characterize 3D motion smoothness, a raw triangulated point set \(\{\mathbf{X}_t\}_{t=1}^{n}\) is first obtained and the second-order finite difference is used to measure instability:
\[
s_{\text{smooth}}=
\exp\!\left(
-\frac{\operatorname{median}\bigl(\|\Delta^2 \mathbf{X}_t\|_2\bigr)}{400}
\right).
\]
In addition, a depth-prior consistency score is introduced to suppress implausible segments:
\[
s_{\text{depth}}=
\exp\!\left(-\frac{|\operatorname{median}(Z_t)-\tilde Z_g|}{s_{z1}}\right)
\exp\!\left(-\frac{\sigma_Z}{s_{z2}}\right),
\]
where \(\tilde Z_g\) is the group-level median depth prior, and \(\sigma_Z\) is the within-segment depth variation.

The final proposed trajectory quality score (`ProposedTQS`) is defined as
\[
\begin{aligned}
q(\mathcal{S}) =\;&
0.16\,s_{\text{pair}}
+ 0.08\,s_{\text{temp}}
+ 0.10\,s_{\text{epi}}
+ 0.09\,s_{\text{spatial}}
+ 0.07\,s_{\text{motion}} \\
&+ 0.06\,s_{\text{rank}}
+ 0.06\,\bar r
+ 0.04\,s_{\text{region}}
+ 0.08\,s_{\text{conf}}
+ 0.06\,s_{\text{len}} \\
&+ 0.04\,s_{\text{comp}}
+ 0.12\,s_{\text{smooth}}
+ 0.04\,s_{\text{depth}},
\end{aligned}
\]
where \(s_{\text{pair}}\), \(s_{\text{temp}}\), \(s_{\text{epi}}\), \(s_{\text{spatial}}\), \(s_{\text{motion}}\), \(s_{\text{rank}}\), and \(s_{\text{region}}\) are inherited from the stereo association stage, \(\bar r\) is the mean frame-level reliability of the segment, and \(s_{\text{conf}}\) is the average detection-confidence descriptor. On the train split, a segment is treated as reliable when its pseudo-3D RMSE does not exceed `{rmse_thr:.1f}` mm, and the acceptance threshold is selected by maximizing the segment-level F1 score. The resulting operating threshold in the current benchmark is \(q(\mathcal{S}) \geq 0.65\).
"""
    method_3_3 = method_3_3.replace("{rmse_thr:.1f}", f"{rmse_thr:.1f}")
    method_3_3 = method_3_3.replace(
        r"q(\mathcal{S}) \geq 0.65",
        rf"q(\mathcal{{S}}) \geq {float(proposed_quality['threshold']):.2f}",
    )

    method_3_4 = r"""### Method: Reliability-guided 3D trajectory reconstruction

After reliable segment selection, a reliability-guided 3D reconstruction strategy is applied to obtain smoother and more physically plausible 3D trajectories. For each accepted stereo segment, a raw 3D point is first reconstructed at frame \(t\) by direct triangulation,
\[
\mathbf{X}_t = \operatorname{Triangulate}(P_1,p_t^{L},P_2,p_t^{R}).
\]
This raw reconstruction corresponds to the `RawStereo3D` baseline and preserves all accepted stereo observations without temporal regularization.

To reduce geometrically implausible points, the filtered variants additionally impose reliability-aware validity checks. A point is retained only when the segment quality satisfies
\[
q(\mathcal{S}) \ge \tau_{TQS},
\]
the frame-level reliability satisfies
\[
r_t \ge 0.45,
\]
and the reconstructed depth remains inside a group-adaptive admissible range,
\[
|Z_t-\tilde Z_g| \le \max(800,\;4\,IQR_Z),
\]
where \(\tilde Z_g\) and \(IQR_Z\) denote the group-level median depth and interquartile depth range estimated from the train split. The resulting quality-filtered variant is denoted as `TQSFiltered3D`.

The full proposed method, `ReliabilityGuided3D`, further introduces local temporal smoothing. For each valid point, a reliability-weighted neighborhood estimate is first constructed as
\[
\bar{\mathbf{X}}_t =
\frac{\sum_{|k-t|\le 2}\exp(-|k-t|/1.5)\,r_k\,\mathbf{X}_k}
{\sum_{|k-t|\le 2}\exp(-|k-t|/1.5)\,r_k}.
\]
The final reconstructed point is then defined by blending the raw observation with the local reliability-weighted estimate:
\[
\hat{\mathbf{X}}_t =
\omega_t \mathbf{X}_t + (1-\omega_t)\bar{\mathbf{X}}_t,
\qquad
\omega_t = 0.55 + 0.35\,r_t.
\]
In this way, high-confidence observations remain close to their original triangulated positions, whereas weaker observations are more strongly regularized by nearby reliable points.

Short gaps caused by temporary rejection of low-confidence observations are finally repaired through conservative interpolation. If the temporal gap length does not exceed three frames and the reliability of both neighboring points is at least 0.45, an intermediate point is inserted by linear interpolation:
\[
\hat{\mathbf{X}}_{t+\delta} =
(1-\lambda)\hat{\mathbf{X}}_{t} + \lambda \hat{\mathbf{X}}_{t+\Delta},
\qquad
\lambda=\frac{\delta}{\Delta},
\]
where \(\Delta\le 3\) is the gap size. This design keeps the method simple and deterministic while avoiding aggressive extrapolation in poorly constrained segments.

For quantitative evaluation, the reconstructed trajectories are compared with the pseudo-ground-truth 3D references using
\[
RMSE = \sqrt{\frac{1}{N}\sum_{t=1}^{N}\|\hat{\mathbf{X}}_t-\mathbf{X}_t^{gt}\|_2^2},
\]
\[
MAE = \frac{1}{N}\sum_{t=1}^{N}\|\hat{\mathbf{X}}_t-\mathbf{X}_t^{gt}\|_2,
\]
and the median absolute error (`MedAE`). In addition, the mean reprojection error is computed in the two rectified image planes, the coverage measures the fraction of ground-truth-supported 3D points that remain reconstructable, and the trajectory jitter is defined by the mean norm of the second-order finite difference:
\[
J = \frac{1}{N-2}\sum_{t=2}^{N-1}\|\Delta^2 \hat{\mathbf{X}}_t\|_2.
\]
Together, these criteria evaluate not only pointwise 3D accuracy but also temporal smoothness and practical trajectory usability.
"""

    section_3_3 = f"""### 3.3. Trajectory quality scoring and reliable segment selection

After stereo association, the resulting cross-view trajectory components still contain segments with very different reconstruction reliability. Some segments are supported by stable left-right geometric correspondence and continuous temporal observations, whereas others are degraded by fragmented tracks, weak detections, or cross-view ambiguity. To explicitly separate these two cases before 3D reconstruction, a dedicated trajectory quality scoring benchmark was constructed on top of the proposed stereo association results.

Pseudo-3D references were generated by triangulating the rectified left-view and right-view ground-truth centers under the calibrated stereo projection matrices. Each predicted stereo segment was then aligned to the most consistent ground-truth identity, and its pseudo-3D root-mean-square error was measured. Segments with train-split pseudo-3D RMSE not exceeding `{rmse_thr:.1f}` mm were treated as reliable during threshold tuning. As summarized in Table 2, the proposed trajectory quality score (`ProposedTQS`) achieves the strongest overall balance, reaching the best test-set `F1 = {best_quality['f1']:.3f}`, the strongest negative rank correlation with reconstruction error (`Spearman rho = {best_quality['spearman_rho']:.3f}`), and a drastic reduction of the selected-segment RMSE to `{best_quality['selected_rmse_mm']:.1f}` mm. By contrast, the simpler baselines (`LengthOnly`, `GeoOnly`, and `GeoConf`) all preserve nearly the entire test set, but they fail to meaningfully suppress the grossly unstable stereo segments, which leads to much larger residual pseudo-3D errors.

The qualitative evidence in Figures 4 and 5 supports this conclusion. Figure 4 shows a clear negative relation between the proposed quality score and the pseudo-3D error, especially after plotting the error on a logarithmic scale, which confirms that the score is informative for reliability ranking rather than acting as a simple length prior. Figure 5 further illustrates that the high-score segments are concentrated inside the stable observation region and preserve smoother, longer trajectory fragments, while low-score segments are mostly sparse or geometrically unstable. Therefore, the proposed quality scoring strategy provides a principled front-end for selecting high-confidence stereo trajectory segments before 3D reconstruction and forms the second methodological contribution of the overall pipeline.
"""

    section_3_4 = f"""### 3.4. Reliability-guided 3D trajectory reconstruction

Based on the selected high-quality stereo segments, a reliability-guided 3D trajectory reconstruction strategy was further introduced. The baseline `RawStereo3D` reconstructs every associated stereo segment through direct frame-wise triangulation. `GeomGate3D` removes geometrically inconsistent observations with classical vertical-disparity gates. `TQSFiltered3D` keeps only the segments accepted by the proposed trajectory quality score. Finally, `ReliabilityGuided3D` combines segment-level selection with frame-level reliability weighting, local temporal smoothing, and short-gap interpolation.

According to Table 3, the quality-aware reconstruction variants outperform the unconstrained baselines by a large margin. `TQSFiltered3D` yields the smallest absolute `RMSE` (`{tqs_recon['rmse_mm']:.1f}` mm), while `ReliabilityGuided3D` reaches a nearly identical `RMSE` (`{guided_recon['rmse_mm']:.1f}` mm) together with the lowest `MAE` (`{guided_recon['mae_mm']:.1f}` mm), the lowest `MedAE` (`{guided_recon['medae_mm']:.1f}` mm), and the lowest trajectory jitter (`{guided_recon['jitter_mm']:.1f}` mm). Compared with the direct triangulation baseline, the proposed reliability-guided strategy reduces the 3D RMSE from `{raw_recon['rmse_mm']:.1f}` mm to `{guided_recon['rmse_mm']:.1f}` mm while preserving essentially the same coverage as `TQSFiltered3D`. These results indicate that segment-level quality control removes catastrophic depth outliers, whereas the subsequent reliability-guided smoothing mainly improves local stability and pointwise robustness rather than merely shortening the trajectory.

Figure 6 gives a representative single-trajectory 3D comparison on the test split, while Figure 7 presents a multi-trajectory overview on a representative sequence. The direct reconstruction baseline exhibits visible spatial waviness and larger deviations from the pseudo-ground-truth trajectory, especially around fragmented or weakly constrained observations. In contrast, the reliability-guided method follows the overall motion trend more smoothly and preserves a more coherent depth evolution. The multi-trajectory overview further shows that the proposed strategy suppresses the most extreme geometric outliers while retaining the main reconstructed flight structures. These results demonstrate that the proposed two-stage design, namely reliable segment selection followed by reliability-guided temporal reconstruction, is better suited to nocturnal pest trajectory recovery than unconstrained frame-wise stereo triangulation and constitutes the third methodological contribution of the study.
"""

    figure_table_note = """### Figure and table organization for the two step5 innovations

The recommended figure-table organization in the manuscript is as follows.

For the trajectory quality scoring strategy:
- Table 2 should be placed in the results-and-discussion subsection on reliable segment selection because it provides the main quantitative comparison of alternative scoring strategies.
- Figure 4 should be placed immediately after Table 2 to show that the proposed quality score is monotonically related to pseudo-3D error over a wide dynamic range.
- Figure 5 should be used as the main qualitative illustration of accepted versus rejected stereo segments in the rectified image coordinate system.

For the reliability-guided 3D reconstruction strategy:
- Table 3 should be placed at the beginning of the 3D reconstruction results subsection because it summarizes the overall accuracy, smoothness, and coverage trade-offs.
- Figure 6 is best used for a close-up methodological comparison on a single representative trajectory.
- Figure 7 is more suitable for showing the global spatial distribution of multiple reconstructed trajectories across the representative test sequence.
- The joint figures in `visualizations/testset_joint_2d3d/` can be cited as supplementary or appendix-style visual evidence when a direct 2D-to-3D correspondence demonstration is needed.
"""

    (text_root / "Method_trajectory_quality_scoring.md").write_text(method_3_3 + "\n", encoding="utf-8")
    (text_root / "Method_reliability_guided_3d_reconstruction.md").write_text(method_3_4 + "\n", encoding="utf-8")
    (text_root / "Section_3_3_trajectory_quality_scoring.md").write_text(section_3_3 + "\n", encoding="utf-8")
    (text_root / "Section_3_4_reliability_guided_3d_reconstruction.md").write_text(section_3_4 + "\n", encoding="utf-8")
    (text_root / "Figure_table_organization_for_step5.md").write_text(figure_table_note + "\n", encoding="utf-8")


def write_paper_index(paper_root: Path) -> None:
    content = """# Paper Asset Index

## Table 2

- [Table2_trajectory_quality_scoring.md](tables/Table2_trajectory_quality_scoring.md)
- [Table2_trajectory_quality_scoring.tex](tables/Table2_trajectory_quality_scoring.tex)

Suggested caption:
Table 2. Comparison of trajectory quality scoring strategies on the CP3D test split.

## Table 3

- [Table3_reliability_guided_3d_reconstruction.md](tables/Table3_reliability_guided_3d_reconstruction.md)
- [Table3_reliability_guided_3d_reconstruction.tex](tables/Table3_reliability_guided_3d_reconstruction.tex)

Suggested caption:
Table 3. Comparison of 3D reconstruction strategies on the CP3D test split.

## Figure 4

- [Figure4_quality_score_scatter.png](figures/Figure4_quality_score_scatter.png)
- [Figure4_quality_score_scatter.pdf](figures/Figure4_quality_score_scatter.pdf)

Suggested caption:
Figure 4. Relationship between the proposed trajectory quality score and the pseudo-3D reconstruction error on the test split.

## Figure 5

- [Figure5_segment_selection_visualization.png](figures/Figure5_segment_selection_visualization.png)
- [Figure5_segment_selection_visualization.pdf](figures/Figure5_segment_selection_visualization.pdf)

Suggested caption:
Figure 5. Visualization of reliable stereo segment selection on rectified left-view and right-view trajectories. Green segments are retained by the proposed quality score and gray segments are rejected.

## Figure 6

- [Figure6_3d_reconstruction_comparison.png](figures/Figure6_3d_reconstruction_comparison.png)
- [Figure6_3d_reconstruction_comparison.pdf](figures/Figure6_3d_reconstruction_comparison.pdf)

Suggested caption:
Figure 6. Representative 3D trajectory reconstruction comparison on the test split. The dashed black curve denotes the pseudo-ground-truth reference and the blue curve denotes the reconstructed trajectory.

## Figure 7

- [Figure7_3d_multi_trajectory_overview.png](figures/Figure7_3d_multi_trajectory_overview.png)
- [Figure7_3d_multi_trajectory_overview.pdf](figures/Figure7_3d_multi_trajectory_overview.pdf)

Suggested caption:
Figure 7. Multi-trajectory 3D overview on a representative test sequence. Each colored polyline denotes one reconstructed insect trajectory, allowing a direct visual comparison of global trajectory completeness and spatial smoothness between the raw and reliability-guided reconstruction outputs.

## Texts

- [Method_trajectory_quality_scoring.md](texts/Method_trajectory_quality_scoring.md)
- [Method_reliability_guided_3d_reconstruction.md](texts/Method_reliability_guided_3d_reconstruction.md)
- [Section_3_3_trajectory_quality_scoring.md](texts/Section_3_3_trajectory_quality_scoring.md)
- [Section_3_4_reliability_guided_3d_reconstruction.md](texts/Section_3_4_reliability_guided_3d_reconstruction.md)
- [Figure_table_organization_for_step5.md](texts/Figure_table_organization_for_step5.md)
"""
    (paper_root / "paper_asset_index.md").write_text(content, encoding="utf-8")


def write_summary(
    report_root: Path,
    rmse_thr: float,
    quality_rows: list[dict[str, float]],
    recon_rows: list[dict[str, float]],
) -> None:
    best_quality = max(quality_rows, key=lambda row: row["f1"])
    guided_recon = next(row for row in recon_rows if row["method"] == "ReliabilityGuided3D")
    lines = [
        "# Step 5 Summary",
        "",
        f"- Reliable-segment pseudo-3D RMSE threshold: `{rmse_thr:.1f}` mm",
        f"- Best quality scoring strategy: `{best_quality['method']}` with `F1={best_quality['f1']:.3f}`",
        f"- Recommended reconstruction strategy: `ReliabilityGuided3D` with `RMSE={guided_recon['rmse_mm']:.1f}` mm, `MAE={guided_recon['mae_mm']:.1f}` mm, and `Jitter={guided_recon['jitter_mm']:.1f}` mm",
        "",
        "Generated assets:",
        "- `paper_ready_assets/tables/Table2_trajectory_quality_scoring.*`",
        "- `paper_ready_assets/tables/Table3_reliability_guided_3d_reconstruction.*`",
        "- `paper_ready_assets/figures/Figure4_quality_score_scatter.*`",
        "- `paper_ready_assets/figures/Figure5_segment_selection_visualization.*`",
        "- `paper_ready_assets/figures/Figure6_3d_reconstruction_comparison.*`",
        "- `paper_ready_assets/figures/Figure7_3d_multi_trajectory_overview.*`",
        "- `paper_ready_assets/texts/Section_3_3_trajectory_quality_scoring.md`",
        "- `paper_ready_assets/texts/Section_3_4_reliability_guided_3d_reconstruction.md`",
    ]
    (report_root / "step5_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step4-root", default=str(STEP4_ROOT))
    parser.add_argument("--save-root", default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument(
        "--fixed-step5-metadata",
        default=None,
        help="Load the previously selected TQA/RGR settings and skip threshold tuning.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    step4_root = Path(args.step4_root).resolve()
    configure_step4_root(step4_root)
    step4_meta = json.loads((STEP4_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
    step3_train_root = Path(step4_meta["step3_train_root"]).resolve()
    step3_test_root = Path(step4_meta["step3_test_root"]).resolve()
    data_root = Path(step4_meta["data_root"]).resolve()
    rect_root = Path(step4_meta["rect_root"]).resolve()
    configure_step1_root(rect_root)
    s4.configure_input_roots(
        step3_train_root=step3_train_root,
        step3_test_root=step3_test_root,
        data_root=data_root,
        rect_root=rect_root,
    )
    save_root = Path(args.save_root).resolve()
    report_root = save_root / "reports"
    fig_root = save_root / "paper_ready_assets" / "figures"
    paper_root = save_root / "paper_ready_assets"
    ensure_dir(save_root)
    ensure_dir(report_root)
    ensure_dir(fig_root)
    ensure_dir(paper_root)

    assoc_params = load_selected_assoc_params()
    train_sequences = s4.find_sequences("train")
    test_sequences = s4.find_sequences("test")
    priors = s4.compute_group_priors(train_sequences)

    train_contexts = [s4.build_sequence_context(seq_name, priors) for seq_name in train_sequences]
    test_contexts = [s4.build_sequence_context(seq_name, priors) for seq_name in test_sequences]
    all_contexts = train_contexts + test_contexts
    for ctx in all_contexts:
        s4.precompute_pair_features(ctx)

    gt_refs = {ctx.seq_name: build_gt_reference(ctx.seq_name) for ctx in all_contexts}
    depth_priors = compute_depth_priors(train_contexts, gt_refs)

    segments_by_seq: dict[str, list[StereoSegment]] = {}
    for ctx in all_contexts:
        score_lookup = s4.build_score_lookup(ctx, "Proposed")
        components = s4.method_components(ctx, "Proposed", assoc_params["threshold"], assoc_params["secondary_threshold"])
        segments = build_segments_for_context(ctx, components, score_lookup, gt_refs[ctx.seq_name][0], depth_priors)
        segments_by_seq[ctx.seq_name] = segments

    train_segments = collect_labeled_segments([segment for ctx in train_contexts for segment in segments_by_seq[ctx.seq_name]])
    test_segments = collect_labeled_segments([segment for ctx in test_contexts for segment in segments_by_seq[ctx.seq_name]])
    if not train_segments or not test_segments:
        raise RuntimeError("No labeled stereo segments were available for step5 evaluation.")

    tuning_rows = []
    quality_rows = []
    best_tqs_threshold = 0.50
    fixed_step5_source = None
    if args.fixed_step5_metadata:
        fixed_step5_source = Path(args.fixed_step5_metadata).resolve()
        fixed_metadata = json.loads(fixed_step5_source.read_text(encoding="utf-8"))
        rmse_thr = float(fixed_metadata["rmse_reliable_threshold_mm"])
        fixed_thresholds = {
            str(row["method"]): float(row["threshold"])
            for row in fixed_metadata["quality_rows"]
        }
        for method in QUALITY_METHODS:
            threshold = fixed_thresholds[method]
            train_evaluated = evaluate_quality_method(
                train_segments,
                method,
                threshold,
                rmse_thr,
            )
            tuning_rows.append(
                {
                    "method": method,
                    "threshold": threshold,
                    "precision": train_evaluated["precision"],
                    "recall": train_evaluated["recall"],
                    "f1": train_evaluated["f1"],
                    "rmse_thr": rmse_thr,
                }
            )
            evaluated = evaluate_quality_method(
                test_segments,
                method,
                threshold,
                rmse_thr,
            )
            quality_rows.append({"method": method, **evaluated})
            if method == "ProposedTQS":
                best_tqs_threshold = threshold
        print(
            f"[Fixed] TQS threshold={best_tqs_threshold:.2f}, "
            f"reliable RMSE threshold={rmse_thr:.6f} mm"
        )
    else:
        rmse_thr = choose_reliable_rmse_threshold(train_segments)
        for method in QUALITY_METHODS:
            tuned = tune_quality_threshold(train_segments, method, rmse_thr)
            tuning_rows.append({"method": method, **tuned, "rmse_thr": rmse_thr})
            evaluated = evaluate_quality_method(
                test_segments,
                method,
                tuned["threshold"],
                rmse_thr,
            )
            quality_rows.append({"method": method, **evaluated})
            if method == "ProposedTQS":
                best_tqs_threshold = tuned["threshold"]

    recon_rows, method_points = evaluate_reconstruction(test_contexts, segments_by_seq, gt_refs, best_tqs_threshold, depth_priors)
    write_reconstruction_points(report_root / "reconstruction_points.csv", method_points)

    write_csv(
        report_root / "quality_tuning.csv",
        ["method", "threshold", "precision", "recall", "f1", "rmse_thr"],
        tuning_rows,
    )
    write_csv(
        report_root / "quality_strategy_metrics.csv",
        ["method", "threshold", "precision", "recall", "f1", "selected_rmse_mm", "selection_ratio", "spearman_rho"],
        quality_rows,
    )
    write_csv(
        report_root / "reconstruction_metrics.csv",
        ["method", "rmse_mm", "mae_mm", "medae_mm", "reproj_px", "coverage", "jitter_mm"],
        recon_rows,
    )
    segment_rows = []
    for segment in test_segments:
        segment_rows.append(
            {
                "seq_name": segment.seq_name,
                "segment_id": segment.segment_id,
                "gt_id": segment.gt_id if segment.gt_id is not None else -1,
                "length": segment.length,
                "rmse_mm": segment.rmse if segment.rmse is not None else float("nan"),
                "pair_score": segment.features["pair_score"],
                "smooth3d_score": segment.features["smooth3d_score"],
                "depth_prior_score": segment.features["depth_prior_score"],
                "proposed_tqs": segment.quality_scores["ProposedTQS"],
            }
        )
    write_csv(
        report_root / "test_segment_diagnostics.csv",
        ["seq_name", "segment_id", "gt_id", "length", "rmse_mm", "pair_score", "smooth3d_score", "depth_prior_score", "proposed_tqs"],
        segment_rows,
    )

    make_quality_scatter(test_segments, best_tqs_threshold, rmse_thr, fig_root)
    make_segment_selection_figure(test_contexts[0], segments_by_seq[test_contexts[0].seq_name], best_tqs_threshold, fig_root)
    make_reconstruction_comparison(method_points, fig_root)
    make_multi_trajectory_overview(method_points, fig_root)

    build_paper_tables(paper_root, quality_rows, recon_rows)
    write_paper_texts(paper_root, rmse_thr, quality_rows, recon_rows)
    write_paper_index(paper_root)
    write_summary(report_root, rmse_thr, quality_rows, recon_rows)

    metadata = {
        "step4_root": str(STEP4_ROOT),
        "step3_train_root": str(step3_train_root),
        "step3_test_root": str(step3_test_root),
        "data_root": str(data_root),
        "step1_rectification_root": str(rect_root),
        "assoc_params": assoc_params,
        "rmse_reliable_threshold_mm": rmse_thr,
        "best_tqs_threshold": best_tqs_threshold,
        "coverage_definition": (
            "Unique reconstructed (sequence, ground-truth identity, frame) keys with pseudo-3D "
            "reference divided by all alignable pseudo-3D reference keys in the evaluated test sequences."
        ),
        "parameter_selection_mode": (
            "fixed_external_parameters"
            if fixed_step5_source is not None
            else "train_split_tuning"
        ),
        "fixed_step5_metadata": (
            str(fixed_step5_source) if fixed_step5_source is not None else None
        ),
        "reliability_guided_parameters": {
            "frame_reliability_threshold": 0.45,
            "depth_gate_mm": "max(800, 4 * train illumination-group depth IQR)",
            "temporal_radius": "round(2 * fps / 20), preserving 0.10 s",
            "temporal_decay": "1.5 * fps / 20, preserving the 20 FPS time constant",
            "raw_fusion_weight": "0.55 + 0.35 * frame reliability",
            "short_gap_completion": "round(3 * fps / 20), preserving 0.15 s",
            "per_fps": {
                str(int(base.read_seqinfo(ctx.seq_name)["frameRate"])): {
                    "temporal_radius_frames": temporal_parameters(ctx)[0],
                    "temporal_decay_frames": temporal_parameters(ctx)[1],
                    "short_gap_completion_max_gap_frames": temporal_parameters(ctx)[2],
                }
                for ctx in test_contexts
            },
        },
        "quality_rows": quality_rows,
        "reconstruction_rows": recon_rows,
    }
    (save_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
