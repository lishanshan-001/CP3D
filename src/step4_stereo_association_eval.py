# -*- coding: utf-8 -*-

"""
Step 4
Stereo trajectory association on top of tuned ByteTrack outputs.

The script:
1. Loads left/right ByteTrack trajectories from step3 results.
2. Builds several representative stereo association baselines.
3. Tunes every method threshold on the train split using trajectory-pair F0.5.
4. Exports fused left-right MOTChallenge files with shared global IDs.
5. Evaluates the fused sequences with official TrackEval metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
THIRD_PARTY_ROOT = ROOT / "third_party"

import trackeval
from scipy.optimize import linear_sum_assignment

import step3_mot_tracking_eval as base


STEP3_ROOT = ROOT / "results" / "step3_motchallenge_official_tuned"
STEP3_TRAIN_ROOT = STEP3_ROOT
STEP3_TEST_ROOT = STEP3_ROOT
DEFAULT_SAVE_ROOT = ROOT / "results" / "step4_stereo_association"
DATA_ROOT = ROOT / "CP3D_dataset"
RECT_ROOT = ROOT / "results" / "step1_rectification"

METHODS = [
    "FrameVote",
    "Epipolar",
    "EpiSpatial",
    "EpiMotion",
    "EpiRank",
    "Proposed",
]


def configure_step3_root(step3_root: Path) -> None:
    global STEP3_ROOT
    global STEP3_TRAIN_ROOT
    global STEP3_TEST_ROOT

    STEP3_ROOT = step3_root.resolve()
    STEP3_TRAIN_ROOT = STEP3_ROOT
    STEP3_TEST_ROOT = STEP3_ROOT


def configure_input_roots(
    step3_train_root: Path,
    step3_test_root: Path,
    data_root: Path,
    rect_root: Path,
) -> None:
    global STEP3_TRAIN_ROOT
    global STEP3_TEST_ROOT
    global DATA_ROOT
    global RECT_ROOT

    STEP3_TRAIN_ROOT = step3_train_root.resolve()
    STEP3_TEST_ROOT = step3_test_root.resolve()
    DATA_ROOT = data_root.resolve()
    RECT_ROOT = rect_root.resolve()
    base.DATA_ROOT = DATA_ROOT
    base.RECT_ROOT = RECT_ROOT


def step3_root_for_sequence(seq_name: str) -> Path:
    split = base.locate_split(seq_name)
    return STEP3_TRAIN_ROOT if split == "train" else STEP3_TEST_ROOT


def gt_root_for_sequence(seq_name: str, view: str) -> Path:
    dataset_name = f"CP3D-{view}-all"
    return step3_root_for_sequence(seq_name) / "gt" / "mot_challenge" / dataset_name


def track_root_for_sequence(seq_name: str, view: str) -> Path:
    dataset_name = f"CP3D-{view}-all"
    return (
        step3_root_for_sequence(seq_name)
        / "trackers"
        / "mot_challenge"
        / dataset_name
        / "ByteTrack"
        / "data"
    )


@dataclass
class Detection:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w * 0.5

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5


@dataclass
class Track:
    track_id: int
    view: str
    detections: dict[int, Detection]
    dominant_gt_id: int | None = None
    dominant_gt_hits: int = 0

    frames: list[int] = field(default_factory=list)
    centers_u: np.ndarray | None = None
    centers_v: np.ndarray | None = None
    heights: np.ndarray | None = None

    def finalize(self) -> None:
        self.frames = sorted(self.detections)
        self.centers_u = np.array([self.detections[f].cx for f in self.frames], dtype=np.float64)
        self.centers_v = np.array([self.detections[f].cy for f in self.frames], dtype=np.float64)
        self.heights = np.array([self.detections[f].h for f in self.frames], dtype=np.float64)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def length(self) -> int:
        return len(self.frames)

    def overlap_frames(self, other: "Track") -> list[int]:
        if self.end < other.start or other.end < self.start:
            return []
        return sorted(set(self.frames).intersection(other.frames))


@dataclass
class GroupPrior:
    disp_median: float
    disp_iqr: float
    disp_p05: float
    disp_p95: float
    v_median: float
    v_p90: float
    v_p95: float


@dataclass
class PairFeature:
    seq_name: str
    left_id: int
    right_id: int
    overlap: int
    temporal_score: float
    epi_score: float
    spatial_score: float
    motion_score: float
    rank_score: float
    local_score: float
    region_score: float
    mean_vdiff: float
    disp_median: float
    disp_std: float
    train_positive: bool
    train_gt_id: int | None


@dataclass
class SequenceContext:
    seq_name: str
    split: str
    width: int
    height: int
    seq_length: int
    left_tracks: dict[int, Track]
    right_tracks: dict[int, Track]
    left_frame_index: dict[int, list[Detection]]
    right_frame_index: dict[int, list[Detection]]
    left_gt_frame_index: dict[int, list[Detection]]
    right_gt_frame_index: dict[int, list[Detection]]
    left_rank_map: dict[int, dict[int, tuple[int, int]]]
    right_rank_map: dict[int, dict[int, tuple[int, int]]]
    roi_left: tuple[int, int, int, int]
    roi_right: tuple[int, int, int, int]
    group_prior: GroupPrior
    pair_features: list[PairFeature] = field(default_factory=list)
    frame_vote_lookup: dict[tuple[int, int], float] | None = None


def natural_sort(items: list[str]) -> list[str]:
    return base.natural_sort(items)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_sequences(split: str) -> list[str]:
    split_dir = DATA_ROOT / split
    if not split_dir.is_dir():
        return []
    return natural_sort([p.name for p in split_dir.glob("CP-*") if p.is_dir()])


def seq_group(seq_name: str) -> str:
    return "-".join(seq_name.split("-")[:2])


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def exp_score(value: float, scale: float) -> float:
    scale = max(scale, 1e-6)
    return math.exp(-max(0.0, value) / scale)


def safe_mean(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(np.mean(values))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_roi(seq_name: str, view: str) -> tuple[int, int, int, int]:
    roi_name = "roi1.npy" if view == "left" else "roi2.npy"
    roi = np.load(RECT_ROOT / seq_name / roi_name).astype(np.int32).tolist()
    return int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])


def inside_roi(det: Detection, roi: tuple[int, int, int, int]) -> bool:
    rx, ry, rw, rh = roi
    return rx <= det.cx <= rx + rw and ry <= det.cy <= ry + rh


def load_mot_file(path: Path) -> tuple[dict[int, Track], dict[int, list[Detection]]]:
    tracks: dict[int, Track] = {}
    frame_index: dict[int, list[Detection]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            det = Detection(
                frame=frame,
                track_id=track_id,
                x=float(parts[2]),
                y=float(parts[3]),
                w=float(parts[4]),
                h=float(parts[5]),
                score=float(parts[6]) if len(parts) > 6 else 1.0,
            )
            frame_index[frame].append(det)
            if track_id not in tracks:
                tracks[track_id] = Track(track_id=track_id, view="", detections={})
            tracks[track_id].detections[frame] = det
    for track in tracks.values():
        track.finalize()
    return tracks, frame_index


def set_track_view(tracks: dict[int, Track], view: str) -> None:
    for track in tracks.values():
        track.view = view


def build_rank_map(frame_index: dict[int, list[Detection]]) -> dict[int, dict[int, tuple[int, int]]]:
    rank_map: dict[int, dict[int, tuple[int, int]]] = {}
    for frame, dets in frame_index.items():
        ordered = sorted(dets, key=lambda det: det.cx)
        size = len(ordered)
        rank_map[frame] = {det.track_id: (rank, size) for rank, det in enumerate(ordered)}
    return rank_map


def load_rectified_gt(seq_name: str, view: str) -> tuple[dict[int, Track], dict[int, list[Detection]]]:
    root = gt_root_for_sequence(seq_name, view)
    tracks, frame_index = load_mot_file(root / seq_name / "gt" / "gt.txt")
    set_track_view(tracks, view)
    return tracks, frame_index


def assign_dominant_gt(pred_tracks: dict[int, Track], gt_frame_index: dict[int, list[Detection]]) -> None:
    for track in pred_tracks.values():
        vote_counter: Counter[int] = Counter()
        for frame in track.frames:
            pred_det = track.detections[frame]
            gt_dets = gt_frame_index.get(frame, [])
            best_gt_id = None
            best_dist = float("inf")
            best_iou = 0.0
            for gt_det in gt_dets:
                dist = math.hypot(pred_det.cx - gt_det.cx, pred_det.cy - gt_det.cy)
                x1 = max(pred_det.x, gt_det.x)
                y1 = max(pred_det.y, gt_det.y)
                x2 = min(pred_det.x + pred_det.w, gt_det.x + gt_det.w)
                y2 = min(pred_det.y + pred_det.h, gt_det.y + gt_det.h)
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = pred_det.w * pred_det.h + gt_det.w * gt_det.h - inter
                iou = inter / union if union > 1e-9 else 0.0
                max_dist = max(18.0, 0.9 * (pred_det.h + gt_det.h))
                if iou >= 0.05 or dist <= max_dist:
                    if iou > best_iou or (abs(iou - best_iou) < 1e-9 and dist < best_dist):
                        best_gt_id = gt_det.track_id
                        best_dist = dist
                        best_iou = iou
            if best_gt_id is not None:
                vote_counter[best_gt_id] += 1
        if vote_counter:
            gt_id, hits = vote_counter.most_common(1)[0]
            if hits >= max(3, int(math.ceil(track.length * 0.15))):
                track.dominant_gt_id = gt_id
                track.dominant_gt_hits = hits


def compute_group_priors(train_sequences: list[str]) -> dict[str, GroupPrior]:
    disp_stats: dict[str, list[float]] = defaultdict(list)
    v_stats: dict[str, list[float]] = defaultdict(list)
    for seq_name in train_sequences:
        left_tracks, _ = load_rectified_gt(seq_name, "left")
        right_tracks, _ = load_rectified_gt(seq_name, "right")
        for track_id, left_track in left_tracks.items():
            right_track = right_tracks.get(track_id)
            if right_track is None:
                continue
            overlap = left_track.overlap_frames(right_track)
            for frame in overlap:
                left_det = left_track.detections[frame]
                right_det = right_track.detections[frame]
                disp_stats[seq_group(seq_name)].append(left_det.cx - right_det.cx)
                v_stats[seq_group(seq_name)].append(abs(left_det.cy - right_det.cy))

    priors: dict[str, GroupPrior] = {}
    for group_name, disparities in disp_stats.items():
        disp_arr = np.asarray(disparities, dtype=np.float64)
        v_arr = np.asarray(v_stats[group_name], dtype=np.float64)
        priors[group_name] = GroupPrior(
            disp_median=float(np.median(disp_arr)),
            disp_iqr=float(np.percentile(disp_arr, 75) - np.percentile(disp_arr, 25)),
            disp_p05=float(np.percentile(disp_arr, 5)),
            disp_p95=float(np.percentile(disp_arr, 95)),
            v_median=float(np.median(v_arr)),
            v_p90=float(np.percentile(v_arr, 90)),
            v_p95=float(np.percentile(v_arr, 95)),
        )
    return priors


def build_sequence_context(seq_name: str, priors: dict[str, GroupPrior]) -> SequenceContext:
    seqinfo = base.read_seqinfo(seq_name)
    split = base.locate_split(seq_name)
    width = int(seqinfo["imWidth"])
    height = int(seqinfo["imHeight"])
    seq_length = int(seqinfo["seqLength"])

    left_tracks, left_frame_index = load_mot_file(
        track_root_for_sequence(seq_name, "left") / f"{seq_name}.txt"
    )
    right_tracks, right_frame_index = load_mot_file(
        track_root_for_sequence(seq_name, "right") / f"{seq_name}.txt"
    )
    set_track_view(left_tracks, "left")
    set_track_view(right_tracks, "right")

    left_gt_tracks, left_gt_frame_index = load_rectified_gt(seq_name, "left")
    right_gt_tracks, right_gt_frame_index = load_rectified_gt(seq_name, "right")
    _ = left_gt_tracks, right_gt_tracks

    assign_dominant_gt(left_tracks, left_gt_frame_index)
    assign_dominant_gt(right_tracks, right_gt_frame_index)

    return SequenceContext(
        seq_name=seq_name,
        split=split,
        width=width,
        height=height,
        seq_length=seq_length,
        left_tracks=left_tracks,
        right_tracks=right_tracks,
        left_frame_index=left_frame_index,
        right_frame_index=right_frame_index,
        left_gt_frame_index=left_gt_frame_index,
        right_gt_frame_index=right_gt_frame_index,
        left_rank_map=build_rank_map(left_frame_index),
        right_rank_map=build_rank_map(right_frame_index),
        roi_left=load_roi(seq_name, "left"),
        roi_right=load_roi(seq_name, "right"),
        group_prior=priors[seq_group(seq_name)],
    )


def compute_local_mutual_score(
    ctx: SequenceContext,
    left_track_id: int,
    right_track_id: int,
    overlap_frames: list[int],
    v_gate: float,
    disp_center: float,
    disp_gate: float,
) -> float:
    mutual_hits = 0
    valid = 0
    for frame in overlap_frames:
        left_det = ctx.left_tracks[left_track_id].detections[frame]
        right_det = ctx.right_tracks[right_track_id].detections[frame]
        right_candidates = []
        for det in ctx.right_frame_index.get(frame, []):
            cost = abs(left_det.cy - det.cy) / max(v_gate, 1e-6)
            cost += abs((left_det.cx - det.cx) - disp_center) / max(disp_gate, 1e-6)
            right_candidates.append((cost, det.track_id))
        left_candidates = []
        for det in ctx.left_frame_index.get(frame, []):
            cost = abs(det.cy - right_det.cy) / max(v_gate, 1e-6)
            cost += abs((det.cx - right_det.cx) - disp_center) / max(disp_gate, 1e-6)
            left_candidates.append((cost, det.track_id))
        if not right_candidates or not left_candidates:
            continue
        valid += 1
        best_right = min(right_candidates)[1]
        best_left = min(left_candidates)[1]
        if best_right == right_track_id and best_left == left_track_id:
            mutual_hits += 1
    if valid == 0:
        return 0.0
    return mutual_hits / valid


def compute_pair_feature(ctx: SequenceContext, left_track: Track, right_track: Track) -> PairFeature | None:
    overlap_frames = left_track.overlap_frames(right_track)
    if len(overlap_frames) < 3:
        return None

    left_u = np.asarray([left_track.detections[f].cx for f in overlap_frames], dtype=np.float64)
    left_v = np.asarray([left_track.detections[f].cy for f in overlap_frames], dtype=np.float64)
    left_h = np.asarray([left_track.detections[f].h for f in overlap_frames], dtype=np.float64)
    right_u = np.asarray([right_track.detections[f].cx for f in overlap_frames], dtype=np.float64)
    right_v = np.asarray([right_track.detections[f].cy for f in overlap_frames], dtype=np.float64)
    right_h = np.asarray([right_track.detections[f].h for f in overlap_frames], dtype=np.float64)

    prior = ctx.group_prior
    vdiff = np.abs(left_v - right_v)
    disparity = left_u - right_u

    temporal_score = clamp01(len(overlap_frames) / max(1, min(left_track.length, right_track.length)))
    epi_scale = max(25.0, prior.v_p90 * 0.75)
    epi_score = exp_score(float(np.mean(vdiff)), epi_scale)

    disp_dev = abs(float(np.median(disparity)) - prior.disp_median)
    disp_spread = float(np.std(disparity))
    disp_center_scale = max(30.0, prior.disp_iqr * 1.8)
    disp_spread_scale = max(24.0, prior.disp_iqr * 1.25)
    disp_center_score = exp_score(disp_dev, disp_center_scale)
    disp_stability_score = exp_score(disp_spread, disp_spread_scale)
    size_score = exp_score(float(np.median(np.abs(np.log((left_h + 1e-6) / (right_h + 1e-6))))), 0.35)
    spatial_score = clamp01((disp_center_score * disp_stability_score * size_score) ** (1.0 / 3.0))

    if len(overlap_frames) >= 4:
        du_left = np.diff(left_u)
        dv_left = np.diff(left_v)
        du_right = np.diff(right_u)
        dv_right = np.diff(right_v)
        corr_u = 0.5 * (cosine_similarity(du_left, du_right) + 1.0)
        corr_v = 0.5 * (cosine_similarity(dv_left, dv_right) + 1.0)
        mag_gap = safe_mean(np.abs((du_left - du_right)).tolist() + np.abs((dv_left - dv_right)).tolist(), 0.0)
        motion_score = clamp01(0.4 * corr_u + 0.3 * corr_v + 0.3 * exp_score(mag_gap, 40.0))
    else:
        motion_score = 0.5

    rank_diffs = []
    valid_rank = 0
    for frame in overlap_frames:
        if frame not in ctx.left_rank_map or frame not in ctx.right_rank_map:
            continue
        left_rank_info = ctx.left_rank_map[frame].get(left_track.track_id)
        right_rank_info = ctx.right_rank_map[frame].get(right_track.track_id)
        if left_rank_info is None or right_rank_info is None:
            continue
        left_rank, left_count = left_rank_info
        right_rank, right_count = right_rank_info
        left_norm = left_rank / max(1, left_count - 1)
        right_norm = right_rank / max(1, right_count - 1)
        rank_diffs.append(abs(left_norm - right_norm))
        valid_rank += 1
    rank_score = exp_score(safe_mean(rank_diffs, 1.0), 0.18) if valid_rank else 0.0

    local_score = compute_local_mutual_score(
        ctx,
        left_track.track_id,
        right_track.track_id,
        overlap_frames,
        v_gate=max(40.0, prior.v_p95 * 1.15),
        disp_center=prior.disp_median,
        disp_gate=max(60.0, prior.disp_iqr * 2.2),
    )

    roi_hits = 0
    for frame in overlap_frames:
        if inside_roi(left_track.detections[frame], ctx.roi_left) and inside_roi(right_track.detections[frame], ctx.roi_right):
            roi_hits += 1
    region_score = roi_hits / len(overlap_frames)

    gt_left = left_track.dominant_gt_id
    gt_right = right_track.dominant_gt_id
    train_positive = bool(gt_left is not None and gt_right is not None and gt_left == gt_right)

    if float(np.mean(vdiff)) > max(170.0, prior.v_p95 * 1.5):
        return None

    return PairFeature(
        seq_name=ctx.seq_name,
        left_id=left_track.track_id,
        right_id=right_track.track_id,
        overlap=len(overlap_frames),
        temporal_score=temporal_score,
        epi_score=epi_score,
        spatial_score=spatial_score,
        motion_score=motion_score,
        rank_score=rank_score,
        local_score=local_score,
        region_score=region_score,
        mean_vdiff=float(np.mean(vdiff)),
        disp_median=float(np.median(disparity)),
        disp_std=float(np.std(disparity)),
        train_positive=train_positive,
        train_gt_id=gt_left if train_positive else None,
    )


def precompute_pair_features(ctx: SequenceContext) -> None:
    features: list[PairFeature] = []
    for left_track in ctx.left_tracks.values():
        for right_track in ctx.right_tracks.values():
            feature = compute_pair_feature(ctx, left_track, right_track)
            if feature is not None:
                features.append(feature)
    ctx.pair_features = features


def feature_score(method: str, feature: PairFeature) -> float:
    if method == "Epipolar":
        return clamp01(0.35 * feature.temporal_score + 0.65 * feature.epi_score)
    if method == "EpiSpatial":
        return clamp01(0.20 * feature.temporal_score + 0.45 * feature.epi_score + 0.35 * feature.spatial_score)
    if method == "EpiMotion":
        return clamp01(
            0.15 * feature.temporal_score
            + 0.30 * feature.epi_score
            + 0.25 * feature.spatial_score
            + 0.30 * feature.motion_score
        )
    if method == "EpiRank":
        return clamp01(
            0.12 * feature.temporal_score
            + 0.25 * feature.epi_score
            + 0.18 * feature.spatial_score
            + 0.20 * feature.rank_score
            + 0.25 * feature.local_score
        )
    if method == "Proposed":
        return clamp01(
            0.12 * feature.temporal_score
            + 0.20 * feature.epi_score
            + 0.16 * feature.spatial_score
            + 0.14 * feature.motion_score
            + 0.14 * feature.rank_score
            + 0.14 * feature.local_score
            + 0.10 * feature.region_score
        )
    raise ValueError(f"Unsupported feature-based method: {method}")


def frame_vote_scores(ctx: SequenceContext) -> dict[tuple[int, int], float]:
    prior = ctx.group_prior
    v_gate = max(40.0, prior.v_p95 * 1.15)
    disp_gate = max(60.0, prior.disp_iqr * 2.25)
    vote_sums: defaultdict[tuple[int, int], float] = defaultdict(float)
    vote_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    all_frames = sorted(set(ctx.left_frame_index) | set(ctx.right_frame_index))
    for frame in all_frames:
        left_dets = ctx.left_frame_index.get(frame, [])
        right_dets = ctx.right_frame_index.get(frame, [])
        if not left_dets or not right_dets:
            continue
        cost = np.full((len(left_dets), len(right_dets)), 9.0, dtype=np.float64)
        for i, left_det in enumerate(left_dets):
            for j, right_det in enumerate(right_dets):
                vdiff = abs(left_det.cy - right_det.cy)
                disp_dev = abs((left_det.cx - right_det.cx) - prior.disp_median)
                if vdiff > v_gate * 1.8 or disp_dev > disp_gate * 1.8:
                    continue
                cost[i, j] = 0.6 * (vdiff / max(v_gate, 1e-6)) + 0.4 * (disp_dev / max(disp_gate, 1e-6))
        row_idx, col_idx = linear_sum_assignment(cost)
        for i, j in zip(row_idx.tolist(), col_idx.tolist()):
            value = cost[i, j]
            if value > 1.0:
                continue
            pair = (left_dets[i].track_id, right_dets[j].track_id)
            vote_sums[pair] += 1.0 / (1.0 + value)
            vote_counts[pair] += 1

    pair_scores: dict[tuple[int, int], float] = {}
    for pair, vote_sum in vote_sums.items():
        left_len = ctx.left_tracks[pair[0]].length
        right_len = ctx.right_tracks[pair[1]].length
        norm = math.sqrt(max(1, left_len) * max(1, right_len))
        score = vote_sum / max(1.0, norm / 12.0)
        pair_scores[pair] = clamp01(score)
    return pair_scores


def hungarian_pairs(
    left_ids: list[int],
    right_ids: list[int],
    score_lookup: dict[tuple[int, int], float],
    threshold: float,
) -> list[tuple[int, int, float]]:
    if not left_ids or not right_ids:
        return []
    score_matrix = np.zeros((len(left_ids), len(right_ids)), dtype=np.float64)
    for i, left_id in enumerate(left_ids):
        for j, right_id in enumerate(right_ids):
            score_matrix[i, j] = score_lookup.get((left_id, right_id), 0.0)
    cost = 1.0 - score_matrix
    row_idx, col_idx = linear_sum_assignment(cost)
    pairs: list[tuple[int, int, float]] = []
    for i, j in zip(row_idx.tolist(), col_idx.tolist()):
        score = float(score_matrix[i, j])
        if score >= threshold:
            pairs.append((left_ids[i], right_ids[j], score))
    return pairs


def build_score_lookup(ctx: SequenceContext, method: str) -> dict[tuple[int, int], float]:
    if method == "FrameVote":
        if ctx.frame_vote_lookup is None:
            ctx.frame_vote_lookup = frame_vote_scores(ctx)
        return ctx.frame_vote_lookup
    if method == "Proposed":
        if ctx.frame_vote_lookup is None:
            ctx.frame_vote_lookup = frame_vote_scores(ctx)
        feature_lookup = {(feature.left_id, feature.right_id): feature for feature in ctx.pair_features}
        score_lookup: dict[tuple[int, int], float] = {}
        for pair, feature in feature_lookup.items():
            frame_vote = ctx.frame_vote_lookup.get(pair, 0.0)
            support_score = clamp01(
                0.18 * feature.temporal_score
                + 0.20 * feature.epi_score
                + 0.15 * feature.spatial_score
                + 0.13 * feature.motion_score
                + 0.12 * feature.rank_score
                + 0.12 * feature.local_score
                + 0.10 * feature.region_score
            )
            score_lookup[pair] = clamp01(
                0.55 * frame_vote
                + 0.35 * support_score
                + 0.10 * math.sqrt(max(0.0, frame_vote * feature.local_score))
            )
        return score_lookup
    return {(feature.left_id, feature.right_id): feature_score(method, feature) for feature in ctx.pair_features}


def time_overlap_count(track_a: Track, track_b: Track) -> int:
    return len(track_a.overlap_frames(track_b))


def proposed_components(
    ctx: SequenceContext,
    score_lookup: dict[tuple[int, int], float],
    primary_threshold: float,
    secondary_threshold: float,
) -> list[dict[str, set[int]]]:
    left_ids = sorted(ctx.left_tracks)
    right_ids = sorted(ctx.right_tracks)
    if ctx.frame_vote_lookup is None:
        ctx.frame_vote_lookup = frame_vote_scores(ctx)
    primary_pairs = hungarian_pairs(left_ids, right_ids, ctx.frame_vote_lookup, primary_threshold)
    feature_lookup = {(feature.left_id, feature.right_id): feature for feature in ctx.pair_features}

    components: list[dict[str, set[int]]] = []
    left_to_comp: dict[int, int] = {}
    right_to_comp: dict[int, int] = {}

    left_best_scores: dict[int, float] = defaultdict(float)
    right_best_scores: dict[int, float] = defaultdict(float)
    for (left_id, right_id), score in score_lookup.items():
        left_best_scores[left_id] = max(left_best_scores[left_id], score)
        right_best_scores[right_id] = max(right_best_scores[right_id], score)

    def add_component(left_id: int, right_id: int) -> None:
        comp_idx = len(components)
        components.append({"left": {left_id}, "right": {right_id}})
        left_to_comp[left_id] = comp_idx
        right_to_comp[right_id] = comp_idx

    def view_consistent(view: str, track_ids: set[int]) -> bool:
        ids = sorted(track_ids)
        track_dict = ctx.left_tracks if view == "left" else ctx.right_tracks
        for i, track_id_a in enumerate(ids):
            for track_id_b in ids[i + 1 :]:
                if time_overlap_count(track_dict[track_id_a], track_dict[track_id_b]) > 2:
                    return False
        return True

    def component_consistent(left_ids: set[int], right_ids: set[int]) -> bool:
        return view_consistent("left", left_ids) and view_consistent("right", right_ids)

    def merge_components(dst_idx: int, src_idx: int) -> None:
        if dst_idx == src_idx:
            return
        dst = components[dst_idx]
        src = components[src_idx]
        for left_id in src["left"]:
            left_to_comp[left_id] = dst_idx
        for right_id in src["right"]:
            right_to_comp[right_id] = dst_idx
        dst["left"].update(src["left"])
        dst["right"].update(src["right"])
        src["left"].clear()
        src["right"].clear()

    def secondary_edge_allowed(left_id: int, right_id: int, score: float) -> bool:
        feature = feature_lookup.get((left_id, right_id))
        if feature is None:
            return False

        left_best = left_best_scores.get(left_id, 0.0)
        right_best = right_best_scores.get(right_id, 0.0)
        near_left_best = score >= max(secondary_threshold, left_best - 0.04)
        near_right_best = score >= max(secondary_threshold, right_best - 0.04)

        if not (near_left_best and near_right_best):
            return False
        if feature.temporal_score < 0.12:
            return False
        if feature.epi_score < 0.38 or feature.spatial_score < 0.28:
            return False
        if feature.local_score < 0.20 and feature.rank_score < 0.30:
            return False
        if ctx.frame_vote_lookup.get((left_id, right_id), 0.0) < max(0.16, primary_threshold - 0.18):
            return False
        return True

    for left_id, right_id, _score in primary_pairs:
        add_component(left_id, right_id)

    candidate_edges = sorted(
        ((score, left_id, right_id) for (left_id, right_id), score in score_lookup.items() if score >= secondary_threshold),
        reverse=True,
    )
    for score, left_id, right_id in candidate_edges:
        if not secondary_edge_allowed(left_id, right_id, score):
            continue
        left_comp = left_to_comp.get(left_id)
        right_comp = right_to_comp.get(right_id)
        if left_comp is None and right_comp is None:
            add_component(left_id, right_id)
            continue
        if left_comp is not None and right_comp is None:
            comp = components[left_comp]
            if component_consistent(set(comp["left"]), set(comp["right"]) | {right_id}):
                comp["right"].add(right_id)
                right_to_comp[right_id] = left_comp
            continue
        if left_comp is None and right_comp is not None:
            comp = components[right_comp]
            if component_consistent(set(comp["left"]) | {left_id}, set(comp["right"])):
                comp["left"].add(left_id)
                left_to_comp[left_id] = right_comp
            continue
        if left_comp is not None and right_comp is not None and left_comp != right_comp:
            comp_left = components[left_comp]
            comp_right = components[right_comp]
            merged_left = set(comp_left["left"]) | set(comp_right["left"])
            merged_right = set(comp_left["right"]) | set(comp_right["right"])
            if component_consistent(merged_left, merged_right):
                merge_components(left_comp, right_comp)

    return [comp for comp in components if comp["left"] or comp["right"]]


def method_components(ctx: SequenceContext, method: str, threshold: float, secondary_threshold: float) -> list[dict[str, set[int]]]:
    score_lookup = build_score_lookup(ctx, method)
    if method == "Proposed":
        return proposed_components(ctx, score_lookup, primary_threshold=threshold, secondary_threshold=secondary_threshold)

    pairs = hungarian_pairs(sorted(ctx.left_tracks), sorted(ctx.right_tracks), score_lookup, threshold)
    components = [{"left": {left_id}, "right": {right_id}} for left_id, right_id, _score in pairs]
    return components


def predicted_pairs_from_components(components: list[dict[str, set[int]]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for comp in components:
        for left_id in comp["left"]:
            for right_id in comp["right"]:
                pairs.add((left_id, right_id))
    return pairs


def ground_truth_pairs(ctx: SequenceContext) -> set[tuple[int, int]]:
    gt_pairs: set[tuple[int, int]] = set()
    gt_to_left: defaultdict[int, set[int]] = defaultdict(set)
    gt_to_right: defaultdict[int, set[int]] = defaultdict(set)
    for track in ctx.left_tracks.values():
        if track.dominant_gt_id is not None:
            gt_to_left[track.dominant_gt_id].add(track.track_id)
    for track in ctx.right_tracks.values():
        if track.dominant_gt_id is not None:
            gt_to_right[track.dominant_gt_id].add(track.track_id)
    for gt_id in sorted(set(gt_to_left) & set(gt_to_right)):
        for left_id in gt_to_left[gt_id]:
            for right_id in gt_to_right[gt_id]:
                gt_pairs.add((left_id, right_id))
    return gt_pairs


def pair_prf(pred_pairs: set[tuple[int, int]], gt_pairs: set[tuple[int, int]]) -> tuple[float, float, float, int, int, int]:
    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1, tp, fp, fn


def beta_f_score(precision: float, recall: float, beta: float) -> float:
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom <= 1e-12:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def tune_method(
    train_contexts: list[SequenceContext],
    method: str,
) -> dict[str, float]:
    threshold_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    secondary_grid = [0.12, 0.16, 0.20, 0.24, 0.28] if method == "Proposed" else [0.0]

    best = {
        "threshold": threshold_grid[0],
        "secondary_threshold": secondary_grid[0],
        "pair_precision": 0.0,
        "pair_recall": 0.0,
        "pair_f1": -1.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    best_objective = -1.0
    # Use one precision-oriented objective for every method so the comparison
    # differs only in the association strategy, not in operating-point policy.
    beta = 0.5

    for threshold in threshold_grid:
        for secondary_threshold in secondary_grid:
            all_pred_pairs: set[tuple[str, int, int]] = set()
            all_gt_pairs: set[tuple[str, int, int]] = set()
            for ctx in train_contexts:
                components = method_components(ctx, method, threshold, secondary_threshold)
                pred_pairs = predicted_pairs_from_components(components)
                gt_pairs = ground_truth_pairs(ctx)
                all_pred_pairs.update((ctx.seq_name, left_id, right_id) for left_id, right_id in pred_pairs)
                all_gt_pairs.update((ctx.seq_name, left_id, right_id) for left_id, right_id in gt_pairs)
            precision, recall, f1, tp, fp, fn = pair_prf(all_pred_pairs, all_gt_pairs)
            objective = beta_f_score(precision, recall, beta)
            if objective > best_objective + 1e-9 or (
                abs(objective - best_objective) <= 1e-9 and precision > best["pair_precision"] + 1e-9
            ):
                best_objective = objective
                best = {
                    "threshold": threshold,
                    "secondary_threshold": secondary_threshold,
                    "pair_precision": precision,
                    "pair_recall": recall,
                    "pair_f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
    return best


def evaluate_pair_threshold(
    contexts: list[SequenceContext],
    method: str,
    threshold: float,
    secondary_threshold: float,
) -> dict[str, float]:
    all_pred_pairs: set[tuple[str, int, int]] = set()
    all_gt_pairs: set[tuple[str, int, int]] = set()
    for ctx in contexts:
        components = method_components(ctx, method, threshold, secondary_threshold)
        pred_pairs = predicted_pairs_from_components(components)
        gt_pairs = ground_truth_pairs(ctx)
        all_pred_pairs.update(
            (ctx.seq_name, left_id, right_id) for left_id, right_id in pred_pairs
        )
        all_gt_pairs.update(
            (ctx.seq_name, left_id, right_id) for left_id, right_id in gt_pairs
        )
    precision, recall, f1, tp, fp, fn = pair_prf(all_pred_pairs, all_gt_pairs)
    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def write_seqinfo_fused(
    dst_path: Path,
    seq_name: str,
    width: int,
    height: int,
    seq_length: int,
    ext: str,
    source_frame_rate: int,
) -> None:
    content = "\n".join(
        [
            "[Sequence]",
            f"name={seq_name}",
            "imDir=img1",
            f"frameRate={source_frame_rate * 2}",
            f"seqLength={seq_length * 2}",
            f"imWidth={width}",
            f"imHeight={height}",
            f"imExt={ext}",
            "",
        ]
    )
    dst_path.write_text(content, encoding="utf-8")


def save_mot_rows(rows: list[list[object]], dst_path: Path) -> None:
    ensure_dir(dst_path.parent)
    with open(dst_path, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")


def fuse_gt_sequence(seq_name: str, dst_gt_path: Path) -> tuple[int, int, int, str]:
    seqinfo = base.read_seqinfo(seq_name)
    width = int(seqinfo["imWidth"])
    height = int(seqinfo["imHeight"])
    seq_length = int(seqinfo["seqLength"])
    ext = seqinfo["imExt"]

    left_lines = (
        gt_root_for_sequence(seq_name, "left") / seq_name / "gt" / "gt.txt"
    ).read_text(encoding="utf-8").splitlines()
    right_lines = (
        gt_root_for_sequence(seq_name, "right") / seq_name / "gt" / "gt.txt"
    ).read_text(encoding="utf-8").splitlines()
    rows = []
    for line in left_lines:
        if not line.strip():
            continue
        frame, track_id, x, y, w, h, mark, cls, vis = line.split(",")[:9]
        fused_frame = 2 * int(float(frame)) - 1
        rows.append([fused_frame, track_id, x, y, w, h, mark, cls, vis])
    for line in right_lines:
        if not line.strip():
            continue
        frame, track_id, x, y, w, h, mark, cls, vis = line.split(",")[:9]
        fused_frame = 2 * int(float(frame))
        rows.append(
            [
                fused_frame,
                track_id,
                x,
                y,
                w,
                h,
                mark,
                cls,
                vis,
            ]
        )
    rows.sort(key=lambda row: (int(float(row[0])), int(float(row[1])), float(row[2])))
    ensure_dir(dst_gt_path.parent)
    with open(dst_gt_path, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    return width, height, seq_length, ext


def build_fused_gt_dataset(save_root: Path, dataset_name: str, sequences: list[str]) -> None:
    gt_root = save_root / "gt" / "mot_challenge"
    dataset_root = gt_root / dataset_name
    ensure_dir(dataset_root)
    ensure_dir(gt_root / "seqmaps")

    seqmap_path = gt_root / "seqmaps" / f"{dataset_name}.txt"
    with open(seqmap_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for seq_name in sequences:
            writer.writerow([seq_name])

    for seq_name in sequences:
        seq_root = dataset_root / seq_name
        ensure_dir(seq_root / "gt")
        width, height, seq_length, ext = fuse_gt_sequence(seq_name, seq_root / "gt" / "gt.txt")
        source_frame_rate = int(base.read_seqinfo(seq_name)["frameRate"])
        write_seqinfo_fused(
            seq_root / "seqinfo.ini",
            seq_name,
            width,
            height,
            seq_length,
            ext,
            source_frame_rate,
        )


def rows_from_components(ctx: SequenceContext, components: list[dict[str, set[int]]]) -> list[list[object]]:
    rows: list[list[object]] = []
    next_global_id = 1
    desired_left: dict[int, int] = {}
    desired_right: dict[int, int] = {}

    for comp in components:
        if not comp["left"] and not comp["right"]:
            continue
        global_id = next_global_id
        next_global_id += 1
        for left_id in comp["left"]:
            desired_left[left_id] = global_id
        for right_id in comp["right"]:
            desired_right[right_id] = global_id

    for left_id in sorted(ctx.left_tracks):
        if left_id not in desired_left:
            desired_left[left_id] = next_global_id
            next_global_id += 1
    for right_id in sorted(ctx.right_tracks):
        if right_id not in desired_right:
            desired_right[right_id] = next_global_id
            next_global_id += 1

    frame_usage: defaultdict[int, set[int]] = defaultdict(set)
    final_left: dict[int, int] = {}
    final_right: dict[int, int] = {}

    def assign_legal_ids(
        tracks: dict[int, Track],
        desired_map: dict[int, int],
        target_map: dict[int, int],
        frame_fn,
    ) -> None:
        nonlocal next_global_id
        ordered_tracks = sorted(tracks.values(), key=lambda track: (-track.length, track.track_id))
        for track in ordered_tracks:
            wanted_id = desired_map[track.track_id]
            fused_frames = [frame_fn(frame) for frame in track.frames]
            chosen_id = wanted_id
            if any(chosen_id in frame_usage[fused_frame] for fused_frame in fused_frames):
                chosen_id = next_global_id
                next_global_id += 1
            target_map[track.track_id] = chosen_id
            for fused_frame in fused_frames:
                frame_usage[fused_frame].add(chosen_id)

    assign_legal_ids(ctx.left_tracks, desired_left, final_left, lambda frame: 2 * int(frame) - 1)
    assign_legal_ids(ctx.right_tracks, desired_right, final_right, lambda frame: 2 * int(frame))

    for track in ctx.left_tracks.values():
        global_id = final_left[track.track_id]
        for frame in track.frames:
            det = track.detections[frame]
            fused_frame = 2 * int(frame) - 1
            rows.append(
                [
                    fused_frame,
                    global_id,
                    f"{det.x:.2f}",
                    f"{det.y:.2f}",
                    f"{det.w:.2f}",
                    f"{det.h:.2f}",
                    f"{det.score:.6f}",
                    -1,
                    -1,
                    -1,
                ]
            )
    for track in ctx.right_tracks.values():
        global_id = final_right[track.track_id]
        for frame in track.frames:
            det = track.detections[frame]
            fused_frame = 2 * int(frame)
            rows.append(
                [
                    fused_frame,
                    global_id,
                    f"{det.x:.2f}",
                    f"{det.y:.2f}",
                    f"{det.w:.2f}",
                    f"{det.h:.2f}",
                    f"{det.score:.6f}",
                    -1,
                    -1,
                    -1,
                ]
            )
    rows.sort(key=lambda row: (int(row[0]), int(row[1]), float(row[2])))
    return rows


def export_method_predictions(
    save_root: Path,
    dataset_name: str,
    contexts: list[SequenceContext],
    method: str,
    threshold: float,
    secondary_threshold: float,
) -> None:
    tracker_root = save_root / "trackers" / "mot_challenge" / dataset_name / method / "data"
    ensure_dir(tracker_root)
    for ctx in contexts:
        components = method_components(ctx, method, threshold, secondary_threshold)
        rows = rows_from_components(ctx, components)
        save_mot_rows(rows, tracker_root / f"{ctx.seq_name}.txt")


def run_trackeval(
    save_root: Path,
    dataset_name: str,
    sequences: list[str],
    methods: list[str],
) -> Path:
    gt_root = save_root / "gt" / "mot_challenge"
    tracker_root = save_root / "trackers" / "mot_challenge"
    output_folder = save_root / "eval" / dataset_name
    ensure_dir(output_folder)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_CONFIG"] = False
    eval_config["TIME_PROGRESS"] = False
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["LOG_ON_ERROR"] = str(save_root / "trackeval_error_log.txt")
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_DETAILED"] = True
    eval_config["PLOT_CURVES"] = False
    eval_config["PRINT_RESULTS"] = False
    eval_config["PRINT_ONLY_COMBINED"] = True

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(gt_root)
    dataset_config["TRACKERS_FOLDER"] = str(tracker_root)
    dataset_config["OUTPUT_FOLDER"] = str(output_folder)
    dataset_config["BENCHMARK"] = dataset_name.replace("-all", "")
    dataset_config["SPLIT_TO_EVAL"] = "all"
    dataset_config["SEQMAP_FOLDER"] = str(gt_root / "seqmaps")
    dataset_config["TRACKERS_TO_EVAL"] = methods
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
    evaluator.evaluate([dataset], metrics_list, show_progressbar=False)
    return output_folder


def extract_summary_metrics(summary_path: Path) -> dict[str, float]:
    lines = [line.strip() for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = lines[0].split()
    values = lines[1].split()
    metrics = dict(zip(header, values))
    return {
        "HOTA": float(metrics["HOTA"]),
        "AssA": float(metrics["AssA"]),
        "IDF1": float(metrics["IDF1"]),
        "MOTA": float(metrics["MOTA"]),
        "IDSW": float(metrics["IDSW"]),
    }


def write_reports(
    report_root: Path,
    tuning_rows: list[dict[str, object]],
    metrics_rows: list[dict[str, object]],
    selected_params: dict[str, dict[str, float]],
) -> None:
    ensure_dir(report_root)

    tuning_csv = report_root / "association_tuning_summary.csv"
    with open(tuning_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "threshold",
                "secondary_threshold",
                "pair_precision",
                "pair_recall",
                "pair_f1",
                "tp",
                "fp",
                "fn",
            ],
        )
        writer.writeheader()
        writer.writerows(tuning_rows)

    long_csv = report_root / "association_metrics_long.csv"
    with open(long_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "method", "HOTA", "AssA", "IDF1", "MOTA", "IDSW"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    wide_csv = report_root / "association_metrics_wide.csv"
    wide_fieldnames = [
        "method",
        "train_HOTA",
        "train_AssA",
        "train_IDF1",
        "train_MOTA",
        "train_IDSW",
        "test_HOTA",
        "test_AssA",
        "test_IDF1",
        "test_MOTA",
        "test_IDSW",
    ]
    wide_rows: dict[str, dict[str, object]] = {}
    for row in metrics_rows:
        method = str(row["method"])
        wide_rows.setdefault(method, {"method": method})
        prefix = str(row["split"])
        for key in ["HOTA", "AssA", "IDF1", "MOTA", "IDSW"]:
            wide_rows[method][f"{prefix}_{key}"] = row[key]
    with open(wide_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=wide_fieldnames)
        writer.writeheader()
        for method in selected_params:
            if method in wide_rows:
                writer.writerow(wide_rows[method])

    paper_md = report_root / "association_metrics_paper.md"
    with open(paper_md, "w", encoding="utf-8") as f:
        f.write("## Test Split Stereo Association Benchmark\n\n")
        f.write("| Method | HOTA | AssA | IDF1 | MOTA | IDSW |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in [r for r in metrics_rows if r["split"] == "test"]:
            f.write(
                f"| {row['method']} | {row['HOTA']:.3f} | {row['AssA']:.3f} | "
                f"{row['IDF1']:.3f} | {row['MOTA']:.3f} | {int(row['IDSW'])} |\n"
            )
        f.write("\n## Train-Split Threshold Selection\n\n")
        f.write("| Method | Threshold | Secondary Threshold | Pair Precision | Pair Recall | Pair F1 |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in tuning_rows:
            f.write(
                f"| {row['method']} | {float(row['threshold']):.2f} | {float(row['secondary_threshold']):.2f} | "
                f"{float(row['pair_precision']):.3f} | {float(row['pair_recall']):.3f} | {float(row['pair_f1']):.3f} |\n"
            )

    paper_tex = report_root / "association_metrics_paper.tex"
    with open(paper_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & HOTA & AssA & IDF1 & MOTA & IDSW \\\\\n")
        f.write("\\midrule\n")
        for row in [r for r in metrics_rows if r["split"] == "test"]:
            f.write(
                f"{row['method']} & {row['HOTA']:.3f} & {row['AssA']:.3f} & "
                f"{row['IDF1']:.3f} & {row['MOTA']:.3f} & {int(row['IDSW'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    summary_md = report_root / "association_summary.md"
    test_rows = [r for r in metrics_rows if r["split"] == "test"]
    best_hota = max(test_rows, key=lambda row: float(row["HOTA"]))
    best_idf1 = max(test_rows, key=lambda row: float(row["IDF1"]))
    best_assa = max(test_rows, key=lambda row: float(row["AssA"]))
    summary_lines = [
        "# Stereo Association Summary",
        "",
        "Main tracker input: `ByteTrack` tuned results from step3.",
        "",
        f"- Best `HOTA`: `{best_hota['method']}` = `{float(best_hota['HOTA']):.3f}`",
        f"- Best `AssA`: `{best_assa['method']}` = `{float(best_assa['AssA']):.3f}`",
        f"- Best `IDF1`: `{best_idf1['method']}` = `{float(best_idf1['IDF1']):.3f}`",
        "",
        "Selected thresholds:",
    ]
    for method in selected_params:
        params = selected_params[method]
        summary_lines.append(
            f"- `{method}`: threshold=`{params['threshold']:.2f}`, "
            f"secondary_threshold=`{params['secondary_threshold']:.2f}`"
        )
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    params_json = report_root / "selected_params.json"
    params_json.write_text(json.dumps(selected_params, ensure_ascii=False, indent=2), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step3-root", default=None)
    parser.add_argument("--step3-train-root", default=None)
    parser.add_argument("--step3-test-root", default=None)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--rect-root", default=str(RECT_ROOT))
    parser.add_argument("--save-root", default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument(
        "--fixed-params-json",
        default=None,
        help="Load previously selected parameters and skip all tuning.",
    )
    parser.add_argument("--skip-build-gt", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.step3_root:
        step3_train_root = Path(args.step3_root).resolve()
        step3_test_root = step3_train_root
    else:
        if not args.step3_train_root or not args.step3_test_root:
            parser.error("Provide --step3-root or both --step3-train-root and --step3-test-root")
        step3_train_root = Path(args.step3_train_root).resolve()
        step3_test_root = Path(args.step3_test_root).resolve()
    data_root = Path(args.data_root).resolve()
    rect_root = Path(args.rect_root).resolve()
    configure_input_roots(step3_train_root, step3_test_root, data_root, rect_root)
    save_root = Path(args.save_root).resolve()
    report_root = save_root / "reports"
    ensure_dir(save_root)
    ensure_dir(report_root)

    train_sequences = find_sequences("train")
    test_sequences = find_sequences("test")
    if not train_sequences or not test_sequences:
        raise FileNotFoundError("Both train and test splits are required for step4")

    priors = compute_group_priors(train_sequences)
    train_contexts = [build_sequence_context(seq_name, priors) for seq_name in train_sequences]
    test_contexts = [build_sequence_context(seq_name, priors) for seq_name in test_sequences]
    all_contexts = train_contexts + test_contexts
    for ctx in all_contexts:
        precompute_pair_features(ctx)

    selected_params: dict[str, dict[str, float]] = {}
    tuning_rows: list[dict[str, object]] = []
    if args.fixed_params_json:
        fixed_source = Path(args.fixed_params_json).resolve()
        fixed_payload = json.loads(fixed_source.read_text(encoding="utf-8"))
        fixed_params = fixed_payload.get("selected_params", fixed_payload)
        for method in args.methods:
            if method not in fixed_params:
                raise KeyError(f"Fixed parameters are missing method {method}: {fixed_source}")
            params = fixed_params[method]
            selected_params[method] = {
                "threshold": float(params["threshold"]),
                "secondary_threshold": float(params.get("secondary_threshold", 0.0)),
            }
            # Recompute descriptive train-pair counts at the frozen operating point.
            fixed_eval = evaluate_pair_threshold(
                train_contexts,
                method,
                selected_params[method]["threshold"],
                selected_params[method]["secondary_threshold"],
            )
            tuning_rows.append(
                {
                    "method": method,
                    "threshold": selected_params[method]["threshold"],
                    "secondary_threshold": selected_params[method]["secondary_threshold"],
                    **fixed_eval,
                }
            )
            print(f"[Fixed] {method}: {selected_params[method]}")
    else:
        for method in args.methods:
            print(f"[Tune] {method}")
            tuned = tune_method(train_contexts, method)
            selected_params[method] = {
                "threshold": float(tuned["threshold"]),
                "secondary_threshold": float(tuned["secondary_threshold"]),
            }
            tuning_rows.append({"method": method, **tuned})

    if not args.skip_build_gt:
        print("[Build] fused GT datasets")
        build_fused_gt_dataset(save_root, "CP3D-stereo-train-all", train_sequences)
        build_fused_gt_dataset(save_root, "CP3D-stereo-test-all", test_sequences)

    print("[Export] fused predictions")
    for method in args.methods:
        params = selected_params[method]
        export_method_predictions(
            save_root,
            "CP3D-stereo-train-all",
            train_contexts,
            method,
            params["threshold"],
            params["secondary_threshold"],
        )
        export_method_predictions(
            save_root,
            "CP3D-stereo-test-all",
            test_contexts,
            method,
            params["threshold"],
            params["secondary_threshold"],
        )

    metrics_rows: list[dict[str, object]] = []
    if not args.skip_eval:
        print("[Eval] stereo train")
        train_eval_root = run_trackeval(save_root, "CP3D-stereo-train-all", train_sequences, args.methods)
        print("[Eval] stereo test")
        test_eval_root = run_trackeval(save_root, "CP3D-stereo-test-all", test_sequences, args.methods)
        for split_name, eval_root in [("train", train_eval_root), ("test", test_eval_root)]:
            for method in args.methods:
                summary_path = eval_root / method / "pedestrian_summary.txt"
                metrics = extract_summary_metrics(summary_path)
                metrics_rows.append({"split": split_name, "method": method, **metrics})

    write_reports(report_root, tuning_rows, metrics_rows, selected_params)

    run_metadata = {
        "input_tracker": "ByteTrack",
        "step3_train_root": str(step3_train_root),
        "step3_test_root": str(step3_test_root),
        "data_root": str(data_root),
        "rect_root": str(rect_root),
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "methods": args.methods,
        "parameter_selection_mode": (
            "fixed_external_parameters" if args.fixed_params_json else "train_split_tuning"
        ),
        "fixed_params_json": (
            str(Path(args.fixed_params_json).resolve()) if args.fixed_params_json else None
        ),
        "selected_params": selected_params,
        "save_root": str(save_root),
        "reports": str(report_root),
        "metrics_rows": metrics_rows,
    }
    (save_root / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
