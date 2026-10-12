# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    tlwh: np.ndarray
    score: float


@dataclass
class TrackResult:
    track_id: int
    tlwh: np.ndarray
    score: float


def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = tlwh
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = tlbr
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)


def bbox_center(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = tlwh
    return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64)


def iou_tlwh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = tlwh_to_tlbr(a)
    bx1, by1, bx2, by2 = tlwh_to_tlbr(b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def iou_matrix(detections: Iterable[np.ndarray], tracks: Iterable[np.ndarray]) -> np.ndarray:
    detections = list(detections)
    tracks = list(tracks)
    if not detections or not tracks:
        return np.zeros((len(detections), len(tracks)), dtype=np.float64)

    matrix = np.zeros((len(detections), len(tracks)), dtype=np.float64)
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracks):
            matrix[i, j] = iou_tlwh(det, trk)
    return matrix


def assign_by_score(score_matrix: np.ndarray, min_score: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if score_matrix.size == 0:
        return [], list(range(score_matrix.shape[0])), list(range(score_matrix.shape[1]))

    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    unmatched_rows = set(range(score_matrix.shape[0]))
    unmatched_cols = set(range(score_matrix.shape[1]))

    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        if score_matrix[row, col] < min_score:
            continue
        matched.append((row, col))
        unmatched_rows.discard(row)
        unmatched_cols.discard(col)

    return matched, sorted(unmatched_rows), sorted(unmatched_cols)


def convert_tlwh_to_z(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = tlwh
    cx = x + w / 2.0
    cy = y + h / 2.0
    s = w * h
    r = w / max(h, 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float64).reshape((4, 1))


def convert_x_to_tlwh(x_state: np.ndarray) -> np.ndarray:
    cx, cy, s, r = x_state[:4].reshape(-1)
    w = np.sqrt(max(s * r, 0.0))
    h = s / max(w, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float64)


class KalmanBoxTrack:
    def __init__(self, tlwh: np.ndarray, score: float, track_id: int):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_tlwh_to_z(tlwh)

        self.track_id = track_id
        self.score = float(score)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.observations = deque(maxlen=5)
        self.last_tlwh = tlwh.astype(np.float64)
        self.velocity = None
        self.is_activated = True
        self.state = "tracked"

        self.observations.append(self.last_tlwh.copy())

    def predict(self) -> np.ndarray:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.last_tlwh = convert_x_to_tlwh(self.kf.x)
        return self.last_tlwh.copy()

    def update(self, tlwh: np.ndarray, score: float) -> None:
        prev_center = bbox_center(self.last_tlwh)
        new_center = bbox_center(tlwh)
        delta = new_center - prev_center
        norm = np.linalg.norm(delta)
        if norm > 1e-6:
            self.velocity = delta / norm

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)
        self.kf.update(convert_tlwh_to_z(tlwh))
        self.last_tlwh = tlwh.astype(np.float64)
        self.observations.append(self.last_tlwh.copy())
        self.state = "tracked"

    def mark_lost(self) -> None:
        self.state = "lost"

    def get_tlwh(self) -> np.ndarray:
        return convert_x_to_tlwh(self.kf.x)


class SortTracker:
    def __init__(self, iou_threshold: float = 0.1, max_age: int = 15, min_hits: int = 1, min_score: float = 0.3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_score = min_score
        self.frame_id = 0
        self.next_id = 1
        self.tracks: list[KalmanBoxTrack] = []

    def update(self, detections: list[Detection]) -> list[TrackResult]:
        self.frame_id += 1
        detections = [det for det in detections if det.score >= self.min_score]

        predicted_boxes = []
        valid_tracks = []
        for track in self.tracks:
            predicted = track.predict()
            if np.any(np.isnan(predicted)):
                continue
            predicted_boxes.append(predicted)
            valid_tracks.append(track)
        self.tracks = valid_tracks

        matches, unmatched_dets, unmatched_tracks = assign_by_score(
            iou_matrix([d.tlwh for d in detections], predicted_boxes),
            self.iou_threshold,
        )

        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx].tlwh, detections[det_idx].score)

        for trk_idx in unmatched_tracks:
            self.tracks[trk_idx].mark_lost()

        for det_idx in unmatched_dets:
            self.tracks.append(KalmanBoxTrack(detections[det_idx].tlwh, detections[det_idx].score, self.next_id))
            self.next_id += 1

        alive_tracks = []
        outputs = []
        for track in self.tracks:
            if track.time_since_update < 1 and (
                track.hit_streak >= self.min_hits or self.frame_id <= self.min_hits
            ):
                outputs.append(TrackResult(track.track_id, track.get_tlwh(), track.score))
            if track.time_since_update <= self.max_age:
                alive_tracks.append(track)
        self.tracks = alive_tracks
        return outputs


class IOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10, min_score: float = 0.3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_score = min_score
        self.next_id = 1
        self.tracks: list[dict] = []

    def update(self, detections: list[Detection]) -> list[TrackResult]:
        detections = [det for det in detections if det.score >= self.min_score]
        score_matrix = iou_matrix([d.tlwh for d in detections], [trk["tlwh"] for trk in self.tracks])
        matches, unmatched_dets, unmatched_tracks = assign_by_score(score_matrix, self.iou_threshold)

        outputs = []
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx]["tlwh"] = detections[det_idx].tlwh.copy()
            self.tracks[trk_idx]["score"] = detections[det_idx].score
            self.tracks[trk_idx]["age"] = 0
            outputs.append(
                TrackResult(
                    self.tracks[trk_idx]["track_id"],
                    self.tracks[trk_idx]["tlwh"].copy(),
                    self.tracks[trk_idx]["score"],
                )
            )

        for trk_idx in unmatched_tracks:
            self.tracks[trk_idx]["age"] += 1

        for det_idx in unmatched_dets:
            trk = {
                "track_id": self.next_id,
                "tlwh": detections[det_idx].tlwh.copy(),
                "score": detections[det_idx].score,
                "age": 0,
            }
            self.next_id += 1
            self.tracks.append(trk)
            outputs.append(TrackResult(trk["track_id"], trk["tlwh"].copy(), trk["score"]))

        self.tracks = [trk for trk in self.tracks if trk["age"] <= self.max_age]
        return outputs


class ByteTrackTracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        second_match_thresh: float = 0.2,
        track_buffer: int = 30,
        min_box_area: float = 4.0,
    ):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.next_id = 1
        self.frame_id = 0
        self.tracked_tracks: list[KalmanBoxTrack] = []
        self.lost_tracks: list[KalmanBoxTrack] = []

    def _new_track(self, det: Detection) -> KalmanBoxTrack:
        track = KalmanBoxTrack(det.tlwh, det.score, self.next_id)
        self.next_id += 1
        return track

    def _cleanup_lost(self) -> None:
        self.lost_tracks = [trk for trk in self.lost_tracks if trk.time_since_update <= self.track_buffer]

    def update(self, detections: list[Detection]) -> list[TrackResult]:
        self.frame_id += 1
        detections = [det for det in detections if det.tlwh[2] * det.tlwh[3] >= self.min_box_area]
        high_dets = [det for det in detections if det.score >= self.track_thresh]
        low_dets = [det for det in detections if self.low_thresh <= det.score < self.track_thresh]

        track_pool = self.tracked_tracks + self.lost_tracks
        for track in track_pool:
            track.predict()

        matches, unmatched_high, unmatched_tracks = assign_by_score(
            iou_matrix([det.tlwh for det in high_dets], [trk.get_tlwh() for trk in track_pool]),
            self.match_thresh,
        )

        activated_tracks = []
        refound_tracks = []
        remaining_track_pool = []
        matched_track_ids = set()

        for det_idx, trk_idx in matches:
            track = track_pool[trk_idx]
            track.update(high_dets[det_idx].tlwh, high_dets[det_idx].score)
            matched_track_ids.add(track.track_id)
            if track in self.lost_tracks:
                refound_tracks.append(track)
            else:
                activated_tracks.append(track)

        for trk_idx, track in enumerate(track_pool):
            if track.track_id not in matched_track_ids:
                remaining_track_pool.append(track)

        active_unmatched = [trk for trk in remaining_track_pool if trk in self.tracked_tracks]
        for track in active_unmatched:
            track.predict()

        matches_low, unmatched_low_dets, unmatched_active_tracks = assign_by_score(
            iou_matrix([det.tlwh for det in low_dets], [trk.get_tlwh() for trk in active_unmatched]),
            self.second_match_thresh,
        )

        matched_active_ids = set()
        for det_idx, trk_idx in matches_low:
            track = active_unmatched[trk_idx]
            track.update(low_dets[det_idx].tlwh, low_dets[det_idx].score)
            activated_tracks.append(track)
            matched_active_ids.add(track.track_id)

        newly_lost = []
        for trk_idx in unmatched_active_tracks:
            track = active_unmatched[trk_idx]
            if track.track_id not in matched_active_ids:
                track.mark_lost()
                newly_lost.append(track)

        for det_idx in unmatched_high:
            det = high_dets[det_idx]
            new_track = self._new_track(det)
            activated_tracks.append(new_track)

        tracked_id_set = {trk.track_id for trk in activated_tracks}
        self.tracked_tracks = [trk for trk in activated_tracks if trk.state == "tracked"]

        kept_lost = []
        for track in self.lost_tracks + newly_lost:
            if track.track_id not in tracked_id_set:
                kept_lost.append(track)
        for track in remaining_track_pool:
            if track.state == "lost" and track.track_id not in tracked_id_set:
                kept_lost.append(track)

        dedup = {}
        for track in kept_lost:
            dedup[track.track_id] = track
        self.lost_tracks = list(dedup.values())
        self._cleanup_lost()

        return [TrackResult(trk.track_id, trk.get_tlwh(), trk.score) for trk in self.tracked_tracks]


class OCSortTracker:
    def __init__(
        self,
        iou_threshold: float = 0.1,
        max_age: int = 15,
        min_hits: int = 1,
        min_score: float = 0.3,
        inertia_weight: float = 0.1,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_score = min_score
        self.inertia_weight = inertia_weight
        self.frame_id = 0
        self.next_id = 1
        self.tracks: list[KalmanBoxTrack] = []

    def _association_scores(self, detections: list[Detection], tracks: list[KalmanBoxTrack]) -> np.ndarray:
        if not detections or not tracks:
            return np.zeros((len(detections), len(tracks)), dtype=np.float64)

        scores = np.zeros((len(detections), len(tracks)), dtype=np.float64)
        for det_idx, det in enumerate(detections):
            det_center = bbox_center(det.tlwh)
            for trk_idx, track in enumerate(tracks):
                track_box = track.get_tlwh()
                score = iou_tlwh(det.tlwh, track_box)
                if track.velocity is not None:
                    motion = det_center - bbox_center(track_box)
                    norm = np.linalg.norm(motion)
                    if norm > 1e-6:
                        direction = motion / norm
                        score += self.inertia_weight * max(0.0, float(np.dot(direction, track.velocity)))
                scores[det_idx, trk_idx] = score
        return scores

    def update(self, detections: list[Detection]) -> list[TrackResult]:
        self.frame_id += 1
        detections = [det for det in detections if det.score >= self.min_score]

        predicted_tracks = []
        valid_tracks = []
        for track in self.tracks:
            predicted = track.predict()
            if np.any(np.isnan(predicted)):
                continue
            predicted_tracks.append(track)
            valid_tracks.append(track)
        self.tracks = valid_tracks

        scores = self._association_scores(detections, predicted_tracks)
        matches, unmatched_dets, unmatched_tracks = assign_by_score(scores, self.iou_threshold)

        for det_idx, trk_idx in matches:
            predicted_tracks[trk_idx].update(detections[det_idx].tlwh, detections[det_idx].score)

        for trk_idx in unmatched_tracks:
            predicted_tracks[trk_idx].mark_lost()

        for det_idx in unmatched_dets:
            self.tracks.append(KalmanBoxTrack(detections[det_idx].tlwh, detections[det_idx].score, self.next_id))
            self.next_id += 1

        alive_tracks = []
        outputs = []
        for track in self.tracks:
            if track.time_since_update < 1 and (
                track.hit_streak >= self.min_hits or self.frame_id <= self.min_hits
            ):
                outputs.append(TrackResult(track.track_id, track.get_tlwh(), track.score))
            if track.time_since_update <= self.max_age:
                alive_tracks.append(track)
        self.tracks = alive_tracks
        return outputs
