# Evaluation protocol implementation

## 1. Detection

File: `src/step2_evaluate_corrected_detection.py`

This script performs one-to-one IoU matching and reports precision, recall, and
F1 on rectified full-frame coordinates.

## 2. Single-view 2D tracking

Files:

- `src/step3_official_mot_tracking_eval.py`
- `src/step3_evaluate_combined_views.py`

The four left and four right test views are treated as eight independent MOT
sequences and jointly evaluated using the official TrackEval HOTA, Identity,
and CLEAR implementations. Reported metrics are HOTA, AssA, IDF1, MOTA, and
IDSW.

## 3. Stereo trajectory association

File: `src/step4_stereo_association_eval.py`

The protocol is implemented by `fuse_gt_sequence`, `rows_from_components`, and
`run_trackeval`. For original time index `t`, the left observation is written
at fused frame `2t-1` and the right observation at `2t`. A correctly associated
pair retains one shared predicted identity across those observations. TrackEval
then computes HOTA, AssA, IDF1, MOTA, and IDSW on the fused sequence.

## 4. Point-based 3D evaluation

Canonical file: `src/step5_point_3d_mot_comparison.py`

Key implementation locations:

| Operation | Function |
|---|---|
| Reference conversion to frame-wise point sets | `gt_by_frame` |
| Point-distance similarity supplied to TrackEval | `build_trackeval_data` |
| Frame-wise Hungarian assignment and 25 mm rejection | `spatial_error_terms` |
| Unnormalized second-coordinate-difference jitter | `jitter_terms` |
| One-sequence metric calculation | `evaluate_one_sequence` |
| Pooled four-sequence calculation | `evaluate_method` |

The Euclidean distance matrix in frame `t` is

```text
d[i,j] = ||X_ref[i] - X_pred[j]||_2.
```

RMSE, MAE, MedAE, and Coverage use frame-wise Hungarian assignment followed by
rejection of assignments with `d > 25 mm`. Coverage is accepted matches divided
by all alignable pseudo-3D reference points. Jitter is calculated from predicted
trajectories and does not use the 25 mm correspondence gate.

### Important HOTA-3D detail

The archived main-table HOTA-3D is not a single binary 25 mm HOTA score. The
canonical implementation supplies TrackEval with

```text
s[i,j] = max(0, 1 - d[i,j] / (2 * 25 mm)).
```

TrackEval evaluates 19 alpha thresholds from 0.05 to 0.95 and averages them.
Thus, 25 mm corresponds to `alpha = 0.5`; it is the distance anchor rather than
a universal maximum distance for every HOTA alpha. IDF1-3D, MOTA-3D, and IDSW
use TrackEval threshold 0.5 and therefore use the 25 mm decision boundary.

The complete audit and the alternative binary/bounded-continuous HOTA variants
are implemented in `src/step5_audit_3d_matching_threshold.py`. This distinction
must remain explicit in the manuscript.

## 5. Applications of the same protocol

- `step5_unified_25mm_framework_ablation.py`: framework ablation.
- `step5_table10_illumination_25mm.py`: per-illumination evaluation.
- `summarize_frame_rate_experiment.py` and
  `supplement_frame_rate_fixed_window_metrics.py`: sampling-rate evaluation and
  fixed-window fluctuation.
- `step5_movement_descriptor_reliability.py`: movement-descriptor errors and
  longest identity-consistent coverage.
- `step5_validate_corrected_reconstruction.py`: independent output validation.

These scripts reuse the canonical types/functions instead of redefining a
different test-set matching rule.
