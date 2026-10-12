# FVSA and HRTR implementation map

## FVSA

File: `src/step4_stereo_association_eval.py`

| Method operation | Function or type |
|---|---|
| Track and pair data structures | `Detection`, `Track`, `PairFeature`, `SequenceContext` |
| Development-set disparity and vertical-residual priors | `compute_group_priors` |
| Seven normalized pair cues | `compute_pair_feature` |
| FVSA affinity fusion | `feature_score` with `method == "Proposed"` |
| Primary one-to-one assignment and secondary fragment completion | `proposed_components` |
| Complete FVSA component construction | `method_components` |
| Shared stereo-ID export | `rows_from_components`, `export_method_predictions` |
| Fixed operating point | `configs/fvsa_fixed_parameters.json` |

The seven FVSA cues are temporal overlap, epipolar consistency, spatial/disparity
consistency, motion consistency, relative horizontal rank, local mutual
consistency, and calibrated-region support. Candidate pairs require at least
three overlapping frames and pass the vertical-residual pre-screen implemented
at the end of `compute_pair_feature`.

## HRTR

File: `src/step5_reliability_guided_3d_reconstruction.py`

| Method operation | Function or type |
|---|---|
| Stereo segment construction and continuity rule | `build_segments_for_context` |
| Observation reliability | `frame_level_reliability` |
| Segment descriptors | `compute_segment_features` |
| Segment reliability score | `compute_quality_scores` (`ProposedTQS`) |
| Development-set depth prior | `compute_depth_priors` |
| Reliability/depth screening, weighted temporal refinement, adaptive fusion, and short-gap completion | `reconstruct_segment_points` (`ReliabilityGuided3D`) |
| End-to-end reconstruction and metric aggregation | `evaluate_reconstruction` |
| Fixed operating point | `configs/hrtr_fixed_parameters.json` |

HRTR is the paper name for the integrated reliability assessment and
reliability-guided temporal reconstruction stage. The source retains historical
internal labels so that archived configurations and result metadata continue to
match exactly.

## Dependency files

`step4_stereo_association_eval.py` imports `step3_mot_tracking_eval.py` for
dataset discovery, sequence metadata, MOT I/O conventions, and TrackEval setup.
That module imports `_mot_trackers.py` and the calibration loader in
`step1_stereo_rectification.py`; all three are included so the snapshot has no
missing local imports.

These dependency files are not additional proposed modules and do not change
FVSA or HRTR.
