## Repository layout

```text
.
|-- commands/
|   |-- run_fvsa_hrtr.ps1
|   |-- run_3d_evaluation.ps1
|   `-- run_fvsa_hrtr.sh
|-- configs/
|   |-- fvsa_fixed_parameters.json
|   |-- hrtr_fixed_parameters.json
|   `-- paper_parameters.yaml
|-- docs/
|   |-- EVALUATION_PROTOCOL.md
|   `-- IMPLEMENTATION_MAP.md
|-- src/
|   |-- step4_stereo_association_eval.py       # complete FVSA implementation
|   |-- step5_reliability_guided_3d_reconstruction.py # complete HRTR implementation
|   |-- step5_point_3d_mot_comparison.py       # canonical 25 mm 3D evaluation
|   |-- step5_audit_3d_matching_threshold.py   # protocol/sensitivity audit
|   |-- step3_mot_tracking_eval.py             # MOT I/O and TrackEval dependency
|   |-- step1_stereo_rectification.py          # calibration loader dependency
|   `-- _mot_trackers.py                       # imported tracking data structures
|-- requirements.txt
```

