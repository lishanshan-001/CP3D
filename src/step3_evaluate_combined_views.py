#!/usr/bin/env python
"""Evaluate left and right tracker outputs jointly as eight MOT sequences."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import trackeval


TRACKERS = ["SORT", "ByteTrack", "OCSort", "IOUTrack"]
VIEWS = ["left", "right"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def alias(sequence: str, view: str) -> str:
    return f"{sequence}-{view}"


def write_seqinfo(path: Path, sequence_alias: str) -> None:
    path.write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={sequence_alias}",
                "imDir=img1",
                "frameRate=20",
                "seqLength=600",
                "imWidth=1920",
                "imHeight=1088",
                "imExt=.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )


def extract_summary(path: Path) -> dict[str, float]:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    values = dict(zip(lines[0], lines[1]))
    return {
        "HOTA": float(values["HOTA"]),
        "DetA": float(values["DetA"]),
        "AssA": float(values["AssA"]),
        "IDF1": float(values["IDF1"]),
        "MOTA": float(values["MOTA"]),
        "IDSW": int(round(float(values["IDSW"]))),
        "Frag": int(round(float(values["Frag"]))),
    }


def run_joint_evaluation(
    source_results_root: str | Path,
    save_root_value: str | Path,
    sequences: list[str],
    trackers: list[str],
    split_tag: str = "test",
    verbose: bool = True,
) -> dict[str, object]:
    source_root = Path(source_results_root).resolve()
    save_root = Path(save_root_value).resolve()
    gt_root = save_root / "gt" / "mot_challenge"
    tracker_root = save_root / "trackers" / "mot_challenge"
    eval_root = save_root / "eval"
    report_root = save_root / "reports"
    dataset_name = f"CP3D-stereo-{split_tag}-all"
    benchmark = f"CP3D-stereo-{split_tag}"
    sequence_aliases = []

    for view in VIEWS:
        source_dataset = f"CP3D-{view}-all"
        for sequence in sequences:
            sequence_alias = alias(sequence, view)
            sequence_aliases.append(sequence_alias)
            destination_sequence = gt_root / dataset_name / sequence_alias
            ensure_dir(destination_sequence / "gt")
            source_gt = (
                source_root
                / "gt"
                / "mot_challenge"
                / source_dataset
                / sequence
                / "gt"
                / "gt.txt"
            )
            shutil.copyfile(source_gt, destination_sequence / "gt" / "gt.txt")
            write_seqinfo(destination_sequence / "seqinfo.ini", sequence_alias)

            for tracker in trackers:
                source_track = (
                    source_root
                    / "trackers"
                    / "mot_challenge"
                    / source_dataset
                    / tracker
                    / "data"
                    / f"{sequence}.txt"
                )
                destination_track = (
                    tracker_root
                    / dataset_name
                    / tracker
                    / "data"
                    / f"{sequence_alias}.txt"
                )
                ensure_dir(destination_track.parent)
                shutil.copyfile(source_track, destination_track)

    ensure_dir(gt_root / "seqmaps")
    with (gt_root / "seqmaps" / f"{dataset_name}.txt").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["name"])
        writer.writerows([[name] for name in sequence_aliases])

    output_folder = eval_root / dataset_name
    ensure_dir(output_folder)
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": True,
            "LOG_ON_ERROR": str(save_root / "trackeval_error_log.txt"),
        }
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(tracker_root),
            "OUTPUT_FOLDER": str(output_folder),
            "BENCHMARK": benchmark,
            "SPLIT_TO_EVAL": "all",
            "SEQMAP_FOLDER": str(gt_root / "seqmaps"),
            "TRACKERS_TO_EVAL": trackers,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "CLASSES_TO_EVAL": ["pedestrian"],
            "DO_PREPROC": False,
            "PRINT_CONFIG": False,
            "SEQ_INFO": {name: None for name in sequence_aliases},
        }
    )
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    evaluator.evaluate(
        [dataset],
        [
            trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
            trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
            trackeval.metrics.Identity({"PRINT_CONFIG": False}),
        ],
        show_progressbar=False,
    )

    rows = []
    for tracker in trackers:
        rows.append(
            {
                "tracker": tracker,
                **extract_summary(output_folder / tracker / "pedestrian_summary.txt"),
            }
        )
    ensure_dir(report_root)
    fields = ["tracker", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag"]
    with (report_root / "combined_view_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "evaluation_definition": "official_joint_evaluation_of_independent_view_sequences",
        "source_results_root": str(source_root),
        "save_root": str(save_root),
        "dataset_name": dataset_name,
        "split_tag": split_tag,
        "sequences": sequences,
        "sequence_aliases": sequence_aliases,
        "trackers": trackers,
        "metrics": rows,
    }
    (save_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if verbose:
        print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results-root", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--trackers", nargs="+", default=TRACKERS, choices=TRACKERS)
    parser.add_argument("--split-tag", default="test")
    args = parser.parse_args()

    run_joint_evaluation(
        args.source_results_root,
        args.save_root,
        args.sequences,
        args.trackers,
        args.split_tag,
        True,
    )


if __name__ == "__main__":
    main()
