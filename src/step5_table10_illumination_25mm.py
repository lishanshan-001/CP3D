"""Reproduce Table 10 under the finalized 25 mm point-matching protocol.

The script evaluates the canonical complete-framework reconstruction separately
for each CP3D test illumination sequence. It reuses the exact matching, spatial
error, jitter, and coverage implementations from the point-based 3D evaluation.
No model or evaluation parameter is selected on the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import step5_point_3d_mot_comparison as p3d


METHOD = "Proposed reliability-guided framework"
ILLUMINATION_LABELS = {
    "CP-I0-03": "I0",
    "CP-I1-03": "I1",
    "CP-I2-03": "I2",
    "CP-I3-03": "I3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the illumination-wise 3D reconstruction table at 25 mm."
    )
    parser.add_argument("--pipeline-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--distance-mm", type=float, default=25.0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pipeline_root = args.pipeline_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    step4_root = pipeline_root / "step4_association"
    step1_root = pipeline_root / "step1_rectification"
    metadata = json.loads((step4_root / "run_metadata.json").read_text(encoding="utf-8"))
    train_sequences = list(metadata["train_sequences"])
    test_sequences = list(metadata["test_sequences"])
    expected = list(ILLUMINATION_LABELS)
    if test_sequences != expected:
        raise RuntimeError(f"Unexpected CP3D test split: {test_sequences}; expected {expected}")

    p3d.s4.configure_input_roots(
        Path(metadata["step3_train_root"]),
        Path(metadata["step3_test_root"]),
        data_root,
        step1_root,
    )
    p3d.s5.configure_step1_root(step1_root)
    priors = p3d.s4.compute_group_priors(train_sequences)
    test_contexts = [p3d.s4.build_sequence_context(sequence, priors) for sequence in test_sequences]

    source = (
        pipeline_root
        / "step5_reconstruction"
        / "paper_assets"
        / "reports"
        / "reconstruction_points_compag.csv"
    )
    predictions = p3d.load_proposed_points(source, test_sequences)
    gt_maps = {sequence: p3d.s5.build_gt_reference(sequence)[0] for sequence in test_sequences}

    detailed_rows: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []
    for context in test_contexts:
        sequence = context.seq_name
        row, _raw = p3d.evaluate_one_sequence(
            sequence,
            gt_maps[sequence],
            predictions[sequence],
            context.seq_length,
            args.distance_mm,
        )
        detailed = {"method": METHOD, "illumination_condition": ILLUMINATION_LABELS[sequence], **row}
        detailed_rows.append(detailed)
        table_rows.append(
            {
                "Illumination condition": ILLUMINATION_LABELS[sequence],
                "RMSE (mm)": float(row["RMSE_mm"]),
                "Jitter (mm)": float(row["Jitter_mm"]),
                "Coverage (%)": float(row["Coverage_pct"]),
            }
        )

    write_csv(tables_dir / "Table10_illumination_3D_performance_25mm.csv", table_rows)
    write_csv(reports_dir / "Table10_illumination_3D_performance_25mm_audit.csv", detailed_rows)

    markdown = [
        "# Table 10. 3D trajectory reconstruction performance of the proposed framework across illumination conditions.",
        "",
        "| Illumination condition | RMSE (mm) | Jitter (mm) | Coverage (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in table_rows:
        markdown.append(
            f"| {row['Illumination condition']} | {float(row['RMSE (mm)']):.2f} | "
            f"{float(row['Jitter (mm)']):.2f} | {float(row['Coverage (%)']):.2f} |"
        )
    markdown.extend(
        [
            "",
            "Note: Each illumination condition contains one held-out test sequence. RMSE is computed over frame-wise Hungarian point matches with Euclidean distance no greater than 25 mm. Jitter is the mean norm of second XYZ differences over consecutive predicted points within the same reconstructed identity. Coverage is the number of accepted 25 mm matches divided by all alignable pseudo-3D reference points in the corresponding sequence.",
        ]
    )
    (tables_dir / "Table10_illumination_3D_performance_25mm.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex_rows = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Illumination condition & RMSE (mm) & Jitter (mm) & Coverage (\\%) \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        latex_rows.append(
            f"{row['Illumination condition']} & {float(row['RMSE (mm)']):.2f} & "
            f"{float(row['Jitter (mm)']):.2f} & {float(row['Coverage (%)']):.2f} \\\\"
        )
    latex_rows.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "Table10_illumination_3D_performance_25mm.tex").write_text(
        "\n".join(latex_rows) + "\n", encoding="utf-8"
    )

    validation = {
        "experiment": "Illumination-wise performance of the complete framework",
        "method": METHOD,
        "test_sequences": test_sequences,
        "one_test_sequence_per_illumination": True,
        "canonical_prediction_source": str(source),
        "pseudo_3d_reference": "Synchronized rectified GT-centre triangulation",
        "distance_threshold_mm": float(args.distance_mm),
        "point_matching": "Frame-wise Hungarian assignment; accept Euclidean distance <= 25 mm",
        "rmse_support": "Accepted 25 mm point matches",
        "jitter": "Mean L2 norm of second XYZ differences over consecutive predicted points within identity",
        "coverage": "Accepted 25 mm matches / all alignable pseudo-3D reference points",
        "parameter_selection_on_test_split": False,
        "all_sequences_present": [row["sequence"] for row in detailed_rows] == test_sequences,
        "counts_consistent": all(
            int(float(row["Matched_points"])) <= int(float(row["Predicted_points"]))
            and int(float(row["Matched_points"])) <= int(float(row["GT_points"]))
            for row in detailed_rows
        ),
    }
    (reports_dir / "Table10_illumination_3D_performance_25mm_config.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Table 10 (25 mm protocol)")
    for row in table_rows:
        print(
            f"{row['Illumination condition']}: RMSE={float(row['RMSE (mm)']):.2f} mm, "
            f"Jitter={float(row['Jitter (mm)']):.2f} mm, "
            f"Coverage={float(row['Coverage (%)']):.2f}%"
        )


if __name__ == "__main__":
    main()
