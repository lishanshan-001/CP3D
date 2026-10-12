# -*- coding: utf-8 -*-

"""
Step 1
Stereo undistortion and rectification

StereoCP3D project
"""

import argparse
import json
import os
import re
from glob import glob

import cv2
import numpy as np


# =========================
# Path
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

CALIB_FILE = os.path.join(ROOT, "calibration", "stereo_params.txt")
DATA_ROOT = os.path.join(ROOT, "CP3D_dataset")
SAVE_ROOT = os.path.join(ROOT, "results", "step1_rectification")


# =========================
# Natural sort
# =========================

def natural_sort(files):
    return sorted(
        files,
        key=lambda x: [
            int(t) if t.isdigit() else t
            for t in re.split(r"(\d+)", x)
        ],
    )


def collect_images(image_dir):
    image_paths = []
    for pattern in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
        image_paths.extend(glob(os.path.join(image_dir, pattern)))
    return natural_sort(image_paths)


# =========================
# Calibration parser
# =========================

def parse_matrix(text, name):
    pattern = name + r"\s*=\s*\[\[(.*?)\]\]"
    result = re.search(pattern, text, re.S)

    if result is None:
        raise ValueError(f"{name} not found in {CALIB_FILE}")

    data = result.group(1).replace("\n", " ").replace("[", " ").replace("]", " ")
    values = np.fromstring(data, sep=" ")

    if values.size == 0:
        raise ValueError(f"{name} is empty in {CALIB_FILE}")

    return values


def load_calibration(calib_file=None):
    calib_path = os.path.abspath(calib_file or CALIB_FILE)
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, "r", encoding="utf-8") as f:
        txt = f.read()

    K1 = parse_matrix(txt, "K1").reshape(3, 3)
    D1 = parse_matrix(txt, "D1").reshape(1, 5)
    K2 = parse_matrix(txt, "K2").reshape(3, 3)
    D2 = parse_matrix(txt, "D2").reshape(1, 5)
    R = parse_matrix(txt, "R").reshape(3, 3)
    T = parse_matrix(txt, "T").reshape(3, 1)

    return K1, D1, K2, D2, R, T


# =========================
# Rectification
# =========================

def create_rectification(image_size, K1, D1, K2, D2, R, T):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.9,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1,
        D1,
        R1,
        P1,
        image_size,
        cv2.CV_32FC1,
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        K2,
        D2,
        R2,
        P2,
        image_size,
        cv2.CV_32FC1,
    )

    return map1x, map1y, map2x, map2y, P1, P2, Q, R1, R2, roi1, roi2


# =========================
# Process one sequence
# =========================

def process_sequence(seq_path, K1, D1, K2, D2, R, T, save_root=SAVE_ROOT, progress_callback=None):
    seq_name = os.path.basename(seq_path)
    left_dir = os.path.join(seq_path, "img1")
    right_dir = os.path.join(seq_path, "img2")

    if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
        raise FileNotFoundError(f"{seq_name} is missing img1 or img2 directory")

    left_imgs = collect_images(left_dir)
    right_imgs = collect_images(right_dir)

    if not left_imgs or not right_imgs:
        raise ValueError(f"{seq_name} contains no stereo images")

    if len(left_imgs) != len(right_imgs):
        raise ValueError(
            f"{seq_name} left/right image count mismatch: "
            f"{len(left_imgs)} vs {len(right_imgs)}"
        )

    first_left = cv2.imread(left_imgs[0])
    if first_left is None:
        raise ValueError(f"Failed to read image: {left_imgs[0]}")

    h, w = first_left.shape[:2]
    maps = create_rectification((w, h), K1, D1, K2, D2, R, T)
    map1x, map1y, map2x, map2y, P1, P2, Q, R1, R2, roi1, roi2 = maps

    save_dir = os.path.join(save_root, seq_name)
    out1 = os.path.join(save_dir, "img1")
    out2 = os.path.join(save_dir, "img2")

    os.makedirs(out1, exist_ok=True)
    os.makedirs(out2, exist_ok=True)

    total_frames = len(left_imgs)
    for frame_index, (left_path, right_path) in enumerate(zip(left_imgs, right_imgs), start=1):
        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)

        if imgL is None:
            raise ValueError(f"Failed to read image: {left_path}")
        if imgR is None:
            raise ValueError(f"Failed to read image: {right_path}")

        rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

        name = os.path.basename(left_path)
        cv2.imwrite(os.path.join(out1, name), rectL)
        cv2.imwrite(os.path.join(out2, name), rectR)
        if progress_callback and (frame_index == 1 or frame_index % 25 == 0 or frame_index == total_frames):
            progress_callback(frame_index, total_frames)

    np.save(os.path.join(save_dir, "P1.npy"), P1)
    np.save(os.path.join(save_dir, "P2.npy"), P2)
    np.save(os.path.join(save_dir, "Q.npy"), Q)
    np.save(os.path.join(save_dir, "R1.npy"), R1)
    np.save(os.path.join(save_dir, "R2.npy"), R2)
    np.save(os.path.join(save_dir, "roi1.npy"), np.array(roi1, dtype=np.int32))
    np.save(os.path.join(save_dir, "roi2.npy"), np.array(roi2, dtype=np.int32))

    print(f"{seq_name}: {len(left_imgs)} frames finished")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Stereo undistortion and rectification")
    parser.add_argument("--calib-file", default=CALIB_FILE)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--save-root", default=SAVE_ROOT, help="Output root for all rectified sequences")
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence names for a targeted validation run.",
    )
    parser.add_argument(
        "--rotation-convention",
        choices=["direct", "matlab-transpose"],
        default="direct",
        help="Use matlab-transpose when converting a MATLAB row-vector rotation to OpenCV.",
    )
    args = parser.parse_args()

    calib_file = os.path.abspath(args.calib_file)
    data_root = os.path.abspath(args.data_root)
    K1, D1, K2, D2, R, T = load_calibration(calib_file)
    rotation_source = R.copy()
    if args.rotation_convention == "matlab-transpose":
        R = R.T
    save_root = os.path.abspath(args.save_root)
    os.makedirs(save_root, exist_ok=True)

    print("==============================")
    print("Stereo calibration loaded")
    print("------------------------------")
    print("Project root:")
    print(ROOT)
    print("K1:")
    print(K1)
    print("K2:")
    print(K2)
    print("Rotation convention:", args.rotation_convention)
    print("Output root:", save_root)
    print("Baseline(mm):", np.linalg.norm(T))
    print("==============================")

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    for split in ["train", "test"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"Skip missing split: {split_dir}")
            continue

        sequences = natural_sort(glob(os.path.join(split_dir, "CP-*")))
        if args.sequences:
            requested = set(args.sequences)
            sequences = [path for path in sequences if os.path.basename(path) in requested]
        print(f"Processing {split}: {len(sequences)} sequences")

        for seq in sequences:
            process_sequence(seq, K1, D1, K2, D2, R, T, save_root=save_root)

    metadata = {
        "calibration_file": calib_file,
        "dataset_root": data_root,
        "save_root": save_root,
        "rotation_convention": args.rotation_convention,
        "requested_sequences": args.sequences,
        "rotation_source_matlab": rotation_source.tolist(),
        "rotation_used_opencv": R.tolist(),
        "translation_used_opencv_mm": T.reshape(-1).tolist(),
        "baseline_mm": float(np.linalg.norm(T)),
        "coordinate_definition": {
            "origin": "rectified first-camera optical center",
            "x_axis": "right",
            "y_axis": "down",
            "z_axis": "forward",
            "unit": "mm",
        },
    }
    with open(os.path.join(save_root, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("DONE")


if __name__ == "__main__":
    main()
