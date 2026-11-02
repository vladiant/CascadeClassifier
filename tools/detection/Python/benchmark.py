#!/usr/bin/env python3
"""Benchmark object detectors on the same positive/negative dataset."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Sample:
    image_path: Path
    gt_bbox: BBox | None


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    samples: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_iou_tp: float
    mean_latency_ms: float
    p95_latency_ms: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    quickstart_dir = repo_root / "quickstart"

    parser = argparse.ArgumentParser(
        description="Benchmark custom CascadeClassifier vs OpenCV cascade, HOG and YOLO."
    )
    parser.add_argument(
        "--info-file",
        type=Path,
        default=quickstart_dir / "positives" / "info.dat",
        help="Path to createsamples info.dat for positive samples.",
    )
    parser.add_argument(
        "--negatives-file",
        type=Path,
        default=quickstart_dir / "negatives" / "bg_abs.txt",
        help="Path to one-image-per-line negatives file.",
    )
    parser.add_argument(
        "--custom-cascade",
        type=Path,
        default=quickstart_dir / "classifier" / "cascade.xml",
        help="Path to the cascade XML produced by this repository.",
    )
    parser.add_argument(
        "--opencv-cascade",
        type=Path,
        default=Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        help="Path to a baseline OpenCV cascade XML.",
    )
    parser.add_argument(
        "--yolo-onnx",
        type=Path,
        default=None,
        help="Path to a tiny YOLO ONNX file (for example yolov5n.onnx).",
    )
    parser.add_argument(
        "--yolo-class-id",
        type=int,
        default=0,
        help="Class index to score for YOLO. Set to -1 for best class per prediction.",
    )
    parser.add_argument(
        "--yolo-input-size",
        type=int,
        default=640,
        help="YOLO input resolution (square).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Score threshold for HOG and YOLO predictions.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.45,
        help="NMS threshold for YOLO predictions.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold to mark a prediction as true positive.",
    )
    parser.add_argument(
        "--max-positives",
        type=int,
        default=None,
        help="Optional cap on number of positive samples.",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=None,
        help="Optional cap on number of negative samples.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write benchmark results as CSV.",
    )
    return parser.parse_args()


def parse_info_line(info_line: str, info_file: Path) -> Sample:
    parts = info_line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid info.dat row: {info_line!r}")

    image_path = (info_file.parent / parts[0]).resolve()
    objects_count = int(parts[1])

    if objects_count == 0:
        return Sample(image_path=image_path, gt_bbox=None)

    if len(parts) < 6:
        raise ValueError(f"Missing bbox coordinates in row: {info_line!r}")

    x, y, w, h = (int(value) for value in parts[2:6])
    return Sample(image_path=image_path, gt_bbox=(x, y, w, h))


def load_samples(
    info_file: Path,
    negatives_file: Path,
    max_positives: int | None,
    max_negatives: int | None,
) -> list[Sample]:
    if not info_file.exists():
        raise FileNotFoundError(f"Missing positives info file: {info_file}")
    if not negatives_file.exists():
        raise FileNotFoundError(f"Missing negatives file: {negatives_file}")

    positives = [
        parse_info_line(line.strip(), info_file)
        for line in info_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_positives is not None:
        positives = positives[:max_positives]

    negatives: list[Sample] = []
    for line in negatives_file.read_text(encoding="utf-8").splitlines():
        path = line.strip()
        if not path:
            continue
        negatives.append(Sample(image_path=Path(path), gt_bbox=None))

    if max_negatives is not None:
        negatives = negatives[:max_negatives]

    return positives + negatives


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(ax + aw, bx + bw)
    y_bottom = min(ay + ah, by + bh)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = (aw * ah) + (bw * bh) - intersection
    return float(intersection / union) if union > 0 else 0.0


def pick_best_box(boxes: list[BBox], scores: list[float] | None) -> BBox | None:
    if not boxes:
        return None
    if scores is None or not scores:
        return max(boxes, key=lambda box: box[2] * box[3])
    best_index = max(range(len(boxes)), key=lambda idx: scores[idx])
    return boxes[best_index]


def build_cascade_predictor(xml_path: Path, name: str) -> Callable[[np.ndarray], tuple[list[BBox], list[float] | None]]:
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("cv2.CascadeClassifier is unavailable in this OpenCV build.")
    if not xml_path.exists():
        raise FileNotFoundError(f"{name} XML not found: {xml_path}")

    classifier = cv2.CascadeClassifier(str(xml_path))
    if classifier.empty():
        raise RuntimeError(f"Failed to load {name} XML: {xml_path}")

    def predict(image_bgr: np.ndarray) -> tuple[list[BBox], list[float] | None]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        detected = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        boxes = [tuple(int(value) for value in row) for row in detected]
        return boxes, None

    return predict


def build_hog_predictor(score_threshold: float) -> Callable[[np.ndarray], tuple[list[BBox], list[float] | None]]:
    if not hasattr(cv2, "HOGDescriptor") or not hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
        raise RuntimeError("HOGDescriptor APIs are unavailable in this OpenCV build.")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def predict(image_bgr: np.ndarray) -> tuple[list[BBox], list[float] | None]:
        detected, weights = hog.detectMultiScale(
            image_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        boxes: list[BBox] = []
        scores: list[float] = []
        for (x, y, w, h), weight in zip(detected, weights):
            score = float(weight)
            if score < score_threshold:
                continue
            boxes.append((int(x), int(y), int(w), int(h)))
            scores.append(score)
        return boxes, scores

    return predict


def parse_yolo_predictions(
    output: np.ndarray,
    image_width: int,
    image_height: int,
    input_size: int,
    class_id: int,
    score_threshold: float,
) -> tuple[list[BBox], list[float]]:
    predictions = output
    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.ndim != 2:
        raise RuntimeError(f"Unexpected YOLO output shape: {output.shape}")

    if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] in (84, 85):
        predictions = predictions.T

    boxes: list[BBox] = []
    scores: list[float] = []

    x_scale = image_width / float(input_size)
    y_scale = image_height / float(input_size)

    for row in predictions:
        if row.shape[0] < 6:
            continue

        cx, cy, w, h = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        class_scores = row[5:] if row.shape[0] > 5 else np.empty(0, dtype=np.float32)

        if row.shape[0] >= 6 and row[4] <= 1.0:
            objectness = float(row[4])
            if class_scores.size == 0:
                score = objectness
            elif class_id >= 0 and class_id < class_scores.size:
                score = objectness * float(class_scores[class_id])
            else:
                score = objectness * float(np.max(class_scores))
        else:
            if class_scores.size == 0:
                continue
            if class_id >= 0 and class_id < class_scores.size:
                score = float(class_scores[class_id])
            else:
                score = float(np.max(class_scores))

        if score < score_threshold:
            continue

        left = int((cx - (w / 2.0)) * x_scale)
        top = int((cy - (h / 2.0)) * y_scale)
        box_w = int(w * x_scale)
        box_h = int(h * y_scale)
        boxes.append((left, top, box_w, box_h))
        scores.append(score)

    return boxes, scores


def build_yolo_predictor(
    onnx_path: Path,
    input_size: int,
    class_id: int,
    score_threshold: float,
    nms_threshold: float,
) -> Callable[[np.ndarray], tuple[list[BBox], list[float] | None]]:
    if not onnx_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {onnx_path}")

    net = cv2.dnn.readNetFromONNX(str(onnx_path))

    def predict(image_bgr: np.ndarray) -> tuple[list[BBox], list[float] | None]:
        height, width = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(input_size, input_size),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        output = net.forward()

        boxes, scores = parse_yolo_predictions(
            output=output,
            image_width=width,
            image_height=height,
            input_size=input_size,
            class_id=class_id,
            score_threshold=score_threshold,
        )

        if not boxes:
            return [], []

        indexes = cv2.dnn.NMSBoxes(
            bboxes=[list(box) for box in boxes],
            scores=scores,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
        if len(indexes) == 0:
            return [], []

        selected = np.array(indexes).reshape(-1).astype(int)
        kept_boxes = [boxes[idx] for idx in selected]
        kept_scores = [scores[idx] for idx in selected]
        return kept_boxes, kept_scores

    return predict


def run_detector(
    name: str,
    predictor: Callable[[np.ndarray], tuple[list[BBox], list[float] | None]],
    samples: list[Sample],
    iou_threshold: float,
) -> BenchmarkResult:
    tp = 0
    fp = 0
    fn = 0
    ious_for_tp: list[float] = []
    latencies_ms: list[float] = []

    for sample in samples:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {sample.image_path}")

        start_time = time.perf_counter()
        boxes, scores = predictor(image)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(elapsed_ms)

        prediction = pick_best_box(boxes, scores)

        if sample.gt_bbox is None:
            if prediction is not None:
                fp += 1
            continue

        if prediction is None:
            fn += 1
            continue

        overlap = iou(sample.gt_bbox, prediction)
        if overlap >= iou_threshold:
            tp += 1
            ious_for_tp.append(overlap)
        else:
            fp += 1
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = (sum(ious_for_tp) / len(ious_for_tp)) if ious_for_tp else 0.0
    mean_latency = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    p95_latency = float(np.percentile(np.array(latencies_ms, dtype=np.float64), 95)) if latencies_ms else 0.0

    return BenchmarkResult(
        name=name,
        samples=len(samples),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou_tp=mean_iou,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95_latency,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    headers = (
        "Detector",
        "Samples",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1",
        "Mean IoU (TP)",
        "Mean latency (ms)",
        "P95 latency (ms)",
    )
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in results:
        print(
            f"{row.name} | {row.samples} | {row.tp} | {row.fp} | {row.fn} | "
            f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | "
            f"{row.mean_iou_tp:.3f} | {row.mean_latency_ms:.2f} | {row.p95_latency_ms:.2f}"
        )


def write_csv(results: list[BenchmarkResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "detector",
                "samples",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_iou_tp",
                "mean_latency_ms",
                "p95_latency_ms",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.name,
                    row.samples,
                    row.tp,
                    row.fp,
                    row.fn,
                    f"{row.precision:.6f}",
                    f"{row.recall:.6f}",
                    f"{row.f1:.6f}",
                    f"{row.mean_iou_tp:.6f}",
                    f"{row.mean_latency_ms:.6f}",
                    f"{row.p95_latency_ms:.6f}",
                ]
            )


def main() -> int:
    args = parse_args()
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    samples = load_samples(
        info_file=args.info_file.resolve(),
        negatives_file=args.negatives_file.resolve(),
        max_positives=args.max_positives,
        max_negatives=args.max_negatives,
    )
    if not samples:
        raise RuntimeError("No benchmark samples were loaded.")

    detectors: list[tuple[str, Callable[[np.ndarray], tuple[list[BBox], list[float] | None]]]] = [
        (
            "cascade-custom",
            build_cascade_predictor(args.custom_cascade.resolve(), "custom cascade"),
        ),
        (
            "cascade-opencv",
            build_cascade_predictor(args.opencv_cascade.resolve(), "opencv cascade"),
        ),
        (
            "hog-default-people",
            build_hog_predictor(score_threshold=args.score_threshold),
        ),
    ]

    if args.yolo_onnx is not None:
        detectors.append(
            (
                f"yolo-onnx:{args.yolo_onnx.name}",
                build_yolo_predictor(
                    onnx_path=args.yolo_onnx.resolve(),
                    input_size=args.yolo_input_size,
                    class_id=args.yolo_class_id,
                    score_threshold=args.score_threshold,
                    nms_threshold=args.nms_threshold,
                ),
            )
        )

    results = [
        run_detector(
            name=name,
            predictor=predictor,
            samples=samples,
            iou_threshold=args.iou_threshold,
        )
        for name, predictor in detectors
    ]

    print_results(results)

    if args.output_csv is not None:
        write_csv(results, args.output_csv.resolve())
        print(f"\nWrote CSV: {args.output_csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
