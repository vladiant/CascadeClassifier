# Benchmark: accuracy and latency

This page compares four detector families on the same dataset:

1. Cascade trained in this repository (`quickstart/classifier/cascade.xml`)
2. Baseline `cv2.CascadeClassifier` (OpenCV frontal-face Haar cascade)
3. OpenCV HOG detector (`cv2.HOGDescriptor_getDefaultPeopleDetector`)
4. Tiny YOLO model loaded from ONNX (for example `yolov5n.onnx`)

## Dataset

The benchmark uses the same files that are already part of `quickstart/`:

* Positives with bounding boxes: `quickstart/positives/info.dat`
* Negatives: `quickstart/negatives/bg_abs.txt`

By default, all rows are used (`400` positives and `467` negatives at the time of writing).

## Metrics

For each detector we report:

* **Precision / Recall / F1** using IoU threshold `0.5`
* **Mean IoU (TP only)**
* **Mean latency (ms/image)**
* **P95 latency (ms/image)**

The evaluator uses one best prediction per image (highest score when available, otherwise largest box).

## Reproducing the benchmark

Run from repository root:

```sh
# optional: download a tiny YOLO ONNX model
mkdir -p models
curl -L -o models/yolov5n.onnx \
  https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.onnx

# run benchmark
python3 tools/detection/Python/benchmark.py \
  --yolo-onnx models/yolov5n.onnx \
  --output-csv quickstart/benchmark_results.csv
```

To benchmark a face-specific tiny YOLO model, provide its ONNX path and class index:

```sh
python3 tools/detection/Python/benchmark.py \
  --yolo-onnx /path/to/tiny-face-yolo.onnx \
  --yolo-class-id 0
```

## Result table (local sampled run)

Local run details:

* Date: 2026-10-24
* Sample size: `25` positives + `25` negatives (`50` total)
* YOLO input size: `320`
* OpenCL: disabled (`cv2.ocl.setUseOpenCL(False)`) for stable CPU timing

| Detector | Samples | TP | FP | FN | Precision | Recall | F1 | Mean IoU (TP) | Mean latency (ms) | P95 latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cascade-custom | 50 | 0 | 50 | 25 | 0.000 | 0.000 | 0.000 | 0.000 | 12357.69 | 13425.62 |
| cascade-opencv | 50 | 0 | 26 | 25 | 0.000 | 0.000 | 0.000 | 0.000 | 25.92 | 68.91 |
| hog-default-people | 50 | 1 | 9 | 24 | 0.100 | 0.040 | 0.057 | 0.554 | 17.72 | 65.63 |
| yolo-onnx:yolov5n.onnx | 50 | 25 | 6 | 0 | 0.806 | 1.000 | 0.893 | 0.874 | 20.10 | 21.50 |

## Notes

* HOG default people detector is optimized for full-body pedestrian detection; on face-centric datasets it is expected to underperform.
* For fair comparison with cascades on face data, use a tiny YOLO model trained for face detection.
* Different OpenCV builds may expose different APIs; this benchmark expects Python bindings with `CascadeClassifier`, `HOGDescriptor`, and `dnn`.
