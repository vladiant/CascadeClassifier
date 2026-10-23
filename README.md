# CascadeClassifier

[![Ubuntu GCC build](https://github.com/vladiant/CascadeClassifier/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/vladiant/CascadeClassifier/actions/workflows/ubuntu.yml)
[![Windows](https://github.com/vladiant/CascadeClassifier/actions/workflows/windows.yml/badge.svg)](https://github.com/vladiant/CascadeClassifier/actions/workflows/windows.yml)
[![macOS](https://github.com/vladiant/CascadeClassifier/actions/workflows/macos.yml/badge.svg)](https://github.com/vladiant/CascadeClassifier/actions/workflows/macos.yml)
[![Static Check](https://github.com/vladiant/CascadeClassifier/actions/workflows/static_check.yml/badge.svg)](https://github.com/vladiant/CascadeClassifier/actions/workflows/static_check.yml)
[![CodeQL](https://github.com/vladiant/CascadeClassifier/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/vladiant/CascadeClassifier/actions/workflows/codeql-analysis.yml)
## About

Haar Cascade Classifier implementation, tools and docs.

This repository revives the classic Viola–Jones cascade trainer that used
to ship with OpenCV (the legacy `opencv_traincascade` program plus its
companion utilities) so it can keep building against modern OpenCV
releases. It contains:

- a stand-alone trainer library and CLI (`traincascade/`) that can train
  a multi-stage cascade with Haar, LBP or HOG features;
- the original sample-preparation, annotation, detection and
  visualisation utilities (`tools/`);
- documentation on the relevant command-line flags (`docs/`).

## Repository layout

| Path | Contents |
| ---- | -------- |
| `traincascade/`          | Cascade trainer library (`lib/`), `traincascade` executable and unit tests (`test/`). |
| `traincascade/lib/include/` | Public headers, fully Doxygen-annotated. |
| `traincascade/lib/src/`     | Implementation; the `o_*.cpp` files are extracted from OpenCV's legacy ML module and retain their original copyright headers. |
| `tools/createsamples/`   | `opencv_createsamples` — generates `.vec` files of positives. |
| `tools/annotation/`      | `opencv_annotation` — interactive bounding-box tool. |
| `tools/detection/Cpp/`   | C++ sample that runs a trained cascade on an image. |
| `tools/detection/Python/`| Equivalent Python detection sample. |
| `tools/visualisation/`   | `opencv_visualisation` — visualises the stages of a trained cascade. |
| `docs/`                  | Markdown documentation of CLI parameters. |
| `external/`              | CMake helpers (e.g. `FindOpenCV.cmake`). |

## Building

The project uses CMake (>= 3.10) and depends on OpenCV (core, imgproc,
imgcodecs, ml, objdetect, highgui).

```sh
cmake -S . -B build -G Ninja
cmake --build build
```

A coverage-enabled build is also available under `build-coverage/`.

## 10-minute end-to-end tutorial (real public dataset)

This walkthrough trains a small **face** cascade in a few minutes using:

- **LFW (Labeled Faces in the Wild)** for positives  
  http://vis-www.cs.umass.edu/lfw/
- **Caltech-101 `BACKGROUND_Google`** for negatives  
  https://data.caltech.edu/records/mzrjq-6wc02

The model is intentionally tiny (for speed), but the pipeline is the same
one used for larger production-quality runs.

> On a modern laptop CPU, the commands below typically finish in about
> 5-10 minutes.

```sh
# 1) Build the trainer + createsamples
cmake -S . -B build -G Ninja
cmake --build build --target traincascade createsamples

# 2) Prepare a quick workspace
mkdir -p quickstart/{downloads,data,positives,negatives,classifier}

# 3) Download public datasets
curl -L "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz" \
  -o quickstart/downloads/lfw-deepfunneled.tgz
curl -L "https://data.caltech.edu/records/mzrjq-6wc02/files/101_ObjectCategories.tar.gz?download=1" \
  -o quickstart/downloads/101_ObjectCategories.tar.gz

# 4) Unpack
tar -xzf quickstart/downloads/lfw-deepfunneled.tgz -C quickstart/data
tar -xzf quickstart/downloads/101_ObjectCategories.tar.gz -C quickstart/data

# 5) Build positive/negative file lists (small subset for speed)
find quickstart/data/lfw-deepfunneled -name "*.jpg" | sort | head -n 600 \
  > quickstart/positives/images.txt
find quickstart/data/101_ObjectCategories/BACKGROUND_Google -name "*.jpg" | sort | head -n 1200 \
  > quickstart/negatives/bg.txt

# LFW deepfunneled images are aligned 250x250 face crops.
# Treat each full crop as one positive object.
awk '{print $0 " 1 0 0 250 250"}' quickstart/positives/images.txt \
  > quickstart/positives/info.dat

# 6) Convert positives to .vec
./build/tools/createsamples/createsamples \
  -info quickstart/positives/info.dat \
  -vec quickstart/positives/lfw_24x24.vec \
  -num 550 \
  -w 24 -h 24

# 7) Train a compact cascade (LBP is much faster than HAAR)
./build/traincascade/traincascade \
  -data quickstart/classifier \
  -vec quickstart/positives/lfw_24x24.vec \
  -bg quickstart/negatives/bg.txt \
  -numPos 500 \
  -numNeg 1000 \
  -numStages 4 \
  -featureType LBP \
  -w 24 -h 24 \
  -minHitRate 0.995 \
  -maxFalseAlarmRate 0.5 \
  -precalcValBufSize 256 \
  -precalcIdxBufSize 256

# 8) Result
ls -lh quickstart/classifier/cascade.xml
```

Notes:

- If training aborts with "insufficient samples", reduce `-numPos` and
  `-numNeg` slightly.
- For higher accuracy, scale up: more positives/negatives, more stages,
  and optionally HAAR features.
- To test interactively, use `tools/detection/Cpp` or
  `tools/detection/Python/detect.py` with
  `quickstart/classifier/cascade.xml`.

## Architecture overview

The trainer is structured as a small hierarchy of legacy OpenCV ML
classes plus cascade-specific subclasses:

```
CvStatModel
  └── CvDTree                       (CART decision tree, o_cvdtree.h)
        └── CvBoostTree              (boosting weak learner, o_cvboostree.h)
              └── CvCascadeBoostTree (cascade-aware weak tree)

CvStatModel
  └── CvBoost                        (AdaBoost ensemble, o_cvboost.h)
        └── CvCascadeBoost           (single cascade stage, boost.h)

CvCascadeClassifier                  (multi-stage trainer, cascadeclassifier.h)
```

Feature evaluation is decoupled through `CvFeatureEvaluator`
(`traincascade_features.h`) with concrete subclasses
`CvHaarEvaluator`, `CvLBPEvaluator` and `CvHOGEvaluator`. Sample
streaming is handled by `CvCascadeImageReader` (`imagestorage.h`).
Refer to the Doxygen comments in `traincascade/lib/include/` for class-
and method-level documentation.

## Documentation

* [Cascade trainer parameters](docs/traincascade_params.md)
* [createsamples parameters](docs/createsamples_params.md)
* [Useful links and references](docs/links.md)
* [Test-suite README](traincascade/test/README.md)

## License

See [LICENSE](LICENSE). Files derived from OpenCV's legacy ML module
(`traincascade/lib/src/o_*.cpp` and the tools under `tools/`) keep their
original Intel / OpenCV Foundation copyright headers.
