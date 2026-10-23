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

This walkthrough trains a small **face** cascade in a few minutes using
the public **Caltech-101** dataset:

- positives from `Faces_easy`
- negatives from `BACKGROUND_Google`
- dataset source: https://data.caltech.edu/records/mzrjq-6wc02

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

# 3) Download and unpack Caltech-101
curl -L "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1" \
  -o quickstart/downloads/caltech-101.zip
unzip -q quickstart/downloads/caltech-101.zip -d quickstart/data
tar -xzf quickstart/data/caltech-101/101_ObjectCategories.tar.gz \
  -C quickstart/data/caltech-101

# 4) Build positive/negative lists
# NOTE: createsamples resolves image paths relative to info.dat location,
# so positives are written as ../data/... relative paths.
find quickstart/data/caltech-101/101_ObjectCategories/Faces_easy -name "*.jpg" | sort | head -n 400 \
  | sed 's#^quickstart/#../#' > quickstart/positives/images_rel.txt

# NOTE: traincascade resolves bg entries as regular paths, so we store
# absolute paths for negatives.
find "$(pwd)/quickstart/data/caltech-101/101_ObjectCategories/BACKGROUND_Google" -name "*.jpg" | sort \
  > quickstart/negatives/bg_abs.txt

# 5) Create info.dat (one object covering full image dimensions)
python3 - <<'PY'
from pathlib import Path
from PIL import Image

images = [line.strip() for line in Path("quickstart/positives/images_rel.txt").read_text().splitlines() if line.strip()]
with Path("quickstart/positives/info.dat").open("w") as out:
    for rel_path in images:
        full_path = (Path("quickstart/positives") / rel_path).resolve()
        with Image.open(full_path) as im:
            w, h = im.size
        out.write(f"{rel_path} 1 0 0 {w} {h}\n")
PY

# 6) Convert positives to .vec
./build/tools/createsamples/createsamples \
  -info quickstart/positives/info.dat \
  -vec quickstart/positives/faces_easy_24x24.vec \
  -num 380 \
  -w 24 -h 24

# 7) Train a compact cascade (LBP is much faster than HAAR)
./build/traincascade/traincascade \
  -data quickstart/classifier \
  -vec quickstart/positives/faces_easy_24x24.vec \
  -bg quickstart/negatives/bg_abs.txt \
  -numPos 350 \
  -numNeg 400 \
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
