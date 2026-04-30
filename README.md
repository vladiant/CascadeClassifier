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
