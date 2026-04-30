# Unit tests for the TrainCascadeLib

These tests exercise the `TrainCascadeLib` static library directly,
without going through the `traincascade` command-line driver. Each file
focuses on one component of the library and follows an
Arrange / Act / Assert structure with comments per step.

## Test framework: doctest
* https://github.com/onqtam/doctest
* commit b7c21ec5ceeadb4951b00396fc1e4642dd347e5f
* 2.4.9

The `doctest/doctest.h` header is vendored under `doctest/`. A single
translation unit (`main.cpp`) defines `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`
so each test executable links against one copy of the framework.

## Test files

| File | Component under test |
| ---- | -------------------- |
| [test_dtree.cpp](test_dtree.cpp)            | `CvDTree` and `CvDTreeTrainData` — direct decision-tree training, prediction, regression mode, pruning and sample-mask preprocessing. |
| [test_features.cpp](test_features.cpp)      | Haar / LBP / HOG feature evaluators (`CvHaarEvaluator`, `CvLBPEvaluator`, `CvHOGEvaluator`). |
| [test_imagestorage.cpp](test_imagestorage.cpp) | Positive/negative sample streaming via `CvCascadeImageReader`. |
| [test_o_utils.cpp](test_o_utils.cpp)        | Small utilities from `o_utils.h` (alignment, comparators, index preprocessing). |
| [test_params.cpp](test_params.cpp)          | Parameter serialisation and CLI scanning for `CvCascadeParams`, `CvCascadeBoostParams`, feature-family params. |
| [test_serialization.cpp](test_serialization.cpp) | Round-trip persistence of trees and stages through `cv::FileStorage`. |
| [test_integration.cpp](test_integration.cpp) | End-to-end smoke test: a small cascade is trained and reloaded. |
| [main.cpp](main.cpp)                        | doctest entry point. |

## Running the tests

The tests are picked up automatically by CTest:

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build/traincascade --output-on-failure
```

A coverage-enabled configuration lives under `build-coverage/`; HTML
reports land in `build-coverage/coverage-html/`.
