# Person B independent audit — 21 November 2026

## Decision: CONDITIONAL / NOT FINAL-APPROVED

This package is a corrected, independently reviewed **partial** handoff. The source and 35 packs pass the checks listed below, but six mandatory TensorFlow Lite engines are absent. Consequently the guide's definition of done is not met and browser/Android runtime parity cannot be certified.

## Source of requirements

`Virtual_Lab_Model_Training_Guide_parallel.docx`, especially §§2.8–2.11, 4.1, 4.3, 4.4, 4.7, 4.8, 4.10, 4.11, 4.13, 5.2. The original `FellowshipModelTraining-main.zip` and the `FellowshipModelTraining-miwa.zip` were used for comparison. Person B's source was edited only in this review copy.

## Verified in this audit

- Counted 35 required pack manifest entries: ML-2=2, ML-3=5, ML-8=12, ML-9=14, ML-11 stacking=2.
- Independently verified the SHA-256 hashes of all 35 pack `.bin` files and all 35 `.json` indexes against their manifest entries.
- Verified all 35 metric JSONs contain `name`, `date`, `seed`, `dataset`, and `settings`.
- Re-ran `python -m tests.test_person_b` and `python -m tests.test_packs`: PASS.
- Re-ran Python compile checks on `train/`, `engines/`, and `tests/`: PASS.
- Tested mixed pack + engine manifest behavior using a temporary synthetic engine entry (restored the original manifest afterwards): PASS after fixes.
- Verified there are **zero** `.tflite` files in `out/models/`.

## Defects discovered and corrected

1. `tests/test_person_b_tflite.py` iterated over all manifest entries, including engine entries, and would try to read `.tflite` files as packs after conversion. It now filters `type == 'pack'`.
2. `tests/test_person_b.py` assumed the manifest always contained exactly 35 total entries and would fail after adding the six engines. Its pack checks now filter `type == 'pack'`.
3. The guide explicitly names individual engine builder entry points. Added `engines/build_engine_linear.py`, `engines/build_engine_kernel.py`, and `engines/build_engine_distance.py` as wrappers around the shared builder.
4. The shared engine builder previously discarded existing engine manifest entries whenever an individual engine was rebuilt. It now preserves unaffected engine entries while replacing only the engines being rebuilt.

## Remaining mandatory gates

1. Install the pinned environment (Python 3.12 and TensorFlow 2.21.0 per repository requirements) in a fresh clone.
2. Run all three engine builders; produce `engine_{linear,kernel,distance}_b{1,2500}.tflite` (six real binaries).
3. Run TensorFlow Lite Interpreter parity tests for both batch sizes, checking tensor signatures and all applicable outputs, and generate engine metrics/goldens/manifest entries.
4. Verify that the generated engines use supported built-in TFLite operations and run in browser WASM/WebGPU and Android.
5. Obtain developer signoff on the **proposed** E1/E3/E4 tensor interface and capacities before integration.
6. Run training and parity from a fresh clone and verify reproducibility under the pinned versions.
7. Merge manifests with Persons A and C only after all team artifacts are present; do not overwrite their parts.

## Important qualification

The tests above establish source-level and NumPy/pack consistency. They do not prove successful TensorFlow conversion, end-to-end runtime behavior, or parity on clients. Do not label this ZIP a fully completed Person B submission until the remaining gates pass.
