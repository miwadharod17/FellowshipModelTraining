# Person B validation record

## Verified here

- All 35 required packs generated and written as little-endian float32 `.bin` plus array-offset `.json`.
- 35 pack metric JSONs, 35 pack golden input/output pairs, and 35 pack manifest entries produced.
- Pack write/read exact roundtrip, finite arrays, manifest binary SHA-256, expected experiment counts, and K-Means recorded-history shapes checked.
- Linear engine NumPy math checked against independent matrix multiplication; kernel reference compared with scikit-learn RBF and linear SVC; distance reference compared with scikit-learn nearest neighbors.
- L1 logistic penalty verified to produce sparse coefficients at small C (corrected for current scikit-learn `l1_ratio` API).
- All 35 pack golden outputs checked against packed float32 parameters and proposed NumPy inference interface. Maximum observed absolute difference after correction: < 2e-6.
- Detected and corrected numerical cancellation in (a) linear SVM large-coefficient accumulation (using a stored effective weight vector), (b) RBF pairwise distances on stretched features (direct coordinate differences), and (c) nearest-neighbor distances on Wine's mixed-scale features (per-feature squared-difference accumulation).
- Verified original `main` and Person C `miwa` ZIPs have identical shared `common/data.py`, `common/packs.py`, `common/manifest.py`, and saved moons/Wine splits.
- `python -m tests.test_person_b`: passed.
- `python -m compileall -q train engines tests`: passed.

## Blocked / not verified here

- TensorFlow package is not installed and network package installation failed due DNS/network restrictions. Therefore **0/6 TFLite engine binaries have been produced** and **0/6 TFLite interpreter parity checks have run**. The engine builder and tests are provided for execution in a TensorFlow-capable environment.
- The inherited `tests.test_data` cannot run here because the shared `common.seed` module imports TensorFlow at import time. `tests.test_packs` passes.
- Fresh-clone reproducibility in the project's pinned Python 3.12/TensorFlow environment and browser/Android runtime parity remain outstanding.

**Do not describe the assignment as fully complete until the six TFLite engines exist and pass the required parity and runtime tests.**

## Independent follow-up audit (21 November 2026)

See `PERSON_B_FINAL_AUDIT.md`. Four issues in the mixed-manifest / engine-builder workflow were corrected, and the regression tests, checksums and metric schema were rechecked. **Final approval is withheld** pending actual TFLite binaries and interpreter/browser/Android parity.
