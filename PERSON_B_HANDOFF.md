# Person B — Virtual Lab Model Training handoff

**Source:** `Virtual_Lab_Model_Training_Guide_parallel.docx`, sections 2, 4.1, 4.3, 4.4, 4.7, 4.8, 4.10, 4.11, 4.13 and 5.2. Base: `FellowshipModelTraining-main.zip`. The Person C `miwa` ZIP was used only to verify that shared utilities and saved splits match; Person C files have not been modified.

## Delivery status (do not omit)

- **Generated and tested:** all 35 required Person B packs (`.bin` + `.json`), 35 per-pack metrics files, 35 golden input/output pairs, `out/manifest_b.json`, KNN precomputed benchmark, fixed-split files for Breast Cancer, separable blobs and overlapping blobs, training scripts and NumPy engine references.
- **Engine source implemented, but NOT built or verified here:** six `.tflite` files (`linear`, `kernel`, `distance` at batch 1 and 2500). TensorFlow is absent from the execution environment and the package index cannot be reached. There are no fake `.tflite` files in this handoff. Run the engine build/parity steps below in a TensorFlow-capable environment before marking the assignment complete.
- **Proposed interface, not developer-approved:** engine input/output signatures below. The guide calls the input names a developer contract but does not actually list those names. The dev team must confirm this proposed contract **before integration**.
- **Browser/Android parity NOT tested:** those tests must be run by the app teams with the generated engines and goldens.

## File map

| Files | Purpose |
|---|---|
| `train/train_ml2.py` | 2 KNN packs + scaling/noise curves + precomputed kd-tree/ball-tree/brute-force benchmark |
| `train/train_ml3.py` | 5 K-Means packs, 2–8 clusters, 2 seeding methods, 10 restarts, max 30 recorded updates |
| `train/train_ml8.py` | 8 Breast Cancer and 4 separable logistic packs, train-only standardization, L1/L2 saga |
| `train/train_ml9.py` | 9 RBF + 3 linear + 2 scaling SVC packs |
| `train/train_ml11_stacking.py` | 2 stacking packs (out-of-fold and deliberately in-fold) |
| `train/train_person_b.py` | Regenerate all 35 packs and manifest part |
| `engines/person_b_contract.py` | Fixed proposed tensor shapes, names and capacities |
| `engines/person_b_numpy.py` | Independently testable NumPy engine reference |
| `engines/build_engine_person_b.py` | Convert three engines × two batch sizes to TFLite, test interpreter and save goldens |
| `tests/test_person_b.py` | Pack integrity, real scikit-learn vs NumPy and golden tests |
| `tests/test_person_b_tflite.py` | Interpreter-vs-pack golden tests; run after conversion |
| `out/manifest_b.json` | Person B only; do not overwrite Person A/C manifests |

## Build instructions — macOS/Linux, from repository root

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m tests.test_packs
python -m tests.test_data
python -m train.train_person_b
python -m tests.test_person_b
python -m engines.build_engine_person_b
python -m tests.test_person_b_tflite
```

Run `python -m common.manifest` **only after** `manifest_a.json` and `manifest_c.json` have been merged into the same repo. Never replace another person's manifest part.

### Proposed tensor interface, version 1

All tensors are float32; indices and counts are encoded as float32. Fixed `batch` is 1 or 2500. The engine output is a named dictionary (TFLite signature runner).

**E1 Linear**: `features[batch,32]`, `weights[32,4]`, `bias[4]`, `feature_mask[32]`, `class_mask[4]` → `scores[batch,4]`. Apps apply sigmoid, softmax or exponential. Pad unused feature/class slots with zeros and use masks. The class-mask suppresses unused output slots with zero (not `-inf`); apps must select active classes before softmax.

**E3 Kernel**: `features[batch,2]`, `support_vectors[256,2]`, `dual_coef[256]`, `bias[1]`, `gamma[1]`, `support_count[1]`, `kernel_code[1]` → `decision[batch,1]`. Kernel code 0=linear, 1=RBF; padded supports ignored using count. The original SVM support vectors/alphas are in each pack for display. For linear SVM, use `engine_support_vectors` and `engine_dual_coef` if present; this reduces float32 cancellation error without changing the learned decision function.

**E4 Distance**: `features[batch,13]`, `reference_points[1000,13]`, `feature_weights[13]`, `reference_count[1]`, `k[1]` → `distances[batch,30]`, `indices[batch,30]`. `feature_weights` are nonnegative weights on **squared** per-feature differences. The first k entries are meaningful; remaining distances are 1e15 and indices -1. `k` must be 1..min(30,reference_count). For K-Means, set reference points to the selected recorded centroids and `k=1` to assign nearest centroid. For KNN, the app performs label voting using indices and the pack's `reference_labels`; exact-zero distance needs explicit handling in inverse-distance weighting.

The `.bin` file is little-endian float32, with arrays concatenated in the order listed in the `.json` index; offsets count **float32 elements**, not bytes. Never silently truncate a pack that exceeds an engine capacity. All currently generated Person B packs fit these proposed capacities.

### Preprocessing and pack-specific notes

- ML-2: `reference_points` contain the fixed stretched/noise-augmented data or Wine data. Use `train_mean` and `train_scale` to standardize if the scaling toggle is enabled. For the moons pack, noise-feature slider uses the first 0/2/5/10 fixed noise columns. Distances use squared feature weights, then sqrt; voting is performed in the app.
- ML-3: `centroid_history` is `[7,2,10,30,8,2]`, for K=2..8, seed method random/k-means++, 10 restarts, up to 30 updates, padded by repeating the final centres. `objective_history` and `iteration_counts` identify valid updates. To select the best of the first r restarts, minimize the final objective over the first r entries. Recorded trajectories are **precomputed**, not live training.
- ML-8: `coef_scaled` and `intercept_scaled` are engine-ready when features are standardized; `coef_original` and `intercept_original` are for original-unit interpretation. Class 1 in Breast Cancer is **benign** (sklearn's dataset encoding), class 0 malignant. `test_probs` are held-out positive-class probabilities.
- ML-9: SVC dual coefficients already include signed `alpha*y`. `support_indices`, `alphas`, `free_mask`, `bounded_mask` support the support-vector visualization. `platt_A` and `platt_B` are saved for calibration; scikit-learn's probability calibration conventions must be respected. Scaling packs have the transform metadata and arrays.
- ML-11: base learner meta-features are probabilities in this exact order: KNN(1), SVC(C=100,gamma=10), RF(200 trees, max depth 12, max_features 2). The in-fold pack is intentionally optimistic and is for a leakage lesson, not a recommended production model. Both use the **same held-out** split.

## Verification checklist

1. Re-run all training scripts from a clean checkout; compare names/counts and verify manifest SHA-256 for every `.bin` and `.json`.
2. Run `tests.test_person_b` (NumPy and scikit-learn, pack roundtrips, golden parity).
3. Generate the six TFLite binaries using TensorFlow, check named signatures and built-in ops only, then run `tests.test_person_b_tflite` (1e-4 max absolute difference to golden outputs).
4. Inspect output tensors in LiteRT.js WASM/WebGPU and Android using the same `.npz` engine goldens and `.npy` pack goldens; test **both** batch sizes.
5. Have dev team approve the proposed tensor interface and fixed capacities before merging; engine input names cannot be changed after integration without updating clients.

## Limitations / not yet validated

- TensorFlow TFLite conversion, interpreter parity, browser WASM/WebGPU, and Android parity are **pending**. Source-level NumPy parity does not prove conversion will work.
- `requirements.txt` is inherited unchanged from the uploaded repository; it pins TensorFlow 2.21.0 and the repository README targets Python 3.12. Local pack generation here used the installed scikit-learn environment; re-run in the pinned environment before merging.
- The benchmark is **precomputed and machine-dependent**; do not present its timings as live client performance.
- Golden `.npy` for packs contains 20 inputs; batch-2500 engine goldens are fixed-size `.npz` with the first 20 rows used for direct parity and the remainder filled to the required shape.
