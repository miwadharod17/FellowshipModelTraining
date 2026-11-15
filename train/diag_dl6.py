"""Diagnostic for the DL-6 fp16 parity failure. Does NOT change any files.
Shows, for each output of the .tflite, how big the activations are and how far
the fp16 model drifts from the Keras fp32 model (absolute, relative, cosine similarity).

Run from the repo root, after train_dl6 has produced the .tflite:
    python -m train.diag_dl6
"""
import argparse
import glob
import json

import numpy as np
import tensorflow as tf

from train.train_dl6 import build, load_image, SAMPLES_DIR, NAME, LABELS_OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--random-weights', action='store_true', help='testing only')
    args = ap.parse_args()

    files = sorted(glob.glob(f'{SAMPLES_DIR}/*.jpg') + glob.glob(f'{SAMPLES_DIR}/*.jpeg')
                   + glob.glob(f'{SAMPLES_DIR}/*.png'))
    x = np.stack([load_image(f) for f in files]).astype(np.float32)
    labels = json.load(open(LABELS_OUT))

    model = build(args.random_weights)
    name_by_shape = {tuple(t.shape[1:]): n for n, t in zip(model.output_names, model.outputs)}

    it = tf.lite.Interpreter(model_path=f'out/models/{NAME}.tflite')
    it.allocate_tensors()
    inp = it.get_input_details()[0]['index']
    outs = it.get_output_details()

    stats = {}
    top1 = []
    for i in range(len(x)):
        ref = model(x[i:i + 1], training=False)
        ref_by_shape = {tuple(r.shape[1:]): np.asarray(r) for r in ref}
        it.set_tensor(inp, x[i:i + 1])
        it.invoke()
        for d in outs:
            got = it.get_tensor(d['index'])
            r = ref_by_shape[tuple(got.shape[1:])]
            n = name_by_shape[tuple(got.shape[1:])]
            diff = np.abs(got - r)
            cos = float(got.ravel() @ r.ravel() / (np.linalg.norm(got) * np.linalg.norm(r) + 1e-12))
            s = stats.setdefault(n, {'max_ref': 0.0, 'abs': 0.0, 'rel': 0.0, 'mean_rel': 0.0, 'cos': 1.0})
            s['max_ref'] = max(s['max_ref'], float(np.abs(r).max()))
            s['abs'] = max(s['abs'], float(diff.max()))
            s['rel'] = max(s['rel'], float(diff.max() / (np.abs(r).max() + 1e-12)))
            s['mean_rel'] = max(s['mean_rel'], float(diff.mean() / (np.abs(r).mean() + 1e-12)))
            s['cos'] = min(s['cos'], cos)
            if n == 'predictions':
                top1.append((files[i].split('/')[-1], labels[int(got[0].argmax())],
                             float(got[0].max())))

    print('\nWorst case over the', len(x), 'images')
    print(f"{'output':24s} {'max|value|':>10s} {'max abs diff':>13s} {'max diff/max':>13s} {'mean diff/mean':>15s} {'min cosine':>11s}")
    for n, s in stats.items():
        print(f"{n:24s} {s['max_ref']:10.2f} {s['abs']:13.4f} {s['rel']:13.4%} {s['mean_rel']:15.4%} {s['cos']:11.6f}")
    print('\nTop-1 prediction for each image (check these make sense):')
    for f, lab, p in top1:
        print(f'  {f:30s} {lab:25s} {p:.2f}')


if __name__ == '__main__':
    main()
