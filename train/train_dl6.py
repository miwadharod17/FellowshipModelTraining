"""DL-6 CNN Feature Map Visualization: no training. Takes pretrained MobileNetV2,
adds 5 intermediate layers as extra outputs, converts to fp16 .tflite, runs parity,
writes labels, metrics, golden files and the manifest entry.

Before running, put 20 sample images (jpg/png, free-to-use licences) in
    out/precomputed/dl6_samples/
and write their sources/licences in
    out/precomputed/dl6_samples_credits.txt

Run from the repo root:
    python -m train.train_dl6                      # strict parity, fp16 (guide default: 1e-2 absolute)
    python -m train.train_dl6 --parity relative    # fp16, relative/cosine parity (needs lead's OK)
    python -m train.train_dl6 --fp32               # fp32 model named dl6_mobilenetv2_fp32, strict 1e-4
"""
import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from common.seed import set_seed
from common.convert import keras_to_tflite
from common.metrics import write_metrics
from common.parity import parity_and_golden, run_tflite_keras
from common.manifest import make_entry, add_entry

NAME = 'dl6_mobilenetv2_fp16'
NAME_FP32 = 'dl6_mobilenetv2_fp32'
# criteria used ONLY with --parity relative (needs your team lead's OK):
REL_COSINE_MIN = 0.999     # cosine similarity of every output, Keras fp32 vs tflite
REL_MEAN_ERR_MAX = 0.05    # mean |diff| / mean |reference| of every feature map
REL_PRED_ABS_MAX = 0.02    # max |diff| of the class probabilities
SAMPLES_DIR = 'out/precomputed/dl6_samples'
CREDITS = 'out/precomputed/dl6_samples_credits.txt'
LABELS_OUT = 'out/precomputed/dl6_imagenet_labels.json'
MANIFEST_PART = 'out/manifest_c.json'
# feature maps from early (edges) to late (object parts)
TAPS = ['Conv1_relu', 'block_2_add', 'block_5_add', 'block_9_add', 'block_16_project_BN']
LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'


def load_image(path):
    """Shorter side resized to 224, centre crop 224x224, RGB, values 0-255 (float32)."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    return np.asarray(img, dtype=np.float32)


def get_labels(random_weights):
    if random_weights:
        return [f'class_{i}' for i in range(1000)]
    p = tf.keras.utils.get_file('imagenet_class_index.json', LABELS_URL, cache_subdir='models')
    d = json.load(open(p))
    return [d[str(i)][1] for i in range(1000)]


def build(random_weights=False):
    """Input: 224x224 RGB pixels 0-255 (float32). Rescaling to -1..1 is done inside the model."""
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), weights=None if random_weights else 'imagenet')
    feat = keras.Model(base.input, [base.output] + [base.get_layer(n).output for n in TAPS])
    inp = keras.Input(shape=(224, 224, 3), batch_size=1, name='image')
    x = keras.layers.Rescaling(1 / 127.5, offset=-1.0, name='pixels_to_minus1_1')(inp)
    outs = feat(x)
    named = [keras.layers.Identity(name='predictions')(outs[0])]
    named += [keras.layers.Identity(name=f'fmap_{n}')(o) for n, o in zip(TAPS, outs[1:])]
    return keras.Model(inp, named, name=NAME)


def tflite_io(path):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    ins = [{'name': d['name'], 'shape': [int(s) for s in d['shape']], 'dtype': 'float32',
            'range': [0.0, 255.0]} for d in it.get_input_details()]
    outs = [{'name': d['name'], 'shape': [int(s) for s in d['shape']], 'dtype': 'float32'}
            for d in it.get_output_details()]
    return ins, outs


def relative_parity(model, path, x20, name, outs, golden_dir='out/golden'):
    """fp16 parity by relative error and cosine similarity instead of absolute difference.
    Also saves the golden files in the same format as common.parity.parity_and_golden."""
    ok, golden = True, [[] for _ in outs]
    worst = {}
    for i in range(len(x20)):
        xi = x20[i:i + 1]
        ref = model(xi, training=False)
        got = run_tflite_keras(path, xi)
        for k, (r, g, o) in enumerate(zip(ref, got, outs)):
            r = np.asarray(r)
            diff = np.abs(r - g)
            nr, ng = np.linalg.norm(r), np.linalg.norm(g)
            cos = 1.0 if (nr < 1e-8 and ng < 1e-8) else float(r.ravel() @ g.ravel() / (nr * ng + 1e-12))
            mean_err = float(diff.mean() / (np.abs(r).mean() + 1e-12))
            w = worst.setdefault(k, {'cos': 1.0, 'mean_err': 0.0, 'abs': 0.0})
            w['cos'], w['mean_err'], w['abs'] = (min(w['cos'], cos), max(w['mean_err'], mean_err),
                                                 max(w['abs'], float(diff.max())))
            golden[k].append(g)
        # (golden outputs are the .tflite outputs, like the strict test)
    for k, o in enumerate(outs):
        w = worst[k]
        good = w['cos'] >= REL_COSINE_MIN
        if o['role'] == 'predictions':
            good &= w['abs'] <= REL_PRED_ABS_MAX
        else:
            good &= w['mean_err'] <= REL_MEAN_ERR_MAX
        ok &= bool(good)
        print(f"{name} output {k} {o['role']:12s} {str(o['shape']):20s} min cosine {w['cos']:.5f} "
              f"mean rel err {w['mean_err']:.4f} max abs {w['abs']:.4f} {'PASS' if good else 'FAIL'}")
    np.save(f'{golden_dir}/{name}_golden_in.npy', x20.astype(np.float32))
    for k in range(len(outs)):
        np.save(f'{golden_dir}/{name}_golden_out_{k}.npy', np.concatenate(golden[k]))
    return bool(ok), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--random-weights', action='store_true',
                    help='testing only: skip the ImageNet weights (results are meaningless)')
    ap.add_argument('--parity', choices=['strict', 'relative'], default='strict')
    ap.add_argument('--fp32', action='store_true', help='ship fp32 instead of fp16 (named dl6_mobilenetv2_fp32)')
    args = ap.parse_args()
    name = NAME_FP32 if args.fp32 else NAME
    fp16 = not args.fp32

    assert os.path.isdir('out/golden'), 'Run this from the repo root (folder containing out/).'
    files = sorted(glob.glob(f'{SAMPLES_DIR}/*.jpg') + glob.glob(f'{SAMPLES_DIR}/*.jpeg')
                   + glob.glob(f'{SAMPLES_DIR}/*.png'))
    assert len(files) == 20, f'Need exactly 20 images in {SAMPLES_DIR}, found {len(files)}.'
    if not os.path.exists(CREDITS):
        print(f'WARNING: {CREDITS} is missing. Write the source and licence of each image there.')

    set_seed()
    labels = get_labels(args.random_weights)
    json.dump(labels, open(LABELS_OUT, 'w'))

    model = build(args.random_weights)
    x20 = np.stack([load_image(f) for f in files]).astype(np.float32)

    path = f'out/models/{name}.tflite'
    keras_to_tflite(model, path, fp16=fp16)
    print('tflite size (MB):', round(os.path.getsize(path) / 1e6, 2))

    ins, outs = tflite_io(path)
    # match Keras outputs to the .tflite order by shape (all shapes are different)
    by_shape = {tuple(t.shape[1:]): t for t in model.outputs}
    ordered = keras.Model(model.input, [by_shape[tuple(o['shape'][1:])] for o in outs])
    for o in outs:
        o['role'] = 'predictions' if o['shape'][-1] == 1000 else 'feature_map'
    rel_worst = None
    if args.parity == 'relative' and fp16:
        ok, rel_worst = relative_parity(ordered, path, x20, name, outs)
    else:
        ok = parity_and_golden(ordered, path, x20, name, fp16=fp16)

    # label agreement on the predictions output
    pi = [i for i, o in enumerate(outs) if o['role'] == 'predictions'][0]
    agree, top5 = 0, []
    for i in range(20):
        ref = np.asarray(model(x20[i:i + 1], training=False)[0])[0]
        got = run_tflite_keras(path, x20[i:i + 1])[pi][0]
        agree += int(np.argmax(ref) == np.argmax(got))
        top5.append({'image': os.path.basename(files[i]),
                     'top5': [{'label': labels[j], 'p': round(float(got[j]), 4)}
                              for j in np.argsort(got)[::-1][:5]]})
    frac = agree / 20
    print('top-1 label agreement Keras vs tflite:', frac)
    ok = bool(ok and frac >= 0.999)

    write_metrics(name, 'none (pretrained ImageNet MobileNetV2, 20 sample images for testing)',
                  {'model': 'MobileNetV2 alpha=1.0 224x224', 'fp16': fp16, 'parity_mode': args.parity if fp16 else 'strict', 'taps': TAPS,
                   'tap_shapes': {n: list(model.get_layer(f'fmap_{n}').output.shape[1:]) for n in TAPS},
                   'size_mb': round(os.path.getsize(path) / 1e6, 2),
                   'random_weights': args.random_weights},
                  sample_top5=top5, top1_agreement=frac, parity_pass=ok,
                  relative_parity_worst=rel_worst,
                  relative_parity_criteria={'cosine_min': REL_COSINE_MIN, 'feature_map_mean_rel_err_max': REL_MEAN_ERR_MAX,
                                            'predictions_abs_max': REL_PRED_ABS_MAX} if rel_worst else None)

    if ok:
        e = make_entry(name, path, 'model', ins, outs,
                       'input is a 224x224 RGB image, pixel values 0 to 255 (shorter side resized to 224, '
                       'centre crop). Rescaling to -1..1 is done inside the model.',
                       labels=labels)
        if rel_worst:
            e['parity'] = ('fp16 vs fp32 Keras: for each output cosine similarity >= %s; feature maps mean |diff|/mean |ref| <= %s; '
                           'predictions max abs diff <= %s. Use the same rule with the golden files.'
                           % (REL_COSINE_MIN, REL_MEAN_ERR_MAX, REL_PRED_ABS_MAX))
        add_entry(e, MANIFEST_PART)
    print('\nRESULT:', 'PASS' if ok else 'FAIL', '|', path)


if __name__ == '__main__':
    main()
