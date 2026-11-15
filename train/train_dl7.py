"""DL-7 CNN Image Classification: trains 4 models
    dl7_digit_c1 / c2 / c3  (1, 2, 3 conv layers, MNIST)
    dl7_shape_c2            (2 conv layers, generated circle/square/triangle images)
Each .tflite outputs the class probabilities plus the feature maps of every conv layer.

Run from the repo root:
    python -m train.train_dl7                    # all 4 models
    python -m train.train_dl7 --only digit_c2 --epochs 1    # quick test

Optional: put your 300 hand-drawn digits in data/raw/handdrawn_digits.npz with arrays
    x  (N, 28, 28) float values 0-1, white digit on black like MNIST
    y  (N,) integer labels 0-9
They are used for evaluation only and reported in each digit model's metrics file.
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

from common.seed import SEED, set_seed
from common.data import mnist, shapes, SHAPE_CLASSES
from common.convert import keras_to_tflite
from common.metrics import write_metrics
from common.parity import parity_and_golden
from common.manifest import make_entry, add_entry

FILTERS = [8, 16, 32]
BATCH = 128
MANIFEST_PART = 'out/manifest_c.json'
HANDDRAWN = 'data/raw/handdrawn_digits.npz'
CONFIGS = [('dl7_digit_c1', 'digit', 1), ('dl7_digit_c2', 'digit', 2),
           ('dl7_digit_c3', 'digit', 3), ('dl7_shape_c2', 'shape', 2)]


# ---------------------------------------------------------------- model
def build(n_conv, n_classes, batch_size=None):
    """Input 28x28x1 image, values 0-1. Outputs: probabilities, then one feature map per conv layer."""
    inp = keras.Input(shape=(28, 28, 1), batch_size=batch_size, name='image')
    x, taps = inp, []
    for i in range(n_conv):
        x = keras.layers.Conv2D(FILTERS[i], 3, padding='same', activation='relu',
                                name=f'conv{i + 1}')(x)
        taps.append(x)
        x = keras.layers.MaxPooling2D(2, name=f'pool{i + 1}')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation='relu', name='dense_hidden')(x)
    p = keras.layers.Dense(n_classes, activation='softmax', name='class_probs')(x)
    outs = [keras.layers.Identity(name='probabilities')(p)]
    outs += [keras.layers.Identity(name=f'fmap_conv{i + 1}')(t) for i, t in enumerate(taps)]
    return keras.Model(inp, outs, name=f'cnn_c{n_conv}_k{n_classes}')


# ---------------------------------------------------------------- data
def get_data(kind):
    """Returns (x_tr, y_tr), (x_val, y_val), (x_te, y_te), class_names. Images are (N,28,28,1)."""
    if kind == 'digit':
        (xtr, ytr), (xv, yv), (xte, yte) = mnist()
        names = [str(i) for i in range(10)]
    else:
        X, y = shapes(6000)                       # training set
        xtr, ytr, xv, yv = X[:5400], y[:5400], X[5400:], y[5400:]   # last 10% = validation
        xte, yte = shapes(500, seed=7)            # separate test set
        names = list(SHAPE_CLASSES)
    ex = lambda a: a[..., None].astype(np.float32)
    return (ex(xtr), ytr), (ex(xv), yv), (ex(xte), yte), names


def label_agreement(path, model, x, n_classes):
    """Fraction of images where Keras and .tflite predict the same class (batch size 1)."""
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    inp = it.get_input_details()[0]['index']
    outs = it.get_output_details()
    pi = [i for i, d in enumerate(outs) if len(d['shape']) == 2 and d['shape'][-1] == n_classes][0]
    ref = model.predict(x, batch_size=500, verbose=0)[0].argmax(1)
    got = []
    for i in range(len(x)):
        it.set_tensor(inp, x[i:i + 1])
        it.invoke()
        got.append(int(it.get_tensor(outs[pi]['index'])[0].argmax()))
    return float(np.mean(ref == np.array(got)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help="e.g. digit_c2 or shape")
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--no-aug', action='store_true', help='turn off shift/rotation augmentation')
    args = ap.parse_args()
    assert os.path.isdir('out/golden'), 'Run this from the repo root (folder containing out/).'

    hand = None
    if os.path.exists(HANDDRAWN):
        d = np.load(HANDDRAWN)
        hand = (d['x'].reshape(-1, 28, 28, 1).astype(np.float32), d['y'].astype(int))
        print('hand-drawn digits loaded:', len(hand[1]))
    else:
        print(f'NOTE: {HANDDRAWN} not found, skipping hand-drawn evaluation.')

    cache, summary = {}, []
    for name, kind, n_conv in CONFIGS:
        if args.only and args.only not in name:
            continue
        print(f'\n===== {name} =====')
        if kind not in cache:
            cache[kind] = get_data(kind)
        (xtr, ytr), (xv, yv), (xte, yte), names = cache[kind]
        k = len(names)

        set_seed()
        model = build(n_conv, k)
        # training wrapper: random shift/rotation (only active while training), shares weights
        tin = keras.Input(shape=(28, 28, 1))
        a = tin
        if not args.no_aug:
            a = keras.layers.RandomTranslation(0.1, 0.1, fill_mode='constant')(a)
            a = keras.layers.RandomRotation(0.06, fill_mode='constant')(a)
        train_model = keras.Model(tin, model(a)[0])
        train_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        hist = train_model.fit(xtr, ytr, batch_size=BATCH, epochs=args.epochs,
                               validation_data=(xv, yv), verbose=2)

        probs = model.predict(xte, batch_size=500, verbose=0)[0]
        pred = probs.argmax(1)
        acc = float(np.mean(pred == yte))
        cm = confusion_matrix(yte, pred, labels=list(range(k)))
        per_class = (cm.diagonal() / cm.sum(1)).tolist()
        print(f'{name} test accuracy {acc:.4f}')

        hand_acc = None
        if hand is not None and kind == 'digit':
            hand_acc = float(np.mean(model.predict(hand[0], verbose=0)[0].argmax(1) == hand[1]))
            print(f'{name} hand-drawn accuracy {hand_acc:.4f}')

        # export with batch size 1
        export = build(n_conv, k, batch_size=1)
        export.set_weights(model.get_weights())
        path = f'out/models/{name}.tflite'
        keras_to_tflite(export, path)

        it = tf.lite.Interpreter(model_path=path)
        it.allocate_tensors()
        ins = [{'name': d['name'], 'shape': [int(s) for s in d['shape']], 'dtype': 'float32',
                'range': [0.0, 1.0]} for d in it.get_input_details()]
        outs = [{'name': d['name'], 'shape': [int(s) for s in d['shape']], 'dtype': 'float32',
                 'role': 'probabilities' if len(d['shape']) == 2 else 'feature_map'}
                for d in it.get_output_details()]

        # parity: compare Keras and .tflite in the .tflite output order (outputs have different shapes)
        by_shape = {tuple(t.shape[1:]): t for t in export.outputs}
        ordered = keras.Model(export.input, [by_shape[tuple(o['shape'][1:])] for o in outs])
        x20 = xte[:20]
        ok = parity_and_golden(ordered, path, x20, name)
        agree = label_agreement(path, model, xte[:1000], k)
        print(f'{name} label agreement over {min(1000, len(xte))} test images: {agree}')
        ok = bool(ok and agree >= 0.999)

        write_metrics(
            name, 'mnist 55000/5000/10000' if kind == 'digit' else 'shapes(6000) train, shapes(500, seed=7) test',
            {'conv_layers': n_conv, 'filters': FILTERS[:n_conv], 'classes': names,
             'epochs': args.epochs, 'batch_size': BATCH, 'optimizer': 'adam 1e-3',
             'augmentation': not args.no_aug, 'params': int(model.count_params())},
            loss_curve=[float(v) for v in hist.history['loss']],
            val_loss_curve=[float(v) for v in hist.history['val_loss']],
            acc_curve=[float(v) for v in hist.history['accuracy']],
            val_acc_curve=[float(v) for v in hist.history['val_accuracy']],
            test_accuracy=acc, per_class_accuracy=per_class,
            confusion_matrix=cm.tolist(), handdrawn_accuracy=hand_acc,
            label_agreement=agree, parity_pass=ok)

        if ok:
            e = make_entry(name, path, 'model', ins, outs,
                           'none inside the model: input is a 28x28 grayscale image (white digit or shape on '
                           'black), shape [1,28,28,1], float32 values already scaled 0 to 1',
                           labels=names)
            add_entry(e, MANIFEST_PART)
        summary.append((name, ok, acc, hand_acc))

    print('\n================ SUMMARY ================')
    for name, ok, acc, hand_acc in summary:
        h = f'  handdrawn={hand_acc:.4f}' if hand_acc is not None else ''
        print(f'{name:16s} parity={"PASS" if ok else "FAIL"}  test_acc={acc:.4f}{h}')
    if not all(s[1] for s in summary):
        print('\nSome models FAILED. Do not commit those. See the output above.')


if __name__ == '__main__':
    main()
