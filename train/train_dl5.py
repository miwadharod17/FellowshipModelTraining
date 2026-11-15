"""DL-5 Autoencoders: trains 9 models (bottleneck 2/8/32 x linear/relu/dae),
converts each to .tflite, runs the parity test, saves metrics, golden files
and manifest entries.

Run from the repo root (the folder that contains common/ and out/):

    python -m train.train_dl5                         # all 9 models
    python -m train.train_dl5 --only b8_relu --epochs 1   # quick test of one model
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA

from common.seed import SEED, set_seed
from common.data import mnist, fashion_mnist
from common.convert import keras_to_tflite
from common.metrics import write_metrics
from common.parity import parity_and_golden
from common.manifest import make_entry, add_entry

BOTTLENECKS = [2, 8, 32]
KINDS = ['linear', 'relu', 'dae']
HIDDEN = 128
BATCH = 256
DAE_NOISE = 0.3            # std of the Gaussian noise used to train the dae models
PCA_COMPARISON = True      # optional extra: PCA reconstruction error at the same sizes
MANIFEST_PART = 'out/manifest_c.json'
ROLE_BY_DIM = {784: 'reconstruction', 1: 'recon_error'}   # anything else is the latent code


# ---------------------------------------------------------------- model
def build(b, kind, batch_size=None):
    """784 -> (hidden) -> b -> (hidden) -> 784. Outputs: reconstruction, latent, recon_error.
    Input is the flattened digit already scaled 0-1, so no preprocessing is needed inside."""
    inp = keras.Input(shape=(784,), batch_size=batch_size, name='image')
    if kind == 'linear':
        z = keras.layers.Dense(b, name='enc_latent')(inp)
        r = keras.layers.Dense(784, name='dec_out')(z)
    else:   # relu and dae share the same network, dae is only trained differently
        h = keras.layers.Dense(HIDDEN, activation='relu', name='enc_hidden')(inp)
        z = keras.layers.Dense(b, name='enc_latent')(h)
        h = keras.layers.Dense(HIDDEN, activation='relu', name='dec_hidden')(z)
        r = keras.layers.Dense(784, activation='sigmoid', name='dec_out')(h)
    err = keras.ops.mean(keras.ops.square(keras.ops.subtract(inp, r)),
                         axis=-1, keepdims=True)
    outs = [keras.layers.Identity(name='reconstruction')(r),
            keras.layers.Identity(name='latent')(z),
            keras.layers.Identity(name='recon_error')(err)]
    return keras.Model(inp, outs, name=f'dl5_ae_b{b}_{kind}')


# ---------------------------------------------------------------- helpers
def per_sample_mse(x, recon):
    return np.mean((x - recon) ** 2, axis=1)


def convert(model, path):
    try:
        keras_to_tflite(model, path)
    except Exception as e:   # fallback suggested in the guide
        print('from_keras_model failed:', e, '\n-> trying model.export fallback')
        tmp = 'out/_tmp_saved_model'
        model.export(tmp)
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        with open(path, 'wb') as f:
            f.write(conv.convert())


def tflite_io(path):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    ins = [{'name': d['name'], 'shape': [int(s) for s in d['shape']],
            'dtype': 'float32', 'range': [0.0, 1.0]} for d in it.get_input_details()]
    outs = []
    for d in it.get_output_details():
        shape = [int(s) for s in d['shape']]
        outs.append({'name': d['name'], 'role': ROLE_BY_DIM.get(shape[-1], 'latent'),
                     'shape': shape, 'dtype': 'float32'})
    return ins, outs


def keras_in_tflite_order(model, outs):
    """TFLite may order outputs differently from Keras. Output sizes are all different
    (784, b, 1), so match by size and return a Keras model with the same order as .tflite."""
    by_dim = {int(t.shape[-1]): t for t in model.outputs}
    return keras.Model(model.input, [by_dim[o['shape'][-1]] for o in outs])


def write_demo_samples(x_te, y_te, fx, fy, path):
    samples = []
    for c in range(10):
        i = int(np.where(y_te == c)[0][0])
        samples.append({'id': f'digit_{c}', 'kind': 'digit', 'label': c,
                        'pixels': np.round(x_te[i], 3).tolist()})
    for c, nm in [(0, 'tshirt'), (1, 'trouser'), (7, 'sneaker'), (8, 'bag')]:
        i = int(np.where(fy == c)[0][0])
        samples.append({'id': f'unusual_{nm}', 'kind': 'unusual', 'label': None,
                        'pixels': np.round(fx[i], 3).tolist()})
    with open(path, 'w') as f:
        json.dump({'note': '28x28 images flattened to 784 values in 0-1. '
                           'unusual = Fashion-MNIST items for the anomaly demo.',
                   'samples': samples}, f)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='run only models whose name contains this, e.g. b8_relu')
    ap.add_argument('--epochs', type=int, default=15)
    args = ap.parse_args()

    assert os.path.isdir('out/golden'), 'Run this from the repo root (folder containing out/).'

    (x_tr, _), (x_val, _), (x_te, y_te) = mnist()
    x_tr, x_val, x_te = (a.reshape(len(a), 784) for a in (x_tr, x_val, x_te))
    fx, fy = fashion_mnist()
    fx = fx.reshape(len(fx), 784)

    # 20 fixed golden inputs: 16 digits + 4 unusual images, same for every model
    x20 = np.concatenate([x_te[:16], fx[:4]]).astype(np.float32)
    write_demo_samples(x_te.reshape(-1, 28, 28), y_te, fx.reshape(-1, 28, 28), fy,
                       'out/precomputed/dl5_demo_samples.json')

    # fixed noisy validation set for the dae models
    rng = np.random.RandomState(SEED)
    x_val_noisy = np.clip(x_val + DAE_NOISE * rng.randn(*x_val.shape), 0, 1).astype(np.float32)

    pca_mse = {}
    if PCA_COMPARISON:
        for b in BOTTLENECKS:
            p = PCA(n_components=b, random_state=SEED).fit(x_tr)
            pca_mse[b] = float(np.mean((x_te - p.inverse_transform(p.transform(x_te))) ** 2))
        print('PCA test MSE:', pca_mse)

    summary = []
    for b in BOTTLENECKS:
        for kind in KINDS:
            name = f'dl5_ae_b{b}_{kind}'
            if args.only and args.only not in name:
                continue
            print(f'\n===== {name} =====')
            set_seed()
            model = build(b, kind)
            train_model = keras.Model(model.input, model.get_layer('reconstruction').output)
            train_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

            # training data: dae gets fresh noise every epoch, the others see clean images
            ds = tf.data.Dataset.from_tensor_slices((x_tr, x_tr)).shuffle(len(x_tr), seed=SEED)
            ds = ds.batch(BATCH)
            if kind == 'dae':
                ds = ds.map(lambda x, y: (tf.clip_by_value(
                    x + DAE_NOISE * tf.random.normal(tf.shape(x)), 0.0, 1.0), y))
                val = (x_val_noisy, x_val)
            else:
                val = (x_val, x_val)
            hist = train_model.fit(ds, epochs=args.epochs, validation_data=val, verbose=2)

            # evaluation on clean images + anomaly threshold
            rec_val = model.predict(x_val, batch_size=1000, verbose=0)[0]
            rec_te = model.predict(x_te, batch_size=1000, verbose=0)[0]
            rec_fx = model.predict(fx, batch_size=1000, verbose=0)[0]
            err_val, err_te, err_fx = (per_sample_mse(x_val, rec_val),
                                       per_sample_mse(x_te, rec_te),
                                       per_sample_mse(fx, rec_fx))
            thr = float(np.percentile(err_val, 95))

            # export model with batch size 1 and the trained weights
            export = build(b, kind, batch_size=1)
            export.set_weights(model.get_weights())
            path = f'out/models/{name}.tflite'
            convert(export, path)

            # parity test + golden files (compare in .tflite output order)
            ins, outs = tflite_io(path)
            ordered = keras_in_tflite_order(export, outs)
            ok = parity_and_golden(ordered, path, x20, name)

            write_metrics(
                name, 'mnist (train 55000 / val 5000 / test 10000); fashion_mnist test for anomaly demo only',
                {'bottleneck': b, 'type': kind, 'hidden': HIDDEN, 'epochs': args.epochs,
                 'batch_size': BATCH, 'optimizer': 'adam 1e-3', 'loss': 'mse',
                 'dae_noise_std': DAE_NOISE if kind == 'dae' else 0.0},
                loss_curve=[float(v) for v in hist.history['loss']],
                val_loss_curve=[float(v) for v in hist.history['val_loss']],
                test_mse=float(np.mean(err_te)),
                pca_test_mse=pca_mse.get(b),
                anomaly_threshold=thr,
                anomaly_rule='flag as unusual when recon_error > anomaly_threshold (95th percentile of clean validation error)',
                mnist_test_flag_rate=float(np.mean(err_te > thr)),
                fashion_flag_rate=float(np.mean(err_fx > thr)),
                parity_pass=bool(ok))

            if ok:
                e = make_entry(
                    name, path, 'model', ins, outs,
                    'none inside the model: input is the 28x28 digit flattened to 784 float32 values already scaled 0 to 1',
                    labels=None)
                e['anomaly_threshold'] = thr
                add_entry(e, MANIFEST_PART)
            summary.append((name, ok, float(np.mean(err_te)), thr))

    print('\n================ SUMMARY ================')
    for name, ok, mse, thr in summary:
        print(f'{name:22s} parity={"PASS" if ok else "FAIL"}  test_mse={mse:.5f}  threshold={thr:.5f}')
    if not all(s[1] for s in summary):
        print('\nSome models FAILED parity. Do not commit those. See the output above.')


if __name__ == '__main__':
    main()
