import numpy as np
import tensorflow as tf


def run_tflite_keras(path, x):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]['index'], x.astype(np.float32))
    it.invoke()
    return [it.get_tensor(d['index']) for d in it.get_output_details()]


def run_tflite_engine(path, **inputs):
    it = tf.lite.Interpreter(model_path=path)
    runner = it.get_signature_runner()
    return runner(**{k: np.asarray(v, dtype=np.float32) for k, v in inputs.items()})


def compare(a, b, tol, label=''):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    print(label, 'max difference', err, 'PASS' if err <= tol else 'FAIL')
    return err <= tol


def parity_and_golden(keras_model, tflite_path, x20, name,
                      fp16=False, classifier=False, golden_dir='out/golden'):
    """DL models, batch size 1. x20: 20 fixed test inputs, shape (20, ...).
    Compares Keras vs .tflite on each input, then saves golden files."""
    tol = 1e-2 if fp16 else 1e-4
    ok, agree, outs = True, 0, []
    for i in range(len(x20)):
        xi = x20[i:i + 1].astype(np.float32)
        ref = keras_model(xi, training=False)
        ref = list(ref) if isinstance(ref, (list, tuple)) else [ref]
        got = run_tflite_keras(tflite_path, xi)
        for r, g in zip(ref, got):
            ok &= compare(np.asarray(r), g, tol, f'{name}[{i}]')
        if classifier:
            agree += int(np.argmax(np.asarray(ref[0])) == np.argmax(got[0]))
        outs.append(got)
    if classifier:
        frac = agree / len(x20)
        print(name, 'label agreement', frac)
        ok &= frac >= 0.999
    np.save(f'{golden_dir}/{name}_golden_in.npy', x20.astype(np.float32))
    for k in range(len(outs[0])):
        suffix = '' if len(outs[0]) == 1 else f'_{k}'
        np.save(f'{golden_dir}/{name}_golden_out{suffix}.npy',
                np.concatenate([o[k] for o in outs]))
    return bool(ok)
