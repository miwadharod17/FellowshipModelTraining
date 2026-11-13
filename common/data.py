"""Shared datasets. Every function returns identical data on every call."""
from pathlib import Path
import numpy as np
from sklearn.datasets import make_moons, load_wine
from sklearn.model_selection import train_test_split
from common.seed import SEED

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / 'data' / 'splits'
SHAPE_CLASSES = ['circle', 'square', 'triangle']  # agree with developers


def _tabular_split(name, X, y):
    """One stratified 70/30 split, made once, saved to data/splits/name.npz,
    then always loaded from the file. Never split again."""
    path = SPLIT_DIR / f'{name}.npz'
    if path.exists():
        d = np.load(path)
        return d['X_train'], d['X_test'], d['y_train'], d['y_test']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
    return X_tr, X_te, y_tr, y_te


def moons_main():
    """300 two-moons points, noise 0.25. Returns X_train, X_test, y_train, y_test."""
    X, y = make_moons(n_samples=300, noise=0.25, random_state=SEED)
    return _tabular_split('moons_main', X, y)


def wine():
    """Wine data. Returns X_train, X_test, y_train, y_test."""
    d = load_wine()
    return _tabular_split('wine', d.data, d.target)


def mnist():
    """Standard split: 55,000 train / 5,000 val / 10,000 test.
    Images are float32, scaled 0-1, shape (N, 28, 28). Flatten in your script."""
    import tensorflow as tf
    (x, y), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    x = x.astype('float32') / 255.0
    x_te = x_te.astype('float32') / 255.0
    return ((x[:55000], y[:55000]),
            (x[55000:], y[55000:]),
            (x_te, y_te))


def fashion_mnist():
    """Test images only, for the DL-5 anomaly demo. Never for training."""
    import tensorflow as tf
    (_, _), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    return x_te.astype('float32') / 255.0, y_te


def shapes(n, seed=SEED):
    """n generated 28x28 shape images (float32, 0-1) and int labels.
    Classes follow SHAPE_CLASSES. Use shapes(6000) to train, shapes(500, seed=7) to test."""
    from PIL import Image, ImageDraw
    rng = np.random.RandomState(seed)
    labels = rng.permutation(np.arange(n) % len(SHAPE_CLASSES))
    X = np.zeros((n, 28, 28), dtype=np.float32)
    for i in range(n):
        img = Image.new('L', (28, 28), 0)
        d = ImageDraw.Draw(img)
        r = rng.uniform(6, 10)
        cx = rng.uniform(r + 1, 27 - r)
        cy = rng.uniform(r + 1, 27 - r)
        c = labels[i]
        if c == 0:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        else:
            sides = 4 if c == 1 else 3
            rot = rng.uniform(0, 2 * np.pi)
            pts = [(cx + r * np.cos(rot + 2 * np.pi * k / sides),
                    cy + r * np.sin(rot + 2 * np.pi * k / sides))
                   for k in range(sides)]
            d.polygon(pts, fill=255)
        X[i] = np.asarray(img, dtype=np.float32) / 255.0
    return X, labels.astype(np.int64)


def make_all_splits():
    for fn in (moons_main, wine):
        X_tr, X_te, _, _ = fn()
        print(fn.__name__, 'train', X_tr.shape, 'test', X_te.shape)


if __name__ == '__main__':
    make_all_splits()
