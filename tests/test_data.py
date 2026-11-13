import numpy as np
from common.data import moons_main, wine, shapes


def test_moons_split():
    a = moons_main()
    b = moons_main()
    assert len(a[0]) == 210 and len(a[1]) == 90
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_wine_split():
    X_tr, X_te, _, _ = wine()
    assert len(X_tr) + len(X_te) == 178


def test_shapes_deterministic():
    a, la = shapes(30)
    b, lb = shapes(30)
    assert a.shape == (30, 28, 28) and np.array_equal(a, b) and np.array_equal(la, lb)
    c, _ = shapes(30, seed=7)
    assert not np.array_equal(a, c)


if __name__ == '__main__':
    test_moons_split(); test_wine_split(); test_shapes_deterministic()
    print('PASS')
