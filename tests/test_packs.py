import os
import tempfile
import numpy as np
from common.packs import write_pack, read_pack


def test_roundtrip():
    arrays = {'a': np.random.RandomState(42).rand(3, 4), 'b': np.arange(5)}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'x')
        write_pack(p, arrays, {'k': 1})
        out, meta = read_pack(p)
    assert meta == {'k': 1}
    for k in arrays:
        expected = np.asarray(arrays[k], dtype='<f4')
        assert out[k].shape == expected.shape
        assert np.array_equal(out[k], expected)


if __name__ == '__main__':
    test_roundtrip()
    print('PASS')
