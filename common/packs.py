import json
import numpy as np


def write_pack(prefix, arrays, meta):
    # arrays: dict of name -> numpy array. Writes prefix.bin and prefix.json
    offset = 0
    index = []
    with open(prefix + '.bin', 'wb') as f:
        for name, arr in arrays.items():
            a = np.asarray(arr, dtype='<f4')
            f.write(a.tobytes())
            index.append({'name': name, 'shape': list(a.shape), 'offset': offset})
            offset += a.size
    with open(prefix + '.json', 'w') as f:
        json.dump({'arrays': index, 'meta': meta}, f, indent=2)


def read_pack(prefix):
    with open(prefix + '.json') as f:
        info = json.load(f)
    data = np.fromfile(prefix + '.bin', dtype='<f4')
    out = {}
    for e in info['arrays']:
        n = int(np.prod(e['shape']))
        out[e['name']] = data[e['offset']:e['offset'] + n].reshape(e['shape'])
    return out, info['meta']
