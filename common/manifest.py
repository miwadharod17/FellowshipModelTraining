import glob
import hashlib
import json
import os


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def add_entry(entry, part_file):
    """part_file is your own file: out/manifest_a.json, _b.json or _c.json."""
    entries = []
    if os.path.exists(part_file):
        with open(part_file) as f:
            entries = json.load(f)
    entries = [e for e in entries if e['id'] != entry['id']]
    entries.append(entry)
    with open(part_file, 'w') as f:
        json.dump(entries, f, indent=2)


def make_entry(name, path, type_, inputs, outputs, preprocessing,
               labels=None, engine=None, version=1):
    e = {'id': name, 'type': type_, 'version': version,
         'file': os.path.basename(path), 'sha256': sha256_of(path),
         'inputs': inputs, 'outputs': outputs,
         'labels': labels, 'preprocessing': preprocessing}
    if engine:
        e['engine'] = engine
    return e


def join_manifest(out='out/manifest.json'):
    entries = []
    for p in sorted(glob.glob('out/manifest_[abc].json')):
        with open(p) as f:
            entries += json.load(f)
    ids = [e['id'] for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f'Duplicate ids: {dupes}'
    with open(out, 'w') as f:
        json.dump({'files': entries}, f, indent=2)
    print(len(entries), 'entries written to', out)


if __name__ == '__main__':
    join_manifest()
