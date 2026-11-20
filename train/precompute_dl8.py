"""DL-8 Inception-ResNet: no training and no .tflite. Runs pretrained Inception-ResNet-v2 on 12 sample images
and saves, into out/precomputed/:
    dl8_samples.json                 top-5 predictions + file names for every image
    dl8_fmaps/imageid_layername.png  36 feature-map pictures (12 images x 3 depths)
    dl8_images/imageid.jpg           the 12 images as the model saw them (299x299)
    dl8_samples_credits.txt          sources and licences of the 12 images

The images come from out/precomputed/dl6_samples/ (the 20 licensed pictures you collected for DL-6).
If there are more than 12, the 12 most clearly recognised ones are used (different top-1 classes first).
Image ids come from the file names ("01.jpg" -> img01). Rename a file first (e.g. retriever.jpg) for nicer ids.

Run from the repo root:
    python -m train.precompute_dl8
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from train.train_dl6 import get_labels

SRC = 'out/precomputed/dl6_samples'
SRC_CREDITS = 'out/precomputed/dl6_samples_credits.txt'
OUT = 'out/precomputed'
SIZE = 299
N_IMAGES = 12
GRID, TILE, GAP = 4, 96, 2          # 4x4 channels per picture, each shown 96x96 pixels
# three depths: early (edges/textures), middle (object parts), late (whole-object concepts)
TAPS = [('early', 'activation_4', 'Early layer: edges, colours and textures'),
        ('middle', 'block17_10_ac', 'Middle layer: parts and patterns'),
        ('late', 'block8_5_ac', 'Late layer: object-level features')]


def load_image(path):
    """Shorter side resized to 299, centre crop 299x299, RGB, as a PIL image."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    s = SIZE / min(w, h)
    img = img.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return img.crop((l, t, l + SIZE, t + SIZE))


def fmap_picture(a, path):
    """a: (H, W, C). Shows the 16 most varied channels as a 4x4 grid, each scaled to its own 0-1 range."""
    n = GRID * GRID
    ch = np.argsort(a.reshape(-1, a.shape[-1]).var(0))[::-1][:n]
    try:
        from matplotlib import cm
        cmap = cm.get_cmap('viridis') if hasattr(cm, 'get_cmap') else cm.viridis
    except Exception:
        cmap = None
    side = GRID * TILE + (GRID - 1) * GAP
    canvas = np.zeros((side, side, 3), np.uint8)
    for k, c in enumerate(ch):
        m = a[..., c]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        tile = Image.fromarray((m * 255).astype(np.uint8)).resize((TILE, TILE), Image.BICUBIC)
        t = np.asarray(tile, np.float32) / 255.0
        rgb = (cmap(t)[..., :3] * 255).astype(np.uint8) if cmap else np.stack([(t * 255).astype(np.uint8)] * 3, -1)
        r, c2 = divmod(k, GRID)
        y0, x0 = r * (TILE + GAP), c2 * (TILE + GAP)
        canvas[y0:y0 + TILE, x0:x0 + TILE] = rgb
    Image.fromarray(canvas).save(path)
    return [int(c) for c in ch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--random-weights', action='store_true', help='testing only: results are meaningless')
    args = ap.parse_args()

    files = sorted(glob.glob(f'{args.src}/*.jpg') + glob.glob(f'{args.src}/*.jpeg') + glob.glob(f'{args.src}/*.png'))
    assert len(files) >= N_IMAGES, f'Need at least {N_IMAGES} images in {args.src}, found {len(files)}.'
    labels = get_labels(args.random_weights)

    base = keras.applications.InceptionResNetV2(weights=None if args.random_weights else 'imagenet')
    names = [t[1] for t in TAPS]
    model = keras.Model(base.input, [base.output] + [base.get_layer(n).output for n in names])

    # ---- run every candidate image once
    runs = []
    for f in files:
        img = load_image(f)
        x = keras.applications.inception_resnet_v2.preprocess_input(
            np.asarray(img, np.float32)[None])
        out = model(x, training=False)
        probs = np.asarray(out[0])[0]
        runs.append({'file': f, 'img': img, 'probs': probs, 'fmaps': [np.asarray(o)[0] for o in out[1:]]})
        print(os.path.basename(f), '->', labels[int(probs.argmax())], round(float(probs.max()), 3))

    # ---- pick 12: most confident first, different top-1 classes first
    order = sorted(range(len(runs)), key=lambda i: -runs[i]['probs'].max())
    chosen, seen = [], set()
    for i in order:
        top1 = int(runs[i]['probs'].argmax())
        if top1 not in seen and len(chosen) < N_IMAGES:
            chosen.append(i); seen.add(top1)
    for i in order:
        if i not in chosen and len(chosen) < N_IMAGES:
            chosen.append(i)
    chosen = sorted(chosen, key=lambda i: files[i])

    os.makedirs(f'{OUT}/dl8_fmaps', exist_ok=True)
    os.makedirs(f'{OUT}/dl8_images', exist_ok=True)
    for old in glob.glob(f'{OUT}/dl8_fmaps/*.png') + glob.glob(f'{OUT}/dl8_images/*.jpg'):
        os.remove(old)    # so the folders contain exactly the 12 chosen images

    images, ids_used = [], set()
    for i in chosen:
        r = runs[i]
        stem = re.sub(r'[^a-z0-9]', '', os.path.splitext(os.path.basename(r['file']))[0].lower()) or 'x'
        iid = stem if not stem[0].isdigit() else 'img' + stem
        while iid in ids_used:
            iid += 'x'
        ids_used.add(iid)
        r['img'].save(f'{OUT}/dl8_images/{iid}.jpg', quality=90)
        fm = {}
        for (depth, lname, _), a in zip(TAPS, r['fmaps']):
            fn = f'dl8_fmaps/{iid}_{lname}.png'
            chans = fmap_picture(a, f'{OUT}/{fn}')
            fm[depth] = {'file': fn, 'layer': lname, 'channels_shown': chans}
        top = np.argsort(r['probs'])[::-1][:5]
        images.append({'id': iid, 'source_file': os.path.basename(r['file']), 'image': f'dl8_images/{iid}.jpg',
                       'top5': [{'label': labels[int(j)], 'p': round(float(r['probs'][j]), 4)} for j in top],
                       'fmaps': fm})
        r['iid'] = iid

    layers = [{'depth': d, 'layer': n, 'description': desc,
               'shape': [int(s) for s in model.get_layer(n).output.shape[1:]]} for d, n, desc in TAPS]
    json.dump({'precomputed': True,
               'note': 'Results made ahead of time with pretrained Inception-ResNet-v2 (ImageNet). Show them labelled as precomputed.',
               'model': 'Inception-ResNet-v2, ImageNet weights, input 299x299 RGB',
               'feature_map_picture': f'each picture shows the {GRID * GRID} most varied channels of the layer, '
                                      f'each scaled to its own range',
               'random_weights': args.random_weights, 'layers': layers, 'images': images},
              open(f'{OUT}/dl8_samples.json', 'w'), indent=1)

    # ---- credits: copy the matching lines from the DL-6 credits file
    lines = open(SRC_CREDITS).read().splitlines() if os.path.exists(SRC_CREDITS) else []
    out_lines, missing = [], []
    for i in chosen:
        fname = os.path.basename(files[i])
        hit = [ln for ln in lines if fname in ln]
        (out_lines.extend if hit else missing.append)(hit if hit else fname)
    open(f'{OUT}/dl8_samples_credits.txt', 'w').write('\n'.join(out_lines) + '\n')
    print(f'\nwrote {len(images)} images, {len(glob.glob(OUT + "/dl8_fmaps/*.png"))} feature-map pictures, dl8_samples.json')
    if missing:
        print('WARNING: no credit line found for', missing, '- add them to', f'{OUT}/dl8_samples_credits.txt')


if __name__ == '__main__':
    main()
