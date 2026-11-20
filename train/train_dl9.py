"""DL-9 Attention Mechanism Visualizer: trains a tiny transformer (masked language model) on
template sentences and exports it with its attention scores as an output.

Sentences look like:   the big dogs chase the ball in the garden
  - determiner (+ optional adjective) + subject noun   (a/the/some agree with singular/plural)
  - verb that agrees in number with the subject        (the verb must look back at the subject)
  - the + object noun
  - optional place phrase: preposition + the + place   (place depends on the preposition)

Run from the repo root:
    python -m train.train_dl9                       # full run
    python -m train.train_dl9 --epochs 1            # quick test
Options: --ids int32|float32 (type of the token-id input, default int32)
         --parity strict|relative (how fp16 parity is judged, default strict)
"""
import argparse
import json
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from common.seed import SEED, set_seed
from common.convert import keras_to_tflite
from common.metrics import write_metrics
from common.manifest import make_entry, add_entry

NAME = 'dl9_tiny_mlm_fp16'
MANIFEST_PART = 'out/manifest_c.json'
L, D, H, FF, NL = 12, 32, 2, 64, 2        # length, model width, heads, feed-forward width, layers
DH = D // H
PAD, MASK, CLS, SEP, UNK = 0, 1, 2, 3, 4
SPECIALS = ['[PAD]', '[MASK]', '[CLS]', '[SEP]', '[UNK]']

# ---------------------------------------------------------------- the toy language
SG_NOUNS = ['dog', 'cat', 'bird', 'fox', 'horse', 'mouse']
PL_NOUNS = ['dogs', 'cats', 'birds', 'foxes', 'horses', 'mice']
SG_VERBS = ['chases', 'sees', 'likes', 'follows', 'watches']
PL_VERBS = ['chase', 'see', 'like', 'follow', 'watch']
OBJECTS = ['ball', 'bone', 'fish', 'seed', 'apple', 'stick']
ADJECTIVES = ['big', 'small', 'red', 'old']
PLACES = {'in': ['garden', 'house', 'barn'], 'on': ['hill', 'roof', 'bridge'],
          'near': ['river', 'tree', 'road'], 'by': ['lake', 'door', 'wall']}
WORDS = (['the', 'a', 'some'] + SG_NOUNS + PL_NOUNS + SG_VERBS + PL_VERBS + OBJECTS + ADJECTIVES
         + list(PLACES) + [p for v in PLACES.values() for p in v])
VOCAB = SPECIALS + sorted(set(WORDS))
TOK = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
SG_VERB_IDS = {TOK[w] for w in SG_VERBS}
PL_VERB_IDS = {TOK[w] for w in PL_VERBS}


def make_sentence(rng):
    """Returns (list of tokens, subject_noun_position, verb_position, is_plural)."""
    plural = rng.rand() < 0.5
    t = ['[CLS]', rng.choice(['the', 'some'] if plural else ['the', 'a'])]
    if rng.rand() < 0.4:
        t.append(rng.choice(ADJECTIVES))
    subj = len(t)
    t.append(rng.choice(PL_NOUNS if plural else SG_NOUNS))
    verb = len(t)
    t.append(rng.choice(PL_VERBS if plural else SG_VERBS))
    t += ['the', rng.choice(OBJECTS)]
    if rng.rand() < 0.5:
        prep = rng.choice(list(PLACES))
        t += [prep, 'the', rng.choice(PLACES[prep])]
    t.append('[SEP]')
    return t, subj, verb, plural


def encode(tokens):
    ids = [TOK.get(w, UNK) for w in tokens][:L]
    return ids + [PAD] * (L - len(ids))


def make_set(n, seed):
    rng = np.random.RandomState(seed)
    X, subj, verb, plural, length = [], [], [], [], []
    for _ in range(n):
        t, s, v, p = make_sentence(rng)
        X.append(encode(t)); subj.append(s); verb.append(v); plural.append(p); length.append(len(t))
    return (np.array(X, dtype=np.int32), np.array(subj), np.array(verb),
            np.array(plural), np.array(length))


def mask_batch(X, seed):
    """Mask 1 or 2 real words per sentence. Returns masked inputs, labels, weights (sum = L per sentence)."""
    rng = np.random.RandomState(seed)
    Xm, W = X.copy(), np.zeros(X.shape, dtype=np.float32)
    for i in range(len(X)):
        real = np.where(X[i] > SEP)[0]               # words only (not PAD/MASK/CLS/SEP)
        pick = rng.choice(real, size=rng.choice([1, 2]), replace=False)
        Xm[i, pick] = MASK
        W[i, pick] = 1.0
    W = W * (L / W.sum(1, keepdims=True))
    return Xm, X.copy(), W


# ---------------------------------------------------------------- model
class AddPositions(keras.layers.Layer):
    """Adds a learned position vector to every token."""
    def build(self, input_shape):
        self.pos = self.add_weight(shape=(L, D), initializer='random_normal', name='pos')

    def call(self, x):
        return x + self.pos


def build(ids_dtype='int32', batch_size=None):
    """Input: token ids [1,12]. Outputs: mlm_logits [1,12,V] and attn_scores [1,4,12,12]
    (raw scores before softmax, heads ordered layer0-head0, layer0-head1, layer1-head0, layer1-head1;
    padding keys are -10000). The app applies softmax(scores / temperature)."""
    ops = keras.ops
    inp = keras.Input(shape=(L,), batch_size=batch_size, dtype=ids_dtype, name='token_ids')
    ids = ops.cast(inp, 'int32')
    pad_mask = ops.reshape(ops.cast(ops.equal(ids, PAD), 'float32') * -1e4, (-1, 1, 1, L))

    x = keras.layers.Embedding(V, D, name='tok_emb')(ids)
    x = AddPositions(name='pos_emb')(x)
    all_scores = []
    for l in range(NL):
        def split(t):   # (B, L, D) -> (B, H, L, DH)
            return ops.transpose(ops.reshape(t, (-1, L, H, DH)), (0, 2, 1, 3))
        q = split(keras.layers.Dense(D, name=f'l{l}_q')(x))
        k = split(keras.layers.Dense(D, name=f'l{l}_k')(x))
        v = split(keras.layers.Dense(D, name=f'l{l}_v')(x))
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / math.sqrt(DH) + pad_mask
        all_scores.append(scores)
        weights = ops.softmax(scores, axis=-1)
        ctx = ops.reshape(ops.transpose(ops.matmul(weights, v), (0, 2, 1, 3)), (-1, L, D))
        x = keras.layers.LayerNormalization(epsilon=1e-5, name=f'l{l}_ln1')(
            x + keras.layers.Dense(D, name=f'l{l}_o')(ctx))
        h = keras.layers.Dense(D, name=f'l{l}_ff2')(keras.layers.Dense(FF, activation='relu', name=f'l{l}_ff1')(x))
        x = keras.layers.LayerNormalization(epsilon=1e-5, name=f'l{l}_ln2')(x + h)
    logits = keras.layers.Dense(V, name='mlm_head')(x)
    attn = ops.concatenate(all_scores, axis=1)                       # (B, NL*H, L, L)
    return keras.Model(inp, [keras.layers.Identity(name='mlm_logits')(logits),
                             keras.layers.Identity(name='attn_scores')(attn)], name='dl9_tiny_mlm')


# ---------------------------------------------------------------- checks
def softmax_np(a):
    a = a - a.max(-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(-1, keepdims=True)


def parity(model, path, x20, name, mode, golden_dir='out/golden'):
    """Keras vs .tflite on 20 inputs, batch size 1. Strict: max abs diff <= 1e-2 (fp16 rule in the guide).
    Relative: cosine >= 0.999 and mean |diff|/mean |ref| <= 5% (needs your lead's OK)."""
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outs = it.get_output_details()
    ok, golden, worst = True, [[] for _ in outs], {}
    for i in range(len(x20)):
        xi = x20[i:i + 1].astype(inp['dtype'])
        ref = model(xi, training=False)
        ref = {tuple(np.asarray(r).shape): np.asarray(r) for r in ref}
        it.set_tensor(inp['index'], xi)
        it.invoke()
        for k, d in enumerate(outs):
            g = it.get_tensor(d['index'])
            r = ref[tuple(g.shape)]
            diff = np.abs(g - r)
            valid = r > -5000           # ignore the -10000 padding entries in the scores
            cos = float(r.ravel() @ g.ravel() / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-12))
            rel = float(diff[valid].mean() / (np.abs(r[valid]).mean() + 1e-12))
            w = worst.setdefault(k, {'abs': 0.0, 'cos': 1.0, 'rel': 0.0})
            w['abs'], w['cos'], w['rel'] = (max(w['abs'], float(diff.max())), min(w['cos'], cos), max(w['rel'], rel))
            golden[k].append(g)
    for k, d in enumerate(outs):
        w = worst[k]
        good = (w['abs'] <= 1e-2) if mode == 'strict' else (w['cos'] >= 0.999 and w['rel'] <= 0.05)
        ok &= bool(good)
        print(f"{name} output {k} {str([int(a) for a in d['shape']]):16s} max abs {w['abs']:.5f}  min cosine {w['cos']:.6f}  "
              f"mean rel err {w['rel']:.4f}  {'PASS' if good else 'FAIL'} ({mode})")
    np.save(f'{golden_dir}/{name}_golden_in.npy', x20.astype(inp['dtype']))
    for k in range(len(outs)):
        np.save(f'{golden_dir}/{name}_golden_out_{k}.npy', np.concatenate(golden[k]))
    return bool(ok), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--n-train', type=int, default=60000)
    ap.add_argument('--ids', choices=['int32', 'float32'], default='int32')
    ap.add_argument('--parity', choices=['strict', 'relative'], default='strict')
    args = ap.parse_args()
    assert os.path.isdir('out/golden'), 'Run this from the repo root (folder containing out/).'

    # ---- data
    Xtr, *_ = make_set(args.n_train, seed=1)
    # test sentences must NOT appear in the training set (the toy language is small, so most random ones would)
    Xte, s_te, v_te, p_te, len_te = make_set(12000, seed=2)
    seen = {tuple(r) for r in Xtr}
    keep = np.array([tuple(r) not in seen for r in Xte])
    Xte, s_te, v_te, p_te, len_te = (a[keep][:2000] for a in (Xte, s_te, v_te, p_te, len_te))
    assert len(Xte) >= 1000, 'too few unseen test sentences, lower --n-train'
    print('test sentences never seen in training:', len(Xte))
    Xtr_m, Ytr, Wtr = mask_batch(Xtr, seed=3)
    Xva_m, Yva, Wva = mask_batch(Xte[:1000], seed=4)
    print('vocab size', V, '| train sentences', len(Xtr))

    # ---- train
    set_seed()
    model = build('int32')
    train_model = keras.Model(model.input, model.output[0])
    train_model.compile(optimizer=keras.optimizers.Adam(2e-3),
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    hist = train_model.fit(Xtr_m, Ytr, sample_weight=Wtr, batch_size=256, epochs=args.epochs,
                           validation_data=(Xva_m, Yva, Wva), verbose=2)

    # ---- does it work? masked-word accuracy + verb agreement
    logits = model.predict(Xva_m, batch_size=500, verbose=0)[0]
    m = Wva > 0
    masked_acc = float(np.mean(logits.argmax(-1)[m] == Yva[m]))
    Xv = Xte.copy()
    Xv[np.arange(len(Xv)), v_te] = MASK                      # hide only the verb
    pred = model.predict(Xv, batch_size=500, verbose=0)[0].argmax(-1)[np.arange(len(Xv)), v_te]
    agree = float(np.mean([(p in PL_VERB_IDS) if pl else (p in SG_VERB_IDS) for p, pl in zip(pred, p_te)]))
    print(f'masked-word accuracy {masked_acc:.3f} | verb agreement accuracy {agree:.3f} on unseen sentences (chance 0.5)')

    # ---- does the attention show structure? attention from the verb to the subject noun (T = 1)
    scores = model.predict(Xte, batch_size=500, verbose=0)[1]          # (N, 4, L, L)
    w = softmax_np(scores)
    n = np.arange(len(Xte))
    head_names = [f'layer{l}_head{h}' for l in range(NL) for h in range(H)]
    verb_to_subj = {hn: float(w[n, i, v_te, s_te].mean()) for i, hn in enumerate(head_names)}
    baseline = float(np.mean(1.0 / len_te))
    print('attention from verb to subject noun (uniform attention would be %.3f):' % baseline)
    for hn, val in verb_to_subj.items():
        print(f'   {hn}: {val:.3f}')
    best = max(verb_to_subj.values())
    if agree < 0.95 or best < 0.30:
        print('WARNING: the model or its attention is weak. Try --epochs 20 before using it.')

    # ---- export (batch size 1, fp16) and parity
    export = build(args.ids, batch_size=1)
    export.set_weights(model.get_weights())
    path = f'out/models/{NAME}.tflite'
    keras_to_tflite(export, path, fp16=True)
    print('tflite size (KB):', round(os.path.getsize(path) / 1e3, 1))

    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    ins = [{'name': d['name'], 'shape': [int(s) for s in d['shape']], 'dtype': np.dtype(d['dtype']).name,
            'range': [0, V - 1]} for d in it.get_input_details()]
    outs = [{'name': d['name'], 'shape': [int(s) for s in d['shape']], 'dtype': 'float32',
             'role': 'mlm_logits' if len(d['shape']) == 3 else 'attn_scores'} for d in it.get_output_details()]

    x20 = np.concatenate([Xva_m[:10], Xte[:10]])             # 10 masked + 10 plain sentences
    ok, worst = parity(export, path, x20, NAME, args.parity)

    # label agreement on masked positions (1000 sentences). Positions where the top two candidate words are
    # almost tied in the fp32 model (margin < 0.05, e.g. a random object noun) are reported but not required:
    # fp16 rounding can legitimately flip a coin-flip.
    itp = tf.lite.Interpreter(model_path=path)
    itp.allocate_tensors()
    ii = itp.get_input_details()[0]
    od = itp.get_output_details()
    li = [i for i, o in enumerate(od) if len(o['shape']) == 3][0]
    ref_logits = model.predict(Xva_m, batch_size=500, verbose=0)[0]
    top2 = np.sort(ref_logits, axis=-1)[..., -2:]
    margin = top2[..., 1] - top2[..., 0]
    same = tot = same_c = tot_c = 0
    for i in range(len(Xva_m)):
        itp.set_tensor(ii['index'], Xva_m[i:i + 1].astype(ii['dtype']))
        itp.invoke()
        got = itp.get_tensor(od[li]['index'])[0].argmax(-1)
        for pos in np.where(Wva[i] > 0)[0]:
            hit = int(got[pos] == ref_logits[i, pos].argmax())
            same += hit; tot += 1
            if margin[i, pos] >= 0.05:
                same_c += hit; tot_c += 1
    label_agree, label_agree_conf = same / tot, same_c / tot_c
    print(f'masked-word label agreement Keras vs tflite: all positions {label_agree:.4f} ({tot - same} flips of {tot}), '
          f'confident positions {label_agree_conf:.4f}')
    ok = bool(ok and label_agree_conf >= 0.999)

    # ---- files for the developers
    vocab = {'tokens': VOCAB, 'pad_id': PAD, 'mask_id': MASK, 'cls_id': CLS, 'sep_id': SEP, 'unk_id': UNK,
             'max_len': L, 'note': 'Lowercase, split on spaces, map unknown words to [UNK], add [CLS] first and '
             '[SEP] last, pad with [PAD] to 12 ids. The user can type [MASK] to hide a word.'}
    json.dump(vocab, open('out/precomputed/dl9_vocab.json', 'w'), indent=1)
    rng = np.random.RandomState(11)
    examples = []
    for i in range(8):
        t, s, v, p = make_sentence(rng)
        examples.append({'text': ' '.join(t[1:-1]), 'ids': encode(t), 'note': 'plain sentence'})
    for i in range(4):
        t, s, v, p = make_sentence(rng)
        t[v] = '[MASK]'
        examples.append({'text': ' '.join(t[1:-1]), 'ids': encode(t), 'note': 'masked verb: look at which word it attends to'})
    json.dump({'examples': examples}, open('out/precomputed/dl9_examples.json', 'w'), indent=1)

    try:   # one picture to look at with your own eyes
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t, s, v, p = make_sentence(np.random.RandomState(5))
        ids = np.array([encode(t)], dtype=np.int32)
        wt = softmax_np(model.predict(ids, verbose=0)[1])[0]
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        for i, ax in enumerate(axs):
            ax.imshow(wt[i][:len(t), :len(t)], cmap='viridis')
            ax.set_xticks(range(len(t))); ax.set_xticklabels(t, rotation=90, fontsize=7)
            ax.set_yticks(range(len(t))); ax.set_yticklabels(t, fontsize=7)
            ax.set_title(head_names[i])
        plt.tight_layout(); plt.savefig('out/metrics/dl9_attention_example.png', dpi=110); plt.close()
    except Exception as e:
        print('(picture skipped:', e, ')')

    write_metrics(NAME, 'template sentences generated in train_dl9.py (see the script), seed 42',
                  {'layers': NL, 'heads': H, 'd_model': D, 'ffn': FF, 'max_len': L, 'vocab_size': V,
                   'epochs': args.epochs, 'n_train': args.n_train, 'masking': '1-2 words per sentence',
                   'fp16': True, 'ids_dtype': args.ids, 'parity_mode': args.parity,
                   'head_order': head_names},
                  loss_curve=[float(v) for v in hist.history['loss']],
                  val_loss_curve=[float(v) for v in hist.history['val_loss']],
                  masked_word_accuracy=masked_acc, verb_agreement_accuracy=agree,
                  attention_verb_to_subject=verb_to_subj, attention_uniform_baseline=baseline,
                  label_agreement=label_agree, label_agreement_confident=label_agree_conf, parity_worst=worst, parity_pass=ok,
                  size_kb=round(os.path.getsize(path) / 1e3, 1))

    if ok:
        e = make_entry(NAME, path, 'model', ins, outs,
                       'input is 12 token ids (see dl9_vocab.json: lowercase, split on spaces, [CLS] first, [SEP] last, '
                       'pad with 0). Outputs: mlm_logits [1,12,%d] and attn_scores [1,4,12,12] = raw scores before '
                       'softmax, heads in the order layer0-head0, layer0-head1, layer1-head0, layer1-head1 (padding '
                       'keys are -10000). The app computes softmax(scores / temperature) itself.' % V)
        if args.parity == 'relative':
            e['parity'] = ('fp16 vs fp32 Keras: cosine >= 0.999 and mean |diff|/mean |ref| <= 0.05 per output '
                           '(padding entries of attn_scores ignored). Use the same rule with the golden files.')
        add_entry(e, MANIFEST_PART)
    print('\nRESULT:', 'PASS' if ok else 'FAIL', '|', path)


if __name__ == '__main__':
    main()
