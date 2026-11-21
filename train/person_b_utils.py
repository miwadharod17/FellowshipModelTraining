"""Person B-only artifact utilities; does not modify shared common/ code."""
import hashlib, json, datetime
from pathlib import Path
import numpy as np
from common.packs import write_pack, read_pack
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out'
SEED=42

def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.floating,)):return float(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    if isinstance(v,Path):return str(v)
    return v

def splits(name):
    d=np.load(ROOT/'data'/'splits'/f'{name}.npz')
    return tuple(d[k] for k in ['X_train','X_test','y_train','y_test'])

def emit(name,arrays,meta,metrics,engine,labels=None,golden=None):
    for a in arrays.values():
        assert np.isfinite(np.asarray(a)).all(),f'{name}: nonfinite array'
    p=OUT/'packs'/name; p.parent.mkdir(parents=True,exist_ok=True)
    meta=dict(meta,seed=SEED,engine=engine)
    write_pack(str(p),arrays,clean(meta))
    check,_=read_pack(str(p))
    for k,a in arrays.items():
        assert np.array_equal(check[k],np.asarray(a,dtype=np.float32)),(name,k)
    m=dict(name=name,date=datetime.date.today().isoformat(),seed=SEED,**metrics)
    q=OUT/'metrics'/f'{name}_metrics.json';q.parent.mkdir(exist_ok=True,parents=True)
    q.write_text(json.dumps(clean(m),indent=2,allow_nan=False))
    if golden:
        g=OUT/'golden';g.mkdir(exist_ok=True,parents=True)
        np.save(g/f'{name}_golden_in.npy',np.asarray(golden[0],dtype=np.float32))
        np.save(g/f'{name}_golden_out.npy',np.asarray(golden[1],dtype=np.float32))
    idx=json.loads(p.with_suffix('.json').read_text())['arrays']
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    from engines.person_b_contract import contract
    engine_kind=engine.split('_')[1]
    signature=contract(engine_kind,1)
    entry=dict(id=name,type='pack',version=1,file=p.with_suffix('.bin').name,sha256=sha(p.with_suffix('.bin')),
       index_file=p.with_suffix('.json').name,index_sha256=sha(p.with_suffix('.json')),
       inputs=[dict(name=k,shape=v,dtype='float32',range='model-dependent') for k,v in signature['inputs'].items()],
       outputs=[dict(name=k,shape=v) for k,v in signature['outputs'].items()],labels=labels,
       preprocessing=meta.get('preprocessing','See pack metadata'),engine=engine,
       metrics_file=q.name,arrays=idx,engine_variants=[f'engine_{engine_kind}_b1.tflite',f'engine_{engine_kind}_b2500.tflite'])
    return entry

def save_entries(entries):
    p=OUT/'manifest_b.json';p.write_text(json.dumps(clean(entries),indent=2,allow_nan=False))
    return p

def label_accuracy(y,p):return float(np.mean(np.asarray(y)==np.asarray(p)))


def fixed_split(name,X,y):
    """Create a 70/30 stratified split once and reuse its exact saved indices."""
    from sklearn.model_selection import train_test_split
    p=ROOT/'data'/'splits'/f'{name}.npz'
    if p.exists():
        d=np.load(p);return d['X_train'],d['X_test'],d['y_train'],d['y_test']
    tr,te=train_test_split(np.arange(len(y)),test_size=.30,stratify=y,random_state=SEED)
    p.parent.mkdir(exist_ok=True,parents=True)
    np.savez(p,X_train=X[tr],X_test=X[te],y_train=y[tr],y_test=y[te])
    return X[tr],X[te],y[tr],y[te]
