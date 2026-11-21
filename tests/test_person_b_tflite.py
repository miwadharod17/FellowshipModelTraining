"""Real TFLite interpreter parity against Person B packs. Requires TensorFlow and built engines.
Run after python -m engines.build_engine_person_b.
"""
import json
import numpy as np
from common.packs import read_pack
from train.person_b_utils import OUT
from engines.person_b_contract import *

def padded(a,shape):
    v=np.zeros(shape,dtype=np.float32);a=np.asarray(a,dtype=np.float32)
    slices=tuple(slice(0,n) for n in a.shape)
    assert len(shape)==a.ndim and all(x<=y for x,y in zip(a.shape,shape))
    v[slices]=a
    return v

def inputs_for(name,a,X,batch):
    X=np.asarray(X,dtype=np.float32)
    if len(X)<batch:X=np.concatenate([X,np.repeat(X[-1:],batch-len(X),axis=0)],axis=0)
    else:X=X[:batch]
    if name.startswith(('ml8_','ml11_')):
        coef=a['coef_scaled'] if name.startswith('ml8_') else a['meta_coef']
        bias=a['intercept_scaled'] if name.startswith('ml8_') else a['meta_intercept']
        features=padded(X,(batch,FEATURES_LINEAR))
        weights=padded(coef.T,(FEATURES_LINEAR,CLASSES_LINEAR))
        return 'linear',dict(features=features,weights=weights,bias=padded(bias,(CLASSES_LINEAR,)),feature_mask=padded(np.ones(X.shape[1]),(FEATURES_LINEAR,)),class_mask=np.array([1,0,0,0],dtype='float32'))
    if name.startswith('ml9_'):
        SV=a.get('engine_support_vectors',a['support_vectors']);coef=a.get('engine_dual_coef',a['dual_coef'])
        return 'kernel',dict(features=padded(X,(batch,FEATURES_KERNEL)),support_vectors=padded(SV,(MAX_SUPPORT,FEATURES_KERNEL)),dual_coef=padded(coef,(MAX_SUPPORT,)),bias=a['intercept'].astype('float32'),gamma=a['gamma'].astype('float32'),support_count=np.array([len(SV)],dtype='float32'),kernel_code=np.array([0 if 'linear' in name else 1],dtype='float32'))
    if name.startswith('ml2_'):
        R=a['reference_points'];k=5
    elif name.startswith('ml3_'):
        R=a['centroid_history'][0,0,0,-1,:2];k=1
    else:raise ValueError(name)
    return 'distance',dict(features=padded(X,(batch,FEATURES_DISTANCE)),reference_points=padded(R,(MAX_REFERENCES,FEATURES_DISTANCE)),feature_weights=padded(np.ones(X.shape[1]),(FEATURES_DISTANCE,)),reference_count=np.array([len(R)],dtype='float32'),k=np.array([k],dtype='float32'))

def run():
    import tensorflow as tf
    entries=[e for e in json.loads((OUT/'manifest_b.json').read_text()) if e.get('type')=='pack']
    for batch in [1,2500]:
      chosen=entries if batch==1 else [next(e for e in entries if e['id'].startswith(p)) for p in ('ml8_','ml9_','ml2_')]
      for e in chosen:
        name=e['id'];a,_=read_pack(str(OUT/'packs'/name));X=np.load(OUT/'golden'/f'{name}_golden_in.npy');Y=np.load(OUT/'golden'/f'{name}_golden_out.npy')
        kind,inp=inputs_for(name,a,X,batch)
        path=OUT/'models'/f'engine_{kind}_b{batch}.tflite'
        interpreter=tf.lite.Interpreter(model_path=str(path));interpreter.allocate_tensors()
        got=interpreter.get_signature_runner()(**inp)
        count=min(len(Y),batch)
        if kind=='linear':result=got['scores'][:count,0];expected=Y[:count]
        elif kind=='kernel':result=got['decision'][:count,0];expected=Y[:count]
        else:result=got['distances'][:count,:Y.shape[1]] if Y.ndim==2 else got['distances'][:count,0];expected=Y[:count]
        err=float(np.max(np.abs(result-expected)))
        assert err<=1e-4,(name,batch,err)
        print(name,'batch',batch,'PASS',err)
    print('TFLite pack parity PASS')
if __name__=='__main__':run()
