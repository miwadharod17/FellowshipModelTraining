"""ML-9: 14 required SVC packs, with sklearn decision-function parity."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from train.person_b_utils import SEED,splits,emit,fixed_split

def code(x):return str(x).replace('.','p')

def run():
    entries=[];xm,xt,ym,yt=splits('moons_main')
    xb,yb=make_blobs(n_samples=400,centers=[(-1,0),(1,0)],cluster_std=1.4,random_state=SEED)
    xtr,xte,ytr,yte=fixed_split('overlap_blobs',xb,yb)
    for mode in ['rbf','linear','scale']:
      settings=([(c,g) for c in [.1,1,100] for g in [.1,1,10]] if mode=='rbf' else [(c,None) for c in [.01,1,100]] if mode=='linear' else [(1,None),(1,None)])
      for i,(C,g) in enumerate(settings):
        if mode=='linear':X,Xt,y,ytest=xtr,xte,ytr,yte; kernel='linear';suffix=f'ml9_blobs_linear_c{code(C)}';transform='none'
        elif mode=='rbf':X,Xt,y,ytest=xm,xt,ym,yt;kernel='rbf';suffix=f'ml9_moons_c{code(C)}_g{code(g)}';transform='none'
        else:
          X=xm.copy();Xt=xt.copy();X[:,0]*=50;Xt[:,0]*=50
          if i==0:
            scaler=StandardScaler().fit(X);X=scaler.transform(X);Xt=scaler.transform(Xt);transform='scaled';suffix='ml9_moons_scale_scaled'
          else:transform='unscaled';suffix='ml9_moons_scale_unscaled'
          y,ytest=ym,yt;kernel='rbf';g=1
        model=SVC(C=C,kernel=kernel,gamma=g if g is not None else 'scale',probability=True,random_state=SEED).fit(X,y)
        sv=model.support_vectors_.astype('float32');coeff=model.dual_coef_[0].astype('float32');bias=float(model.intercept_[0]);
        pred=np.sum((X[:20,None,:]@sv.T[None,:,:]).reshape(20,-1)*coeff,axis=1)+bias if kernel=='linear' else (np.exp(-g*np.sum((X[:20,None,:]-sv[None,:,:])**2,axis=2))@coeff+bias)
        assert np.max(np.abs(pred-model.decision_function(X[:20])))<2e-4,(suffix,'kernel parity')
        alpha=np.abs(coeff); bounded=alpha>=C-1e-4
        arrays=dict(support_vectors=sv,dual_coef=coeff,intercept=[bias],gamma=[g if g is not None else 0],support_indices=model.support_,alphas=alpha,free_mask=(~bounded).astype('float32'),bounded_mask=bounded.astype('float32'),platt_A=model.probA_,platt_B=model.probB_,classes=model.classes_)
        if kernel=='linear':
          # Avoid float32 cancellation when summing hundreds of support-vector terms.
          # Retain the original support vectors for visualization and store a single
          # effective linear weight vector for accurate engine inference.
          arrays['engine_support_vectors']=np.asarray(model.coef_,dtype=np.float32)
          arrays['engine_dual_coef']=np.array([1.],dtype=np.float32)
        if mode=='scale':
          arrays['feature_stretch']=np.array([50.,1.],dtype='float32')
          if i==0:arrays['scale_mean']=scaler.mean_;arrays['scale_std']=scaler.scale_
        metrics=dict(dataset='moons_main' if mode!='linear' else 'overlap_blobs',settings=dict(kernel=kernel,C=C,gamma=g,scaling=transform),train_accuracy=model.score(X,y),test_accuracy=model.score(Xt,ytest),n_support=model.n_support_,free_count=int((~bounded).sum()),bounded_count=int(bounded.sum()))
        meta=dict(dataset=metrics['dataset'],kernel=kernel,C=C,gamma=g,scaling=transform,preprocessing='Input transformed according to pack scaling metadata; use stored support vectors in the same coordinates',n_support=len(sv),feature_count=sv.shape[1])
        from engines.person_b_numpy import kernel as kernel_reference
        from engines.person_b_contract import MAX_SUPPORT
        engine_sv=arrays.get('engine_support_vectors',sv);engine_coef=arrays.get('engine_dual_coef',coeff)
        padded_sv=np.zeros((MAX_SUPPORT,2),dtype=np.float32);padded_sv[:len(engine_sv)]=engine_sv
        padded_coef=np.zeros(MAX_SUPPORT,dtype=np.float32);padded_coef[:len(engine_coef)]=engine_coef
        reference=kernel_reference(X[:20].astype('float32'),padded_sv,padded_coef,[bias],[g or 0],[len(engine_sv)],[0 if kernel=='linear' else 1]).ravel()
        assert np.max(np.abs(reference-model.decision_function(X[:20])))<3e-4
        entries.append(emit(suffix,arrays,meta,metrics,'engine_kernel_b1.tflite',labels=[str(v) for v in model.classes_],golden=(X[:20],reference)))
    assert len(entries)==14
    return entries
if __name__=='__main__':print(len(run()))
