"""ML-8: 12 required logistic regression packs."""
import numpy as np
from sklearn.datasets import load_breast_cancer,make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score,log_loss,confusion_matrix
from train.person_b_utils import SEED,emit,clean,fixed_split

def run():
    entries=[]
    d=load_breast_cancer(); cols=[i for i,n in enumerate(d.feature_names) if n.startswith('mean ')]
    X=d.data[:,cols];y=d.target
    # The guide specifies a 70/30 stratified split, fixed seed 42; BC split is not in shared data/splits.
    Xtr,Xte,ytr,yte=fixed_split('breast_cancer_mean10',X,y)
    scaler=StandardScaler().fit(Xtr);Ztr=scaler.transform(Xtr);Zte=scaler.transform(Xte)
    for pen in ['l1','l2']:
      for C in [.01,.1,1,10]:
        name=f'ml8_bc_{pen}_c{str(C).replace(".","p")}'
        model=LogisticRegression(l1_ratio=(1 if pen=='l1' else 0),C=C,solver='saga',max_iter=12000,tol=1e-3,random_state=SEED).fit(Ztr,ytr)
        probs=model.predict_proba(Zte)[:,1]; raw=model.decision_function(Zte)
        orig=model.coef_[0]/scaler.scale_; b_orig=model.intercept_[0]-np.dot(orig,scaler.mean_)
        frac,mean=calibration_curve(yte,probs,n_bins=10,strategy='uniform')
        arrays=dict(coef_scaled=model.coef_,intercept_scaled=model.intercept_,coef_original=orig,intercept_original=np.array([b_orig]),mean=scaler.mean_,scale=scaler.scale_,test_probs=probs,test_labels=yte,calibration_mean=mean,calibration_fraction=frac)
        metrics=dict(dataset='breast_cancer_mean10',settings=dict(penalty=pen,C=C),train_accuracy=model.score(Ztr,ytr),test_accuracy=model.score(Zte,yte),test_log_loss=log_loss(yte,probs),confusion_matrix=confusion_matrix(yte,model.predict(Zte)),converged=model.n_iter_[0]<model.max_iter)
        assert np.max(np.abs(raw-(Zte@model.coef_[0]+model.intercept_[0])))<1e-9
        entries.append(emit(name,arrays,dict(dataset='breast_cancer_mean10',penalty=pen,C=C,preprocessing='10 mean features standardized with stored train-only mean/scale',feature_names=[str(d.feature_names[i]) for i in cols]),metrics,'engine_linear_b1.tflite',labels=['malignant','benign'],golden=(Zte[:20],raw[:20])))
    X,y=make_blobs(n_samples=400,centers=[(-4,-4),(4,4)],cluster_std=.8,random_state=SEED)
    Xtr,Xte,ytr,yte=fixed_split('separable_blobs',X,y)
    scaler=StandardScaler().fit(Xtr);Ztr=scaler.transform(Xtr);Zte=scaler.transform(Xte);
    for C in [.01,1,100,10000]:
      name=f'ml8_separable_l2_c{str(C).replace(".","p")}'
      model=LogisticRegression(C=C,l1_ratio=0,solver='saga',max_iter=12000,tol=1e-3,random_state=SEED).fit(Ztr,ytr)
      raw=model.decision_function(Zte);probs=model.predict_proba(Zte)[:,1]
      orig=model.coef_[0]/scaler.scale_;b_orig=model.intercept_[0]-orig@scaler.mean_
      arrays=dict(coef_scaled=model.coef_,intercept_scaled=model.intercept_,coef_original=orig,intercept_original=[b_orig],mean=scaler.mean_,scale=scaler.scale_,test_probs=probs,test_labels=yte)
      metrics=dict(dataset='separable_blobs',settings=dict(C=C),train_accuracy=model.score(Ztr,ytr),test_accuracy=model.score(Zte,yte),coef_norm=float(np.linalg.norm(model.coef_)),converged=model.n_iter_[0]<model.max_iter)
      entries.append(emit(name,arrays,dict(dataset='separable_blobs',C=C,preprocessing='2 features standardized using stored mean/scale'),metrics,'engine_linear_b1.tflite',labels=['0','1'],golden=(Zte[:20],raw[:20])))
    assert len(entries)==12
    return entries
if __name__=='__main__':print(len(run()))
