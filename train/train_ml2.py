"""ML-2: KNN two packs, scaling/noise curves and precomputed algorithm benchmark."""
import json,time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score
from train.person_b_utils import SEED,splits,emit,OUT

def run():
    rng=np.random.default_rng(SEED);entries=[];bench=[]
    for dataset in ['moons_noise10','wine']:
      if dataset=='wine':X,Xt,y,yt=splits('wine');raw=np.vstack([X,Xt]);base_features=X.shape[1]
      else:
        X,Xt,y,yt=splits('moons_main');
        X=X.copy();Xt=Xt.copy();X[:,0]*=50;Xt[:,0]*=50
        X=np.column_stack([X,rng.normal(size=(len(X),10))]);Xt=np.column_stack([Xt,rng.normal(size=(len(Xt),10))]);raw=np.vstack([X,Xt]);base_features=2
      scaler=StandardScaler().fit(X)
      curves=[]
      for scaled in [False,True]:
        for noise in ([0,2,5,10] if dataset=='moons_noise10' else [0]):
          count=base_features+noise if dataset=='moons_noise10' else base_features
          data=(scaler.transform(X) if scaled else X)[:,:count];test=(scaler.transform(Xt) if scaled else Xt)[:,:count]
          for k in range(1,31):
            m=KNeighborsClassifier(n_neighbors=k,weights='uniform').fit(data,y)
            cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
            score=cross_val_score(m,data,y,cv=cv,n_jobs=1).mean()
            curves.append(dict(scaled=scaled,noise_features=noise,k=k,train_accuracy=m.score(data,y),cv_accuracy=float(score),test_accuracy=m.score(test,yt)))
      full=KNeighborsClassifier(n_neighbors=5).fit(X,y)
      dist,idx=full.kneighbors(Xt[:20]);assert np.all(idx>=0)
      from engines.person_b_numpy import distance
      from engines.person_b_contract import MAX_REFERENCES,FEATURES_DISTANCE
      R=np.zeros((MAX_REFERENCES,FEATURES_DISTANCE),dtype=np.float32);R[:len(X),:X.shape[1]]=X
      Q=np.zeros((20,FEATURES_DISTANCE),dtype=np.float32);Q[:,:X.shape[1]]=Xt[:20]
      weights=np.zeros(FEATURES_DISTANCE,dtype=np.float32);weights[:X.shape[1]]=1
      ref_dist,_=distance(Q,R,weights,[len(X)],[5])
      assert np.max(np.abs(ref_dist[:,:5]-dist))<2e-4
      name='ml2_'+dataset
      arrays=dict(reference_points=X,reference_labels=y,train_mean=scaler.mean_,train_scale=scaler.scale_,test_points=Xt,test_labels=yt,feature_count=[X.shape[1]])
      meta=dict(dataset=dataset,preprocessing='Training points already include stretch/noise when applicable; toggle standardization via stored train-only mean/scale',noise_feature_count=10 if dataset=='moons_noise10' else 0)
      metrics=dict(dataset=dataset,settings=dict(k_values=list(range(1,31)),scaling=[False,True]),curves=curves)
      entries.append(emit(name,arrays,meta,metrics,'engine_distance_b1.tflite',labels=[str(v) for v in np.unique(y)],golden=(Xt[:20],ref_dist[:,:5])))
      for algorithm in ['brute','kd_tree','ball_tree']:
        for dim in [2,10,30]:
          sample=rng.normal(size=(1000,dim)).astype('float32');query=rng.normal(size=(250,dim)).astype('float32')
          t=time.perf_counter();m=KNeighborsClassifier(n_neighbors=5,algorithm=algorithm).fit(sample,np.arange(1000)%2);fit=time.perf_counter()-t
          t=time.perf_counter();m.kneighbors(query);elapsed=time.perf_counter()-t
          bench.append(dict(algorithm=algorithm,dimension=dim,reference_count=1000,query_count=250,fit_seconds=fit,query_seconds=elapsed))
    p=OUT/'precomputed'/'ml2_knn_benchmark.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(dict(label='precomputed benchmark, not live engine execution',seed=SEED,measurements=bench),indent=2))
    assert len(entries)==2
    return entries
if __name__=='__main__':print(len(run()))
