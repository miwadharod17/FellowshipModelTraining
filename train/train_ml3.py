"""ML-3: record K-means histories for all K/seeding/restarts, no retraining in apps."""
import numpy as np
from sklearn.datasets import make_blobs,make_moons,make_circles
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from train.person_b_utils import SEED,emit

def init_centers(X,k,method,rng):
    if method=='random':return X[rng.choice(len(X),k,replace=False)].copy()
    chosen=[int(rng.integers(len(X)))];nearest=np.sum((X-X[chosen[0]])**2,axis=1)
    for _ in range(1,k):
        if nearest.sum()<=1e-12:
            remaining=np.setdiff1d(np.arange(len(X)),chosen);j=int(rng.choice(remaining))
        else:j=int(rng.choice(len(X),p=nearest/nearest.sum()))
        chosen.append(j);nearest=np.minimum(nearest,np.sum((X-X[j])**2,axis=1))
    return X[chosen].copy()

def lloyd(X,k,method,seed,max_iter=30):
    rng=np.random.default_rng(seed);C=init_centers(X,k,method,rng);hist=[];loss=[]
    for it in range(max_iter):
        D=np.sum((X[:,None,:]-C[None,:,:])**2,axis=2);labels=D.argmin(axis=1)
        new=C.copy()
        for j in range(k):
            if np.any(labels==j):new[j]=X[labels==j].mean(axis=0)
            else:new[j]=X[np.argmax(np.min(D,axis=1))]
        C=new
        d=np.sum((X[:,None,:]-C[None,:,:])**2,axis=2)
        J=float(np.min(d,axis=1).sum());hist.append(C.copy());loss.append(J)
        if len(hist)>1 and np.allclose(hist[-1],hist[-2],atol=1e-7):break
    return np.asarray(hist),np.asarray(loss),np.min(np.sum((X[:,None,:]-C[None,:,:])**2,axis=2),axis=1).sum()

def dataset(name):
    if name=='blobs':X,_=make_blobs(n_samples=600,centers=[(-3,-2),(0,3),(3,-2)],cluster_std=.85,random_state=SEED)
    elif name=='moons':X,_=make_moons(n_samples=600,noise=.12,random_state=SEED)
    elif name=='rings':X,_=make_circles(n_samples=600,factor=.45,noise=.08,random_state=SEED)
    else:
      frac=.05 if name=='blobs_out5' else .10;n=int(600*(1-frac));X,_=make_blobs(n_samples=n,centers=[(-3,-2),(0,3),(3,-2)],cluster_std=.85,random_state=SEED)
      rng=np.random.default_rng(SEED);X=np.vstack([X,rng.uniform(-10,10,size=(600-n,2))])
    return np.asarray(X,dtype=np.float32)

def run():
    entries=[]
    for name in ['blobs','moons','rings','blobs_out5','blobs_out10']:
      X=dataset(name);histories=[];losses=[];counts=[];summaries=[];best_centers=[]
      # Dense arrays: [7 K values, 2 seed methods, 10 restarts, 30 steps, 8 centers, 2 coordinates]
      H=np.zeros((7,2,10,30,8,2),dtype=np.float32);J=np.zeros((7,2,10,30),dtype=np.float32);lengths=np.zeros((7,2,10),dtype=np.float32)
      for ki,k in enumerate(range(2,9)):
        for mi,method in enumerate(['random','k-means++']):
          for restart in range(10):
            h,loss,_=lloyd(X,k,method,SEED+ki*1000+mi*100+restart)
            n=len(loss);H[ki,mi,restart,:n,:k]=h;H[ki,mi,restart,n:,:k]=h[-1]
            J[ki,mi,restart,:n]=loss;J[ki,mi,restart,n:]=loss[-1];lengths[ki,mi,restart]=n
          best=int(J[ki,mi,:,-1].argmin());C=H[ki,mi,best,-1,:k]
          labels=np.argmin(np.sum((X[:,None,:]-C[None,:,:])**2,axis=2),axis=1)
          silhouette=float(silhouette_score(X,labels)) if len(np.unique(labels))>1 else None
          summaries.append(dict(k=k,initialization=method,best_restart=best,objective=float(J[ki,mi,best,-1]),silhouette=silhouette))
      arrays=dict(points=X,centroid_history=H,objective_history=J,iteration_counts=lengths,k_values=np.arange(2,9),initialization_codes=np.arange(2),restart_count=[10])
      metrics=dict(dataset=name,settings=dict(k_values=list(range(2,9)),seeding=['random','k-means++'],restarts=10,max_steps=30),summary=summaries)
      entries.append(emit('ml3_'+name,arrays,dict(dataset=name,preprocessing='2D points; select recorded K, seed method, restart and iteration',history_shape=list(H.shape)),metrics,'engine_distance_b1.tflite',golden=(X[:20],np.min(np.sum((X[:20,None,:]-H[0,0,0,-1,:2][None,:,:])**2,axis=2),axis=1)**.5)))
    assert len(entries)==5
    return entries
if __name__=='__main__':print(len(run()))
