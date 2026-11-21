"""NumPy reference for the proposed engine interface."""
import numpy as np
from engines.person_b_contract import *

def linear(features,weights,bias,feature_mask,class_mask):
    X=np.asarray(features,dtype=np.float32);W=np.asarray(weights,dtype=np.float32)
    out=(X*np.asarray(feature_mask,dtype=np.float32))@W+np.asarray(bias,dtype=np.float32)
    return (out*np.asarray(class_mask,dtype=np.float32)).astype(np.float32)

def kernel(features,support_vectors,dual_coef,bias,gamma,support_count,kernel_code):
    X=np.asarray(features,dtype=np.float32);S=np.asarray(support_vectors,dtype=np.float32)
    n=int(np.asarray(support_count).reshape(-1)[0]);g=float(np.asarray(gamma).reshape(-1)[0]);c=int(np.asarray(kernel_code).reshape(-1)[0]);assert 0<=n<=len(S)
    if c==0: K=X@S[:n].T
    elif c==1:K=np.exp(-g*np.sum((X[:,None,:]-S[None,:n,:])**2,axis=2))
    else:raise ValueError('kernel_code must be 0 or 1')
    return (K@np.asarray(dual_coef,dtype=np.float32)[:n]+np.asarray(bias,dtype=np.float32).reshape(-1)[0]).reshape(-1,1).astype(np.float32)

def distance(features,reference_points,feature_weights,reference_count,k):
    X=np.asarray(features,dtype=np.float32);R=np.asarray(reference_points,dtype=np.float32)
    n=int(np.asarray(reference_count).reshape(-1)[0]);k=int(np.asarray(k).reshape(-1)[0]);assert 1<=k<=MAX_K and k<=n<=len(R)
    w=np.asarray(feature_weights,dtype=np.float32);assert np.all(w>=0)
    # Accumulate squared differences feature-by-feature to avoid catastrophic
    # cancellation on large-scale features (e.g., Wine alcohol/proline).
    d2=np.zeros((len(X),n),dtype=np.float32)
    for j in range(X.shape[1]):
        d2+=w[j]*np.square(X[:,j,None]-R[None,:n,j])
    order=np.argsort(d2,axis=1,kind='stable')[:,:k];values=np.sqrt(np.take_along_axis(d2,order,axis=1))
    ds=np.full((len(X),MAX_K),1e15,dtype=np.float32);ix=np.full((len(X),MAX_K),-1,dtype=np.float32)
    ds[:,:k]=values;ix[:,:k]=order
    return ds,ix
