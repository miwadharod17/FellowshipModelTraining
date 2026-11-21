"""ML-11: leak-free OOF vs deliberately leaky in-fold stacking comparison."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from train.person_b_utils import SEED,splits,emit

def learners():
    return [KNeighborsClassifier(n_neighbors=1),SVC(C=100,gamma=10,kernel='rbf',probability=True,random_state=SEED),RandomForestClassifier(n_estimators=200,max_depth=12,max_features=2,random_state=SEED,n_jobs=1)]

def run():
    X,Xt,y,yt=splits('moons_main');cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    oof=np.zeros((len(X),3));infold=np.zeros_like(oof);test=np.zeros((len(Xt),3));entries=[]
    for j,model in enumerate(learners()):
      for tr,val in cv.split(X,y):
        fresh=learners()[j].fit(X[tr],y[tr]);oof[val,j]=fresh.predict_proba(X[val])[:,1]
      full=model.fit(X,y);infold[:,j]=full.predict_proba(X)[:,1];test[:,j]=full.predict_proba(Xt)[:,1]
    for name,Z in [('oof',oof),('infold',infold)]:
      meta=LogisticRegression(random_state=SEED,max_iter=2000).fit(Z,y)
      p=meta.predict_proba(test)[:,1];raw=meta.decision_function(test)
      arrays=dict(meta_coef=meta.coef_,meta_intercept=meta.intercept_,meta_features_train=Z,meta_features_test=test,labels_train=y,labels_test=yt,test_probabilities=p,base_oof=oof,base_infold=infold)
      metrics=dict(dataset='moons_main',settings=dict(mode=name,base_learners=['knn_1','svc_c100_g10','forest_200_depth12_m2']),meta_train_accuracy=meta.score(Z,y),heldout_accuracy=accuracy_score(yt,p>=.5),heldout_log_loss=log_loss(yt,p),leakage_warning=name=='infold')
      entries.append(emit('ml11_stacking_'+name,arrays,dict(dataset='moons_main',mode=name,preprocessing='Meta input: positive-class probability from KNN(1), SVC(C=100,gamma=10), RF(200,depth12,max_features2) in this order'),metrics,'engine_linear_b1.tflite',labels=['0','1'],golden=(test[:20],raw[:20])))
    assert len(entries)==2
    return entries
if __name__=='__main__':print(len(run()))
