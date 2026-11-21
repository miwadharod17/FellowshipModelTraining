"""Proposed Person B TFLite signature contract, v1. Dev approval required.

Every input float32; zero-pad unused slots. Count inputs are scalar tensors [1].
Batch 1 and 2500 have distinct fixed-shape files.
"""
FEATURES_LINEAR=32
CLASSES_LINEAR=4
FEATURES_KERNEL=2
MAX_SUPPORT=256
FEATURES_DISTANCE=13
MAX_REFERENCES=1000
MAX_K=30

def contract(engine,batch):
    assert batch in (1,2500)
    if engine=='linear':
      return dict(inputs={'features':[batch,FEATURES_LINEAR], 'weights':[FEATURES_LINEAR,CLASSES_LINEAR], 'bias':[CLASSES_LINEAR], 'feature_mask':[FEATURES_LINEAR], 'class_mask':[CLASSES_LINEAR]},outputs={'scores':[batch,CLASSES_LINEAR]},description='Raw scores; app applies sigmoid, softmax, or exponential. Mask unused features/classes.')
    if engine=='kernel':
      return dict(inputs={'features':[batch,FEATURES_KERNEL], 'support_vectors':[MAX_SUPPORT,FEATURES_KERNEL], 'dual_coef':[MAX_SUPPORT], 'bias':[1], 'gamma':[1], 'support_count':[1], 'kernel_code':[1]},outputs={'decision':[batch,1]},description='kernel_code=0 linear, 1 RBF; dual_coef contains signed alpha*y; pad support vectors and coefficients with zeros.')
    if engine=='distance':
      return dict(inputs={'features':[batch,FEATURES_DISTANCE], 'reference_points':[MAX_REFERENCES,FEATURES_DISTANCE], 'feature_weights':[FEATURES_DISTANCE], 'reference_count':[1], 'k':[1]},outputs={'distances':[batch,MAX_K],'indices':[batch,MAX_K]},description='Euclidean distances with weighted squared feature differences; valid first k entries, remaining entries masked with +large distance and index -1; references beyond count ignored.')
    raise ValueError(engine)
