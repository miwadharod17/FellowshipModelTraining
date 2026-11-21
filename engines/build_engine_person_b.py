"""Build 6 TFLite engines with builtin ops only. Requires TensorFlow (pinned by team).

Run: python -m engines.build_engine_person_b
Each signature is fixed shape and all inputs float32. Conversion MUST be verified
with TensorFlow Lite Interpreter and developer browser/Android runtimes.
"""
import json,hashlib,datetime
from pathlib import Path
import numpy as np
from engines.person_b_contract import contract,MAX_K,MAX_REFERENCES,MAX_SUPPORT
from train.person_b_utils import OUT,SEED,save_entries

def build(only=None):
    if only is not None and only not in ('linear','kernel','distance'):
        raise ValueError(f'Unknown engine: {only}')
    import tensorflow as tf
    tf.random.set_seed(SEED)
    entries=[]
    for engine in ([only] if only else ['linear','kernel','distance']):
      for batch in [1,2500]:
        spec=contract(engine,batch)
        input_specs={k:tf.TensorSpec(v,tf.float32,name=k) for k,v in spec['inputs'].items()}
        class Module(tf.Module):
          @tf.function(input_signature=[input_specs])
          def serve(self,args):
            if engine=='linear':
              X=args['features']*args['feature_mask'][None,:]
              scores=tf.linalg.matmul(X,args['weights'])+args['bias'][None,:]
              return {'scores':scores*args['class_mask'][None,:]}
            if engine=='kernel':
              X=args['features'];S=args['support_vectors']
              linear=tf.linalg.matmul(X,S,transpose_b=True)
              sq=tf.reduce_sum(tf.square(X[:,None,:]-S[None,:,:]),axis=2)
              rbf=tf.exp(-args['gamma'][0]*sq)
              K=tf.where(tf.equal(args['kernel_code'][0],1.),rbf,linear)
              mask=tf.cast(tf.range(MAX_SUPPORT)<tf.cast(args['support_count'][0],tf.int32),tf.float32)
              terms=K*(args['dual_coef']*mask)[None,:]
              total=tf.zeros((batch,),dtype=tf.float32)
              compensation=tf.zeros_like(total)
              for j in range(MAX_SUPPORT):
                y=terms[:,j]-compensation
                t=total+y
                compensation=(t-total)-y
                total=t
              decision=total+args['bias'][0]
              return {'decision':decision[:,None]}
            X=args['features'];R=args['reference_points'];w=tf.maximum(args['feature_weights'],0.)
            # Fixed 13-feature accumulation avoids subtracting nearly equal large norms.
            sq=tf.zeros((batch,MAX_REFERENCES),dtype=tf.float32)
            for j in range(13):
              delta=X[:,j:j+1]-R[None,:,j]
              sq=sq+w[j]*tf.square(delta)
            mask=tf.range(MAX_REFERENCES)[None,:]>=tf.cast(args['reference_count'][0],tf.int32)
            sq=tf.where(mask,tf.fill(tf.shape(sq),tf.constant(1e30,tf.float32)),sq)
            # fixed top 30; input k masks entries after the requested count
            vals,idx=tf.math.top_k(-sq,k=MAX_K,sorted=True)
            valid=tf.range(MAX_K)[None,:]<tf.cast(args['k'][0],tf.int32)
            distances=tf.where(valid,tf.sqrt(tf.maximum(0.,-vals)),tf.fill(tf.shape(vals),tf.constant(1e15,tf.float32)))
            indices=tf.where(valid,tf.cast(idx,tf.float32),tf.fill(tf.shape(vals),tf.constant(-1.,tf.float32)))
            return {'distances':distances,'indices':indices}
        module=Module();concrete=module.serve.get_concrete_function()
        converter=tf.lite.TFLiteConverter.from_concrete_functions([concrete],module)
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
        name=f'engine_{engine}_b{batch}';p=OUT/'models'/f'{name}.tflite';p.parent.mkdir(exist_ok=True,parents=True)
        p.write_bytes(converter.convert())
        interpreter=tf.lite.Interpreter(model_path=str(p));interpreter.allocate_tensors()
        signatures=interpreter.get_signature_list();assert signatures,'TFLite signature missing'
        runner=interpreter.get_signature_runner()
        rng=np.random.default_rng(SEED);inp={k:rng.normal(size=shape).astype('float32') for k,shape in spec['inputs'].items()}
        if engine=='kernel':
          inp['gamma']=np.array([.7],dtype='float32');inp['support_count']=np.array([100],dtype='float32');inp['kernel_code']=np.array([1],dtype='float32')
        if engine=='distance':
          inp['reference_count']=np.array([100],dtype='float32');inp['k']=np.array([10],dtype='float32');inp['feature_weights']=np.ones(inp['feature_weights'].shape,dtype='float32')
        if engine=='linear':inp['feature_mask'][:]=1;inp['class_mask'][:]=1
        from engines.person_b_numpy import linear,kernel,distance
        if engine=='linear':expected={'scores':linear(**inp)}
        elif engine=='kernel':expected={'decision':kernel(**inp)}
        else:
          d,idx=distance(**inp);expected={'distances':d,'indices':idx}
        output=runner(**inp)
        for key in expected:
          assert key in output,(name,key,output.keys())
          if engine=='distance':
            # tie order can vary; compare distances and unordered nearest neighbors if needed
            n=10
            if key=='indices':
              assert np.allclose(output[key][:,:n],expected[key][:,:n],atol=0),name
            else:assert np.allclose(output[key][:,:n],expected[key][:,:n],atol=1e-3),name
          else:assert np.allclose(output[key],expected[key],atol=1e-4,rtol=1e-5),name
        g=OUT/'golden';g.mkdir(parents=True,exist_ok=True)
        np.savez(g/f'{name}_golden_inputs.npz',**inp)
        np.savez(g/f'{name}_golden_outputs.npz',**expected)
        m=dict(name=name,date=datetime.date.today().isoformat(),seed=SEED,kind='engine',parity='passed',inputs=spec['inputs'],outputs=spec['outputs'])
        (OUT/'metrics'/f'{name}_metrics.json').write_text(json.dumps(m,indent=2))
        entries.append(dict(id=name,type='engine',version=1,file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),inputs=[dict(name=k,shape=v,dtype='float32',range='model-dependent') for k,v in spec['inputs'].items()],outputs=[dict(name=k,shape=v) for k,v in spec['outputs'].items()],labels=None,preprocessing=spec['description']))
        print(name,'built and interpreter parity passed')
    part=OUT/'manifest_b.json';previous=json.loads(part.read_text()) if part.exists() else []
    save_entries([e for e in previous if e['type']=='pack' or (e.get('type')=='engine' and e['id'] not in {new['id'] for new in entries})]+entries)
if __name__=='__main__':build()
