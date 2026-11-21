"""Rebuild all 35 Person B packs and the separate manifest part."""
from train.person_b_utils import save_entries
from train import train_ml2,train_ml3,train_ml8,train_ml9,train_ml11_stacking

def main():
    entries=[]
    for mod,expected in [(train_ml2,2),(train_ml3,5),(train_ml8,12),(train_ml9,14),(train_ml11_stacking,2)]:
        batch=mod.run();assert len(batch)==expected,(mod.__name__,len(batch));entries+=batch
        print(mod.__name__,len(batch),'packs')
    assert len(entries)==35
    save_entries(entries)
    print('35 packs, 35 manifest entries')
if __name__=='__main__':main()
