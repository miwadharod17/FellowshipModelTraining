import datetime
import json
from common.seed import SEED


def write_metrics(name, dataset, settings, out_dir='out/metrics', **extra):
    d = {'name': name, 'date': datetime.date.today().isoformat(),
         'seed': SEED, 'dataset': dataset, 'settings': settings}
    d.update(extra)   # loss curves, accuracy, confusion matrix, etc.
    with open(f'{out_dir}/{name}_metrics.json', 'w') as f:
        json.dump(d, f, indent=2)
