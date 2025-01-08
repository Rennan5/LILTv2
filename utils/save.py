import json
import os

from utils.globals import NpEncoder


def save_results(path, results):
    _results = {}

    os.makedirs(os.path.join(path, 'results', 'raw'), exist_ok=True)
    for filename, result in results.items():
        _results = []
        for i in range(len(result['texts'])):
            _results.append({
                'text': result['texts'][i],
                'bbox': result['bbox'][i],
                'label': result['labels'][i],
                'prediction': result['predictions'][i]
            })

        with open(os.path.join(path, 'results', 'raw', f'{ filename }.json'), 'w') as f:
            f.write(json.dumps(
                _results, indent=4, ensure_ascii=False, cls=NpEncoder
            ))

def save_metrics(path, metrics):
    os.makedirs(os.path.join(path, 'results'), exist_ok=True)
    with open(os.path.join(path, 'results', 'metrics.json'), 'w') as f:
        f.write(json.dumps(
            metrics, indent=4, ensure_ascii=False, cls=NpEncoder
        ))