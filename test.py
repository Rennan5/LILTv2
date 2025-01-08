import argparse
import json
import os
import numpy as np

from utils.dataset import load_cord_dataset, load_custom_dataset, load_mixed_dataset
from utils.lambert.model import LambertModel
from utils.layoutlm.model import LayoutLMModel
# from utils.layoutlmv2.model import LayoutLMv2Model
from utils.layoutlmv3.model import LayoutLMv3Model
from utils.lilt.model import LiltModel
from utils.save import save_metrics, save_results


def process_results(results, other_label):
    processed_result = {}

    for filename, result in results.items():
        processed_result[filename] = {
            'texts': result['texts'], 'bbox': [],  'labels': [],
            'predictions_logits': [], 'predictions': []
        }

        for i, texts in enumerate(result['texts']):
            try:
                word_id = result['word_ids'].index(i)
            except:
                word_id = None

            if word_id is not None:
                l = result['labels'][word_id]
                b = result['bbox'][word_id]
                p_log = result['predictions_logits'][word_id]
                p = np.argmax(p_log)
            else:
                l = other_label
                b = [-1, -1, -1, -1]
                p_log = []
                p = other_label


            assert l != -100

            processed_result[filename]['bbox'].append(b)
            processed_result[filename]['labels'].append(l)
            processed_result[filename]['predictions_logits'].append(p_log)
            processed_result[filename]['predictions'].append(p)

        assert len(processed_result[filename]['bbox']) == len(processed_result[filename]['texts']) and \
               len(processed_result[filename]['bbox']) == len(processed_result[filename]['labels']) and \
               len(processed_result[filename]['bbox']) == len(processed_result[filename]['predictions_logits']) and \
               len(processed_result[filename]['bbox']) == len(processed_result[filename]['predictions'])

    return processed_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model-path', type=str, default=None)

    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    # Args
    try:
        with open(os.path.join(args.model_path, 'info.json'), 'r') as f:
            info_data = json.loads(f.read())
        model_type = info_data['model_type'].upper()
        labels = info_data['labels']
    except:
        model_type = 'LILT'
        labels = None

    # Dataset parameters
    load_image = bool(model_type in ('LAYOUTLMV2', 'LAYOUTLMV3'))
    bbox_scale = 1 if model_type not in ('LAYOUTLMV3', ) else 1000

    # Dataset
    if args.dataset == 'cord':
        dataset, dataset_labels = load_cord_dataset(
            load_image=load_image, smoke_test=args.smoke_test, bbox_scale=bbox_scale
        )
    elif args.dataset in ('cord_mixed', ):
        dataset, dataset_labels = load_mixed_dataset(
            load_image=load_image, smoke_test=args.smoke_test, bbox_scale=bbox_scale
        )
    else:
        dataset, dataset_labels = load_custom_dataset(
            f'dataset/{ args.dataset }/', load_image=load_image, smoke_test=args.smoke_test, bbox_scale=bbox_scale
        )

    if labels is None:
        labels = dataset_labels

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    if args.smoke_test:
        dataset['train'] = dataset['train'].select([0, 1, 2])
        dataset['val'] = dataset['val'].select([0, 1, 2])
        dataset['test'] = dataset['test'].select([0, 1, 2])

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")
    print(f"Available labels: {labels}")

    # Loading
    if model_type == 'LILT':
        model = LiltModel(args.model_path, labels)
    elif model_type == 'LAYOUTLMV3':
        model = LayoutLMv3Model(args.model_path, labels)
    elif model_type == 'LAMBERT_ROBERTA':
        model = LambertModel(args.model_path, labels, 'ROBERTA')
    elif model_type == 'LAMBERT_BERTIMBAU':
        model = LambertModel(args.model_path, labels, 'BERTIMBAU')
    elif model_type == 'LAYOUTLM':
        model = LayoutLMModel(args.model_path, labels)
    # elif model_type == 'LAYOUTLMV2':
    #     model = LayoutLMv2Model(args.model_path, labels)
    else:
        raise Exception
    model.load(os.path.join(args.model_path, "weights.pt"))

    # Results
    test_results, test_metrics, test_duration = model.predict(dataset['test'])
    print(test_metrics)

    _test_results = process_results(test_results, max(id2label.keys()))
    save_results(os.path.join(args.model_path, f'test_{ args.dataset }'), _test_results)
    save_metrics(os.path.join(args.model_path, f'test_{ args.dataset }'), test_metrics)
