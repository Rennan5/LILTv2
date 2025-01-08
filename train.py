import argparse
import datetime
import os

import numpy as np

from utils.dataset import load_cord_dataset, load_custom_dataset, load_mixed_dataset
from utils.lambert.model import LambertModel
from utils.layoutlm.model import LayoutLMModel
from utils.layoutlmv2.model import LayoutLMv2Model
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
    parser.add_argument('--model-type', type=str, required=True)

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-path', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=None)

    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    args.model_type = args.model_type.upper()

    # IDs
    if args.output_path is None:
        repository_id = os.path.join(
            'output', f'{ args.dataset }_{args.model_type}_model_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        )
    else:
        repository_id = args.output_path

    # Dataset parameters
    load_image = bool(args.model_type in ('LAYOUTLMV2', 'LAYOUTLMV3'))
    bbox_scale = 1 if args.model_type not in ('LAYOUTLMV3', ) else 1000

    # Dataset
    if args.dataset == 'cord':
        dataset, labels = load_cord_dataset(
            load_image=load_image, smoke_test=args.smoke_test, bbox_scale=bbox_scale
        )
    elif args.dataset in ('cord_mixed', ):
        dataset, labels = load_mixed_dataset(
            load_image=load_image, smoke_test=args.smoke_test, bbox_scale=bbox_scale
        )
    else:
        dataset, labels = load_custom_dataset(
            f'dataset/{ args.dataset }/', load_image=load_image, smoke_test=args.smoke_test, bbox_scale=bbox_scale
        )

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    if args.smoke_test:
        dataset['train'] = dataset['train'].select([0, 1, 2])
        dataset['val'] = dataset['val'].select([0, 1, 2])
        dataset['test'] = dataset['test'].select([0, 1, 2])

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")
    print(f"Available labels: {labels}")

    # Training
    model_type = args.model_type.upper()
    if model_type == 'LILT':
        model = LiltModel(repository_id, labels)
    elif model_type == 'LAYOUTLMV3':
        model = LayoutLMv3Model(repository_id, labels)
    elif model_type == 'LAMBERT_ROBERTA':
        model = LambertModel(repository_id, labels, 'ROBERTA')
    elif model_type == 'LAMBERT_BERTIMBAU':
        model = LambertModel(repository_id, labels, 'BERTIMBAU')
    elif model_type == 'LAYOUTLM':
        model = LayoutLMModel(repository_id, labels)
    elif model_type == 'LAYOUTLMV2':
        model = LayoutLMv2Model(repository_id, labels)
    else:
        raise Exception

    model.fit(
        dataset['train'], dataset['val'],
        max_epochs=args.epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size, patience=args.patience
    )
    model.save()

    # Results
    val_results, val_metrics, val_duration = model.predict(dataset['val'])
    test_results, test_metrics, test_duration = model.predict(dataset['test'])
    print(val_metrics)
    print(test_metrics)

    _val_results = process_results(val_results, max(id2label.keys()))
    _test_results = process_results(test_results, max(id2label.keys()))

    save_results(os.path.join(repository_id, 'val'), _val_results)
    save_metrics(os.path.join(repository_id, 'val'), val_metrics)
    save_results(os.path.join(repository_id, 'test'), _test_results)
    save_metrics(os.path.join(repository_id, 'test'), test_metrics)
