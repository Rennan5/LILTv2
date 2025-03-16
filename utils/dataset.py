import os
import pickle
import sys

from PIL import Image
from pdf2image import convert_from_path
from datasets import DatasetDict, Dataset
import json

def __get_dimensions__(bboxes):
    min_w, min_h = sys.maxsize, sys.maxsize
    max_w, max_h = 0, 0
    for bbox in bboxes:
        min_w = min(min_w, bbox[0], bbox[2])
        max_w = max(max_w, bbox[0], bbox[2])
        min_h = min(min_h, bbox[1], bbox[3])
        max_h = max(max_h, bbox[1], bbox[3])
    width, height = max_w - min_w, max_h - min_h

    return width, height, min_w, min_h


def reset_bboxes(original_bboxes, bbox_scale=1):
    all_bboxes = []
    for bboxes in original_bboxes:
        width, height, min_w, min_h = __get_dimensions__(bboxes)

        _bboxes = []
        for bbox in bboxes:
            _bboxes.append([
                bbox_scale * ((min(bbox[0], bbox[2]) - min_w) / width),
                bbox_scale * ((min(bbox[1], bbox[3]) - min_h) / height),
                bbox_scale * ((max(bbox[0], bbox[2]) - min_w) / width),
                bbox_scale * ((max(bbox[1], bbox[3]) - min_h) / height)
            ])
        all_bboxes.append(_bboxes)

    return all_bboxes


def __replace_elem__(elem, replacing_labels):
    try:
        return replacing_labels[elem]
    except KeyError:
        return elem


def replace_list(ls, replacing_labels):
    return [__replace_elem__(elem, replacing_labels) for elem in ls]


def get_labels(path: str, encoding='utf8'):
    """Get labels."""
    class_list = []

    with open(os.path.join(path, 'class_list.txt'), 'r',  encoding=encoding) as f:
        for line in f.readlines():
            class_list.append(line[:-1].split(' ')[-1])

    return class_list

def load_custom_dataset(path, partitions=('train', 'val', 'test'), smoke_test=False, bbox_scale=1):
    dataset_dict = {}

    for partition in partitions:
        partition_dict = {
            'id': [],
            'filenames': [],
            'tokens': [],
            'bboxes': [],
            'ner_tags': [],
            'img_size': [],
            'image_path': []
        }

        with open(os.path.join(path, f'{partition}.txt')) as f:
            partition_docs = [json.loads(line) for line in f.readlines()]

        if smoke_test:
            partition_docs = partition_docs[:3]

        for doc_i, doc in enumerate(partition_docs):
            doc_texts, doc_bboxes, doc_ner_tags = [], [], []
            w, h = doc['width'], doc['height']

            for word in doc['annotations']:
                doc_texts.append(word['text'])

                x0 = bbox_scale * min(word['box'][::2]) / w
                y0 = bbox_scale * min(word['box'][1::2]) / h
                x1 = bbox_scale * min(word['box'][::2]) / w
                y1 = bbox_scale * min(word['box'][1::2]) / h
                doc_bboxes.append([
                    min(max(x0, 0), bbox_scale),
                    min(max(y0, 0), bbox_scale),
                    min(max(x1, 0), bbox_scale),
                    min(max(y1, 0), bbox_scale)
                ])
                doc_ner_tags.append(word['label'])

            partition_dict['id'].append(doc_i)
            partition_dict['filenames'].append(doc['file_name'])
            partition_dict['tokens'].append(doc_texts)
            partition_dict['bboxes'].append(doc_bboxes)
            partition_dict['ner_tags'].append(doc_ner_tags)
            partition_dict['img_size'].append([w, h])
            
            image_path = os.path.abspath(os.path.join(path, "images", doc["file_name"]))
            partition_dict["image_path"].append(image_path)  

        dataset_dict[partition] = Dataset.from_dict(partition_dict)

    return DatasetDict(dataset_dict), get_labels(path)

def load_cord_dataset(path='dataset/cord/', partitions=('train', 'val', 'test'), load_image=False, smoke_test=False, bbox_scale=1):
    dataset_dict = {}

    labels = get_labels(path)
    label2id = {label: i for (i, label) in enumerate(labels)}

    # labels that contains few examples
    change_labels = {
        'menu.etc': 'O', 'menu.itemsubtotal': 'O', 'menu.sub_etc': 'O',
        'menu.sub_unitprice': 'O', 'menu.vatyn': 'O', 'void_menu.nm': 'O',
        'void_menu.price': 'O', 'sub_total.othersvc_price': 'O'
    }

    for partition in partitions:
        partition_dict = {
            'id': [],
            'filenames': [],
            'tokens': [],
            'bboxes': [],
            'ner_tags': [],
        }

        if load_image:
            partition_dict['image'] = []

        data = pickle.load(
            open(os.path.join(path, f'{ partition }.pkl'), 'rb'))
        data[1] = [replace_list(ls, change_labels) for ls in data[1]]

        for i in range(len(data[1])):
            for ii in range(len(data[1][i])):
                label = data[1][i][ii]
                if label not in ['O', 'Others', 'Ignore']:
                    label = 'I-' + label
                data[1][i][ii] = label2id[label]

        num_docs = len(data[0]) if not smoke_test else 3
        for doc_i in range(num_docs):
            doc_texts = data[0][doc_i]
            doc_ner_tags = data[1][doc_i]
            doc_bboxes = reset_bboxes(data[2], bbox_scale=bbox_scale)[doc_i]
            image_path = data[3][doc_i]

            # check if bboxes are normalized
            for bbox in doc_bboxes:
                assert min(bbox) >= 0
                assert max(bbox) <= bbox_scale

            partition_dict['id'].append(doc_i)
            partition_dict['filenames'].append(image_path.split(os.sep)[-1])
            partition_dict['tokens'].append(doc_texts)
            partition_dict['bboxes'].append(doc_bboxes)
            partition_dict['ner_tags'].append(doc_ner_tags)

            if load_image:
                partition_dict['image'].append(
                    Image.open(os.path.join('dataset', image_path)).convert("RGB")
                    # if not image_path.lower().endswith(('.pdf')) else
                    # convert_from_path(os.path.join('dataset', image_path))[0].convert("RGB")
                )

        dataset_dict[partition] = Dataset.from_dict(partition_dict)

    return DatasetDict(dataset_dict), labels


# TODO: deal with image
def load_mixed_dataset(load_image=False, smoke_test=False, bbox_scale=1):
    def replace(data, paths, cols=('id', 'filenames', 'tokens', 'bboxes', 'ner_tags')):
        _data = {col: [] for col in cols}
        for i, fn in enumerate(data['filenames']):
            if fn not in paths:
                continue

            for col in cols:
                _data[col].append(data[col][i])

        return _data

    def join(datasets, cols=('id', 'filenames', 'tokens', 'bboxes', 'ner_tags')):
        _dataset = {col: [] for col in cols}

        for dataset in datasets:
            for col in cols:
                _dataset[col] += dataset[col]

        return _dataset

    cols = ['id', 'filenames', 'tokens', 'bboxes', 'ner_tags']
    if load_image:
        cols += ['image']

    dataset_original, labels = load_cord_dataset(
        load_image=load_image, smoke_test=smoke_test, bbox_scale=bbox_scale
    )
    dataset_easy, _ = load_custom_dataset(
        os.path.join('dataset', 'cord_easy_ocr'), load_image=load_image, smoke_test=smoke_test, bbox_scale=bbox_scale
    )
    dataset_paddle, _ = load_custom_dataset(
        os.path.join('dataset', 'cord_paddle_ocr'), load_image=load_image, smoke_test=smoke_test, bbox_scale=bbox_scale
    )
    dataset_azure, _ = load_custom_dataset(
        os.path.join('dataset', 'cord_azure_ocr'), load_image=load_image, smoke_test=smoke_test, bbox_scale=bbox_scale
    )

    dataset_original = {
        'train': replace(
            dataset_original['train'], [f'receipt_00{str(i).zfill(3)}.png' for i in range(0, 200)], cols=cols
        ),
        'val': replace(
            dataset_original['val'], [f'receipt_00{str(i).zfill(3)}.png' for i in range(0, 25)], cols=cols
        ),
        'test': replace(
            dataset_original['test'], [f'receipt_00{str(i).zfill(3)}.png' for i in range(0, 25)], cols=cols
        )
    }
    dataset_paddle = {
        'train': replace(
            dataset_paddle['train'], [f'train_receipt_00{str(i).zfill(3)}.png' for i in range(200, 400)], cols=cols
        ),
        'val': replace(
            dataset_paddle['val'], [f'val_receipt_00{str(i).zfill(3)}.png' for i in range(25, 50)], cols=cols
        ),
        'test': replace(
            dataset_paddle['test'], [f'test_receipt_00{str(i).zfill(3)}.png' for i in range(25, 50)], cols=cols
        )
    }
    dataset_easy = {
        'train': replace(
            dataset_easy['train'], [f'train_receipt_00{str(i).zfill(3)}.png' for i in range(400, 600)], cols=cols
        ),
        'val': replace(
            dataset_easy['val'], [f'val_receipt_00{str(i).zfill(3)}.png' for i in range(50, 75)], cols=cols
        ),
        'test': replace(
            dataset_easy['test'], [f'test_receipt_00{str(i).zfill(3)}.png' for i in range(50, 75)], cols=cols
        )
    }
    dataset_azure = {
        'train': replace(
            dataset_azure['train'], [f'train_receipt_00{str(i).zfill(3)}.png' for i in range(600, 800)], cols=cols
        ),
        'val': replace(
            dataset_azure['val'], [f'val_receipt_00{str(i).zfill(3)}.png' for i in range(75, 100)], cols=cols
        ),
        'test': replace(
            dataset_azure['test'], [f'test_receipt_00{str(i).zfill(3)}.png' for i in range(75, 100)], cols=cols
        )
    }

    dataset = DatasetDict({
        'train': Dataset.from_dict(
            join(
                [dataset_original['train'], dataset_paddle['train'], dataset_easy['train'], dataset_azure['train']],
                cols=cols
            )
        ),
        'test': Dataset.from_dict(
            join(
                [dataset_original['test'], dataset_paddle['test'], dataset_easy['test'], dataset_azure['test']],
                cols=cols
            )
        ),
        'val': Dataset.from_dict(
            join(
                [dataset_original['val'], dataset_paddle['val'], dataset_easy['val'], dataset_azure['val']],
                cols=cols
            )
        )
    })
    return dataset, labels
