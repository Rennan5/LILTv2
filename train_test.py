import argparse
import datetime
import os
import json

# import numpy as np

from utils.dataset import load_cord_dataset, load_custom_dataset, load_mixed_dataset
from utils.liltv2.model_test import LILTv2
from utils.save import save_metrics, save_results
from utils.ags import convert_ocr_format, sort_boxes

# Converts the .txt file to JSONL
def txt_to_json(arquivo_txt: str) -> list:
    try:
        with open(arquivo_txt, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [json.loads(line) for line in lines]
    except Exception as e:
        print(f".txt to JSON error: {e}")
        return []
    
def list_files(dir: str) -> list:
    try:
        arquivos = os.listdir(dir)
        return arquivos
    except Exception as e:
        print(f"Images list error: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument('--output-path', type=str, default='outputs/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=None)
    args = parser.parse_args()

    dataset_path = f'datasets/{args.dataset}'
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    train = txt_to_json(f'{dataset_path}/train.txt')
    val = txt_to_json(f'{dataset_path}/val.txt')
    test = txt_to_json(f'{dataset_path}/test.txt')
    
    model = LILTv2(num_tasks=2, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    # dataset = load_mixed_dataset(load_image=False, smoke_test=False, bbox_scale=1, train=train, dev=val, test=test)
    
    boxes = []
    for annotation in val[0]['annotations']:
        annotation['box'] = convert_ocr_format(annotation['box'])
        boxes.append(annotation['box'])

    boxes = sort_boxes(boxes)

    print(model)
