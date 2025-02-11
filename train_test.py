import argparse
import datetime
import os
import json
import torch

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

    # dataset = load_mixed_dataset(load_image=False, smoke_test=False, bbox_scale=1, train=train, dev=val, test=test)
    
    # Convert the boxes to the format expected by the model and getting input texts
    #train_texts = []
    #train_boxes = []
    #for itera in train:
    #    for annotation in itera['annotations']:
    #        train_texts.append(annotation['text'])
    #        annotation['box'] = convert_ocr_format(annotation['box'])
    #        train_boxes.append(annotation['box'])
    
    #val_texts = []
    #val_boxes = []
    #for itera in val:
    #    for annotation in itera['annotations']:
    #        val_texts.append(annotation['text'])
    #        annotation['box'] = convert_ocr_format(annotation['box'])
    #        val_boxes.append(annotation['box'])

    test_texts = []
    test_boxes = []
    for itera in test:
        for annotation in itera['annotations']:
            test_texts.append(annotation['text'])
            annotation['box'] = convert_ocr_format(annotation['box'])
            test_boxes.append(annotation['box'])

    # AGS algorithm to sort boxes
    #train_boxes = sort_boxes(train_boxes)
    #val_boxes = sort_boxes(val_boxes)
    test_boxes = sort_boxes(test_boxes)

    # Model instatiation
    model = LILTv2(num_tasks=2, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # Tokenization of inputs
    input_tokens = model.tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
    labels = torch.tensor([1]).unsqueeze(0)
    
    #for input_id, box in enumerate(test_boxes):
    #    for task_id, task in enumerate(model.task_heads):
    #        if task['type'] == 'classification':
    #            model.train_step(input_ids=test_boxes, attention_mask=box, labels=task_id, task_id=task_id)
    model.train_step(input_ids=input_tokens['input_ids'], attention_mask=input_tokens['attention_mask'], labels=labels, task_id=0)
