import datetime
import json
import os
from functools import partial

import numpy as np
from datasets import (Array2D, ClassLabel, Features, Sequence, Value)
from transformers import (LiltForTokenClassification, LayoutLMv3TokenizerFast)

import datetime
import os

import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments
import torch

def process(sample, tokenizer=None):
    encoding = tokenizer(
        sample["tokens"],
        boxes=sample["bboxes"],
        word_labels=sample["ner_tags"],
        padding="max_length",
        truncation=True,
    )

    encoding['word_ids'] = encoding.word_ids()

    return encoding

def process_lilt(dataset, labels, tokenizer):
    ## we need to define custom features
    features = Features({
        # useful(not required) columns
        "filenames": Value(dtype="string"),
        "tokens": Sequence(feature=Value(dtype="string")),
        "word_ids": Sequence(feature=Value(dtype="int64")),

        # required columns
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "attention_mask": Sequence(feature=Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(512, 4)),
        "labels": Sequence(ClassLabel(names=labels)),
    })

    ## process the dataset and format it to pytorch
    dataset_processed = dataset.map(
        partial(process, tokenizer=tokenizer),
        # remove_columns=["image", "tokens", "ner_tags", "id", "bboxes"],
        remove_columns=["ner_tags", "id", "bboxes"],
        features=features,
    ).with_format("torch")

    return dataset_processed


class LiltModel:
    def __init__(self, repository_id, labels):
        self.repository_id = repository_id
        model_id = "SCUT-DLVCLab/lilt-roberta-en-base"

        id2label = {v: k for v, k in enumerate(labels)}
        label2id = {k: v for v, k in enumerate(labels)}

        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_id)
        self.model = LiltForTokenClassification.from_pretrained(
            model_id, num_labels=len(labels), label2id=label2id, id2label=id2label
        )

    def fit(self, train_dataset, val_dataset, max_epochs=1, learning_rate=5e-5, batch_size=1, patience=None):
        ner_labels = list(self.model.config.id2label.values())
        metric = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            all_predictions = []
            all_labels = []
            for prediction, label in zip(predictions, labels):
                for predicted_idx, label_idx in zip(prediction, label):
                    if label_idx == -100:
                        continue
                    all_predictions.append(ner_labels[predicted_idx])
                    all_labels.append(ner_labels[label_idx])
            return metric.compute(predictions=[all_predictions], references=[all_labels])

        # Define training args
        training_args = TrainingArguments(
            # logging
            output_dir=os.path.join(self.repository_id, 'checkpoints'),
            logging_dir=f"{self.repository_id}/logs", logging_strategy="epoch",

            # evaluation
            evaluation_strategy="epoch", metric_for_best_model="overall_f1",
            save_strategy="no" if patience is None else "epoch",
            save_total_limit=1 if patience is None else patience + 1,
            load_best_model_at_end=bool(patience is not None),

            # hyper parameters
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate, num_train_epochs=max_epochs
        )

        # Create Trainer instance
        train_dataset_proc = process_lilt(train_dataset, ner_labels, self.tokenizer)
        val_dataset_proc   = process_lilt(val_dataset, ner_labels, self.tokenizer)

        trainer = Trainer(
            model=self.model, args=training_args, compute_metrics=compute_metrics,
            train_dataset=train_dataset_proc, eval_dataset=val_dataset_proc
        )

        # Start training
        trainer.train()

    def predict(self, dataset, batch_size=1):
        ner_labels = list(self.model.config.id2label.values())
        metric = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            all_predictions = []
            all_labels = []
            for prediction, label in zip(predictions, labels):
                for predicted_idx, label_idx in zip(prediction, label):
                    if label_idx == -100:
                        continue
                    all_predictions.append(ner_labels[predicted_idx])
                    all_labels.append(ner_labels[label_idx])
            return metric.compute(predictions=[all_predictions], references=[all_labels])

        # Define training args
        training_args = TrainingArguments(
            # logging
            output_dir=os.path.join(self.repository_id, 'checkpoints'),
            logging_dir=f"{self.repository_id}/logs", log_level='passive',

            # hyper parameters
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        )

        # Create Trainer instance
        dataset_proc = process_lilt(dataset, ner_labels, self.tokenizer)
        trainer = Trainer(
            model=self.model, args=training_args, compute_metrics=compute_metrics
        )

        # Predict
        start = datetime.datetime.now()
        predictions_logits, labels, metrics = trainer.predict(dataset_proc)
        prediction_duration = datetime.datetime.now() - start

        # TODO: change from filenames to ids
        results = {}
        for i in range(len(dataset_proc)):
            results[dataset_proc['filenames'][i]] = {
                'input_ids': dataset_proc['input_ids'].tolist()[i],
                'texts': dataset_proc['tokens'][i],
                'word_ids': dataset_proc['word_ids'][i],
                'bbox': dataset_proc['bbox'].tolist()[i],
                'predictions_logits': predictions_logits.tolist()[i],
                'labels': labels.tolist()[i]
            }

        return results, metrics, prediction_duration

    def load(self, path):
        pretrained_dict = torch.load(
            path, map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.load_state_dict(pretrained_dict)

    def save(self):
        torch.save(
            self.model.state_dict(), os.path.join(self.repository_id, 'weights.pt')
        )

        with open(os.path.join(self.repository_id, 'info.json'), 'w') as f:
            f.write(json.dumps({
                'model_type': 'LILT',
                'labels': list(self.model.config.id2label.values())
            }))
