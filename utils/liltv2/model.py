import datetime
import json
import os
from functools import partial
from utils.liltv2.pre_training import *
from utils.liltv2.pre_training_classes import *
from utils.liltv2.lora import *
from utils.liltv2.daem import *
from utils.liltv2.embeddings_patch import *
from utils.liltv2.embeddings_visual import *
from utils.dual_stream_arch import *
from utils.ags import *

import numpy as np
from datasets import (Array2D, ClassLabel, Features, Sequence, Value)
from transformers import (LiltForTokenClassification, LayoutLMv3TokenizerFast)

import datetime
import os

import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments
import torch

def process_lilt(dataset, labels, tokenizer,ags):
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
    def process(sample, tokenizer=None):
        encoding = tokenizer(
            sample["tokens"],
            boxes=ags.adaptive_gap_aware_sorting(sample["bboxes"]),
            word_labels=sample["ner_tags"],
            padding="max_length",
            truncation=True,
        )

        encoding['word_ids'] = encoding.word_ids()

        return encoding

    ## process the dataset and format it to pytorch
    dataset_processed = dataset.map(
        partial(process, tokenizer=tokenizer),
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

        self.ags = AGS()

        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_id)
        self.model = LiltForTokenClassification.from_pretrained(
            model_id, num_labels=len(labels), label2id=label2id, id2label=id2label
        )

        self.dual_stream = DualStreamAttentionLiLTv2(
            base_model_name="bert-base-uncased",
            num_tasks=3,
            task_heads=[{"type": "classification", "output_size": 10}]
        )

        self.daem = DAEM(
            d_model=self.model.config.hidden_size,
            n_heads=8, ff_dim=2048, n_layers=2, dropout=0.1
        )

        self.patch_embedding = PatchEmbedding(img_size=224, patch_size=16, embed_dim=self.model.config.hidden_size)
        self.visual_embedding = TokenVisualEmbedding(embed_dim=self.model.config.hidden_size)
        
        self.tasks = {
            #'MVLM': MVLMTask(labels=labels, embed_dim=self.model.config.hidden_size),
            'KPL': KPLTask(embed_dim=self.model.config.hidden_size),
            #'RPC': RPCTask(num_classes=8, embed_dim=self.model.config.hidden_size),
            'WPA': WPATask(embed_dim=self.model.config.hidden_size)
        }

        self.apply_lora()

    def apply_lora(self):
        """Substitui camadas lineares por versões com LoRA."""
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):  # Aplicar apenas em camadas lineares
                setattr(self.model, name, LoRALayer(module.in_features, module.out_features, rank=4))

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

        train_dataset_proc = process_lilt(train_dataset, ner_labels, self.tokenizer,self.ags)
        val_dataset_proc = process_lilt(val_dataset, ner_labels, self.tokenizer,self.ags)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.repository_id, 'checkpoints'),
            logging_dir=f"{self.repository_id}/logs",
            evaluation_strategy="epoch", metric_for_best_model="overall_f1",
            save_strategy="no" if patience is None else "epoch",
            save_total_limit=1 if patience is None else patience + 1,
            load_best_model_at_end=bool(patience is not None),
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate, num_train_epochs=max_epochs
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset_proc,
            eval_dataset=val_dataset_proc
        )

        trainer.train()

        # Realize o pré-treinamento para cada tarefa adicional
        for task_name, task in self.tasks.items():
            print(f"Treinando a tarefa {task_name}...")
            task.train(train_dataset_proc, val_dataset_proc, training_args)

        trainer.save_model()

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

        training_args = TrainingArguments(
            output_dir=os.path.join(self.repository_id, 'checkpoints'),
            logging_dir=f"{self.repository_id}/logs", log_level='passive',
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        )

        dataset_proc = process_lilt(dataset, ner_labels, self.tokenizer,self.ags)
        
        trainer = Trainer(
            model=self.model, args=training_args, compute_metrics=compute_metrics,
            eval_dataset=dataset_proc
        )

        start = datetime.datetime.now()
        predictions_logits, labels, metrics = trainer.predict(dataset_proc)
        prediction_duration = datetime.datetime.now() - start

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

        task_results = {}
        for task_name, task in self.tasks.items():
            print(f"Executando predição para a tarefa {task_name}...")
            task_results[task_name] = task.predict(dataset_proc)

        return results, metrics, prediction_duration, task_results

    def load(self, path):
        pretrained_dict = torch.load(
            path, map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.load_state_dict(pretrained_dict)

    def save(self):

        os.makedirs(self.repository_id, exist_ok=True)

        torch.save(
            self.model.state_dict(), os.path.join(self.repository_id, 'weights.pt')
        )

        with open(os.path.join(self.repository_id, 'info.json'), 'w') as f:
            f.write(json.dumps({
                'model_type': 'LILT',
                'labels': list(self.model.config.id2label.values())
            }))

    def forward(self, input_ids, attention_mask, labels=None, images=None, visual_features=None, text_layout=None, image_layout=None, task_id=0):
        """
        Forward pass combinando DAEM e Dual-Stream Attention.

        Args:
            input_ids (torch.Tensor): IDs dos tokens do texto.
            attention_mask (torch.Tensor): Máscara de atenção.
            labels (torch.Tensor, opcional): Rótulos para o treinamento.
            images (torch.Tensor, opcional): Representações das imagens/layouts.
            visual_features (torch.Tensor, opcional): Embeddings visuais dos tokens.
            text_layout (torch.Tensor, opcional): Layout dos textos.
            image_layout (torch.Tensor, opcional): Layout das imagens.
            task_id (int): Índice da tarefa a ser processada.

        Returns:
            Dict com logits e, opcionalmente, loss.
        """
        encoder_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_outputs.last_hidden_state  # Representação textual

        image_embeds = self.patch_embedding(images) if images is not None else None

        if visual_features is not None:
            visual_embeds = self.visual_embedding(visual_features)
            hidden_state += visual_embeds  # Incorporando informações visuais no texto

        if image_embeds is not None and text_layout is not None and image_layout is not None:
            hidden_state, _ = self.daem(hidden_state, image_embeds, text_layout, image_layout)

        fused_state = self.dual_stream(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layout_images=images,  # Passando imagens para processamento
            task_id=task_id
        )

        combined_state = hidden_state + fused_state  # Fusão dos embeddings

        logits = self.model.classifier(combined_state)

        output = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            output["loss"] = loss

        return output