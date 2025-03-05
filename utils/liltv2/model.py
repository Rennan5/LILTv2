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
from transformers import Trainer, TrainingArguments, LiltConfig
import torch

class CustomLiltModel(LiltForTokenClassification, nn.Module):
    def __init__(self, config, tokenizer, daem, dual_stream, patch_embedding, visual_embedding, tasks):
        super().__init__(config)
        self.tokenizer = tokenizer  
        self.daem = daem
        self.dual_stream = dual_stream
        self.patch_embedding = patch_embedding
        self.visual_embedding = visual_embedding
        self.tasks = tasks

    def forward(self, input_ids, attention_mask, labels=None, images=None, visual_features=None, text_layout=None, image_layout=None, task_id=0):
        """
        Forward pass combinando DAEM e Dual-Stream Attention.
        """
        
        encoder_outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_outputs.hidden_states[-1]  # Última camada oculta

        image_embeds = self.patch_embedding(images) if images is not None else None

        if visual_features is not None:
            visual_embeds = self.visual_embedding(visual_features)
            hidden_state += visual_embeds  # Incorporando informações visuais no texto

        if image_embeds is not None and text_layout is not None and image_layout is not None:
            hidden_state, _ = self.daem(hidden_state, image_embeds, text_layout, image_layout)

        '''
        TODO

        masked_indices = (input_ids == self.tokenizer.mask_token_id)
        mvlm_preds = self.tasks["MVLM"](text_tokens=input_ids, layout_boxes=text_layout, image_features=image_embeds, masked_indices=masked_indices)
        
        if mvlm_preds.numel() > 0:
            mvlm_preds_mean = mvlm_preds.mean(dim=-1, keepdim=True)
            hidden_state += mvlm_preds_mean.expand_as(hidden_state).to(hidden_state.device)

        keypoint_preds = self.tasks["KPL"](hidden_state)
        hidden_state += keypoint_preds.mean(dim=-1, keepdim=True)  

        rpc_preds = self.tasks["RPC"](text_layout, image_layout)
        hidden_state += rpc_preds.mean(dim=-1, keepdim=True)   

        alignment_scores = self.tasks["WPA"](hidden_state, image_embeds)
        hidden_state += alignment_scores.mean(dim=-1, keepdim=True)
        '''

        # Dual-Stream Attention
        '''
        TODO

        fused_state = self.dual_stream(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layout_images=images,  
            task_id=task_id
        )
        '''
        
        #combined_state = hidden_state + fused_state # Fusão dos embeddings
        combined_state = hidden_state
        
        logits = self.classifier(combined_state)

        output = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
            output["loss"] = loss

        return output

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

        config = LiltConfig.from_pretrained(
            model_id, 
            num_labels=len(labels), 
            label2id=label2id, 
            id2label=id2label,
            output_hidden_states=True  # Permite acessar as camadas ocultas
        )

        self.dual_stream = DualStreamAttentionLiLTv2(
            base_model_name="bert-base-uncased",
            num_tasks=3,
            task_heads=[{"type": "classification", "output_size": 10}]
        )

        self.daem = DAEM(
            d_model=config.hidden_size,
            n_heads=8, ff_dim=2048, n_layers=2, dropout=0.1
        )

        self.patch_embedding = PatchEmbedding(img_size=224, patch_size=16, embed_dim=config.hidden_size)
        self.visual_embedding = TokenVisualEmbedding(embed_dim=config.hidden_size)

        self.model = CustomLiltModel(
            config=config,
            tokenizer=self.tokenizer,  
            daem=self.daem,
            dual_stream=self.dual_stream,
            patch_embedding=self.patch_embedding,
            visual_embedding=self.visual_embedding,
            tasks=None
        )

        self.tasks = {
            'MVLM': MVLMTask(model=self.model, tokenizer=self.tokenizer, vocab_size=len(self.tokenizer)),
            'WPA': WPATask(model=self.model, patch_size=16, img_size=224),
            'KPL': KPLTask(model=self.model, grid_size=7, num_classes=49),
            'RPC': RPCTask(model=self.model, input_dim=config.hidden_size, hidden_dim=256, num_directions=8, num_distances=5)
        }

        self.model.tasks = self.tasks
        
        '''
        TODO

        Adicionar treinamento das tasks (chamar train_task para cada task de self.tasks)
        '''

        self.apply_lora()

    def get_parent_module(self, module_name):
        """
        Retorna o módulo pai e o nome do submódulo a ser substituído.

        Args:
            module_name (str): Nome completo do módulo a ser substituído (ex: "dual_stream.text_encoder.encoder.layer.11.attention.self.query").

        Returns:
            parent_module (torch.nn.Module): Módulo pai.
            child_name (str): Nome do submódulo dentro do módulo pai.
        """
        names = module_name.split('.')
        parent = self.model  # Começa no modelo principal

        for name in names[:-1]:  # Navega pelos submódulos até o penúltimo nível
            parent = getattr(parent, name)

        return parent, names[-1]  # Retorna o módulo pai e o nome do submódulo

    def apply_lora(self):
        modules_to_replace = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and "dual_stream.text_encoder" in name:
                modules_to_replace.append((name, module))
                
        for name, module in modules_to_replace:
            parent_module, child_name = self.get_parent_module(name)
            setattr(parent_module, child_name, LoRALinear(module))

    def fit(self, train_dataset, val_dataset, max_epochs=1, learning_rate=5e-5, batch_size=1, patience=None,training_args=None):

        if training_args is None:
            training_args = TrainingArguments(
                output_dir=os.path.join(self.repository_id, 'checkpoints'),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                num_train_epochs=max_epochs,
                weight_decay=0.01,
                warmup_ratio=0.1,
                lr_scheduler_type="linear",
            )

        ner_labels = list(self.model.config.id2label.values())
        metric = evaluate.load("seqeval")

        """
        for task_name, task in self.tasks.items():
            print(f"Treinando {task_name}...")
            task.train_task(train_dataset, val_dataset, training_args)

            task_weights_path = f"{training_args.output_dir}/{task_name.lower()}_task.pth"
            print(f"Carregando pesos de {task_name}...")
            task.load_state_dict(torch.load(task_weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        """
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

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset_proc,
            eval_dataset=val_dataset_proc
        )

        trainer.train()

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

        return results, metrics, prediction_duration

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