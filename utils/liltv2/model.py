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
from datasets import (Array2D, ClassLabel, Features, Sequence, Value,Array4D) 
from transformers import (LiltForTokenClassification, LayoutLMv3TokenizerFast)
from transformers import AutoTokenizer,default_data_collator
from torchvision import transforms
from PIL import Image
import io

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

    def forward(self, input_ids, attention_mask, labels=None, images=None, visual_features=None, img_size=None, task_id=0):
        """
        Forward pass combinando DAEM e Dual-Stream Attention para inferência.
        """
        image_size = max(img_size[0][0][0], img_size[0][0][1]) if img_size is not None else 800
        image_layout = torch.zeros((1, image_size, image_size, 3), device=input_ids.device)
        text_layout = torch.zeros((input_ids.shape[0], 512, 4), device=input_ids.device)
        encoder_outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_outputs.hidden_states[-1]  
        image_embeds = self.patch_embedding(images) if images is not None else None
        
        if visual_features is not None:
            visual_embeds = self.visual_embedding(visual_features)
            hidden_state += visual_embeds  

        # Aplicação do DAEM para fusão dos embeddings textuais e visuais
        if image_embeds is not None:
            hidden_state, _ = self.daem(hidden_state, image_embeds, text_layout, image_layout)

        # Camada final de classificação
        logits = self.classifier(hidden_state)

        output = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
            output["loss"] = loss

        return output
    
def process_lilt(dataset, labels, tokenizer, ags, model):
    features = Features({
        "filenames": Value(dtype="string"),
        "tokens": Sequence(feature=Value(dtype="string")),
        "word_ids": Sequence(feature=Value(dtype="int64")),
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "attention_mask": Sequence(feature=Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(512, 4)),
        "img_size": Array2D(dtype="int64", shape=(1, 2)),
        "image_path": Value(dtype="string"), 
        "labels": Sequence(ClassLabel(names=labels)),
        "patch_embeddings": Array4D(dtype="float32", shape=(1,1, 49, 768)) # <-- Adicionando embeddings dos patches
    })

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def process(sample, tokenizer=None, model=None):
        encoding = tokenizer(
            sample["tokens"],
            boxes=ags.adaptive_gap_aware_sorting(sample["bboxes"]),
            word_labels=sample["ner_tags"],
            padding="max_length",
            truncation=True,
        )

        encoding['word_ids'] = encoding.word_ids()
        encoding['img_size'] = [sample["img_size"]]
        encoding['image_path'] = sample["image_path"]

        # Carregar imagem e extrair patches
        image = Image.open(sample["image_path"]).convert("RGB")
        image = transform(image).unsqueeze(0)  # Adiciona batch dimension
        with torch.no_grad():
            patch_embeddings = model.patch_embedding(image).squeeze(0).cpu().numpy()
        patch_embeddings = patch_embeddings.reshape(1, 196, 768)
        patch_embeddings = patch_embeddings[:, :49, :]  # Mantém apenas os primeiros 49 patches
        patch_embeddings = patch_embeddings.reshape(1, 1, 49, 768)  # Ajusta para a forma esperada

        encoding["patch_embeddings"] = patch_embeddings
        return encoding

    dataset_processed = dataset.map(
        partial(process, tokenizer=tokenizer, model=model),
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        config = LiltConfig.from_pretrained(
            model_id, 
            num_labels=len(labels), 
            label2id=label2id, 
            id2label=id2label,
            output_hidden_states=True  # Permite acessar as camadas ocultas
        )

        self.dual_stream = DualStreamAttentionLiLTv2(
            base_model_name="bert-base-uncased",
            tokenizer=self.tokenizer,  
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
        }

        self.model.tasks = self.tasks
        
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
                if module.in_features in [768, 3072]:
                    modules_to_replace.append((name, module))

        for name, module in modules_to_replace:
            parent_module, child_name = self.get_parent_module(name)
            setattr(parent_module, child_name, LoRALayer(module.in_features, module.out_features, rank=4))

    def fit(self, train_dataset, val_dataset, max_epochs=1, learning_rate=5e-5, batch_size=1, patience=None,training_args=None):
        
        ner_labels = list(self.model.config.id2label.values())
        metric = evaluate.load("seqeval")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

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
        
        val_dataset_proc = process_lilt(val_dataset, ner_labels, self.tokenizer,self.ags,self.model)
        train_dataset_proc = process_lilt(train_dataset, ner_labels, self.tokenizer,self.ags,self.model)
        
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)
        
        # Criando DataLoader para MVLM
        train_dataset_mvlm = train_dataset_proc.remove_columns(["filenames", "tokens", "word_ids", "bbox", "img_size", "image_path"])
        train_dataset_mvlm.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        train_dataloader_mvlm = DataLoader(
            train_dataset_mvlm, batch_size=8, shuffle=True, collate_fn=data_collator
        )
        
        # Criando DataLoader para WPA (pegando apenas embeddings de patches)
        train_dataset_wpa = train_dataset_proc.remove_columns(["filenames", "tokens", "word_ids", "bbox", "img_size", "image_path", "labels"])
        train_dataset_wpa.set_format(type="torch", columns=["patch_embeddings"])

        train_dataloader_wpa = DataLoader(
            train_dataset_wpa, batch_size=8, shuffle=True
        )

        train_dataset_kpl = train_dataset_proc.remove_columns(["filenames", "tokens", "image_path"]) 

        data_collator_kpl = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, padding=True
        )

        train_dataloader_kpl = DataLoader(
            train_dataset_kpl, batch_size=8, shuffle=True, collate_fn=data_collator_kpl
        )
        
        if "MVLM" in self.tasks:
            print("Treinando MVLM...")
            self.tasks["MVLM"].train_task(train_dataloader_mvlm, 10, "mvlm_checkpoint")
            self.model.save_pretrained("mvlm_checkpoint/final")  # Salva o modelo após MVLM

        if "WPA" in self.tasks:
            print("Treinando WPA...")
            self.tasks["WPA"].train_task(train_dataloader_wpa, 100, "wpa_checkpoint")
            self.model.save_pretrained("wpa_checkpoint/final")  # Salva o modelo após WPA

        if "KPL" in self.tasks:
            print("Treinando KPL...")
            self.tasks["KPL"].train_task(train_dataloader_kpl, 10, "kpl_checkpoint")
            self.model.save_pretrained("kpl_checkpoint/final")
        
        i = 1
        def load_images(dataset,i):
            images = []
            for image_path in dataset["image_path"]:
                if i%100 == 0: print(i)
                i += 1
                img = Image.open(image_path).convert("RGB")
                img = transform(img)  
                images.append(img) 
            return images

        train_images = load_images(train_dataset_proc,i)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset_proc,
            eval_dataset=val_dataset_proc,
            compute_metrics=compute_metrics,
            data_collator=lambda data: {
                "input_ids": torch.stack([torch.as_tensor(f["input_ids"]) for f in data]),
                "attention_mask": torch.stack([torch.as_tensor(f["attention_mask"]) for f in data]),
                "labels": torch.stack([torch.as_tensor(f["labels"]) for f in data]),
                "images": torch.stack([train_images[i] for i in range(len(data))]),
                "img_size": torch.stack([torch.as_tensor(f["img_size"]) for f in data])
            }
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