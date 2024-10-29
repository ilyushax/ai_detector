import json
from multiprocessing import cpu_count

import numpy as np
import torch
from peft import LoraConfig
from sklearn.metrics import roc_auc_score
from transformers import (BitsAndBytesConfig, Trainer, TrainingArguments,
                          get_scheduler)
from trl import SFTTrainer


class MyTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate
        )
        self.lr_scheduler = make_scheduler(optimizer, num_training_steps)
        self.optimizer = optimizer


def compute_roc_auc(pred):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    probabilities = sigmoid(pred.predictions)[:, 1]
    true_labels = pred.label_ids
    roc_auc = roc_auc_score(true_labels, probabilities)
    return {"roc_auc": roc_auc}


def make_scheduler(optimizer, num_training_steps):
    warmup_steps = int(0.2 * num_training_steps)  # 20% warmup
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


def get_trainer(model, tokenized_dataset, out_dir):
    with open("references/config.json", "r") as f:
        config = json.load(f)

    training_args_config = config["training_args"]
    training_args_config['outpy_dir'] = out_dir
    training_args = TrainingArguments(**training_args_config)
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_roc_auc,
    )
    return trainer


def get_lora_config(path):
    with open("references/configs/lora.json", "r") as f:
        config = json.load(f)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization_config"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization_config"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=(
            torch.float16
            if config["quantization_config"]["bnb_4bit_compute_dtype"] == "float16"
            else torch.float32
        ),
    )
    model_kwargs = dict(
        torch_dtype="auto",
        use_cache=False,
        device_map="cuda",
        quantization_config=quantization_config,
    )

    training_args_config = config["training_args"]
    training_args_config['outpy_dir'] = path
    training_args = TrainingArguments(**training_args_config)

    peft_config_data = config["peft_config"]
    peft_config = LoraConfig(**peft_config_data)

    return (model_kwargs, peft_config, training_args)


def get_lora_trainer(model, dataset, out):
    model_args, peft_config, train_args = get_lora_config(out)
    trainer = SFTTrainer(
        model=model,
        model_init_kwargs=model_args,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["train"],
        dataset_text_field=None,
        packing=True,
        peft_config=peft_config,
        max_seq_length=1024,
    )
    return trainer


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(dataset):
        return tokenizer(
            dataset["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
        )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=cpu_count() - 1
    )
    dataset = dataset.rename_column("generated", "labels")
    dataset.set_format("torch",
                       columns=["input_ids", "attention_mask", "labels"])

    return dataset
