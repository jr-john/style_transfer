import os
import ast
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader
from datasets import Dataset, DatasetDict, Value, ClassLabel, Features, load_metric
from transformers.integrations import MLflowCallback
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

print(torch.cuda.is_available())


def construct_classification_dataset(path: str):
    SPLITS = ["train", "test", "valid"]
    LABELS = ["src", "tgt"]
    dataset = defaultdict(dict)
    for split in SPLITS:
        for label in LABELS:
            with open(path + "/" + split + "." + label) as f:
                if split in ["test", "valid"] and label == "tgt":
                    dataset[split][label] = [
                        ast.literal_eval(line.strip()) for line in f
                    ]
                else:
                    dataset[split][label] = [line.strip() for line in f]

    dataset_dict = defaultdict(dict)
    FEATURES = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(num_classes=2, names=["informal", "formal"]),
        }
    )

    for split in SPLITS:
        split_dict = defaultdict(list)
        for label in LABELS:
            src_text = list()
            tgt_text = list()

            src_text.extend(dataset[split]["src"])
            if split in ["test", "valid"]:
                tgt_text.extend(
                    [item for sublist in dataset[split]["tgt"] for item in sublist]
                )
            else:
                tgt_text.extend(dataset[split]["tgt"])

        # reorder records so pairs alternate in sequence
        split_dict["text"].extend(
            [elem for pair in zip(src_text, tgt_text) for elem in pair]
        )
        split_dict["label"].extend(
            ["informal", "formal"] * min(len(src_text), len(tgt_text))
        )
        if len(src_text) > len(tgt_text):
            split_dict["text"].extend(src_text[len(tgt_text) :])
            split_dict["label"].extend(["informal"] * (len(src_text) - len(tgt_text)))
        if len(src_text) < len(tgt_text):
            split_dict["text"].extend(tgt_text[len(src_text) :])
            split_dict["label"].extend(["formal"] * (len(tgt_text) - len(src_text)))

        dataset_dict[split] = Dataset.from_dict(split_dict, features=FEATURES)

    return DatasetDict(dataset_dict)


class CustomTrainer(Trainer):
    """
    A custom Trainer that overwrites and subclasses the `get_train_dataloader()` method.
    This customization allows us to introduce a flag that disables shuffling on the DataLoader. When
    `shuffle_train` flag is True, a RandomSampler is used via `self._get_train_sampler`. When set to False,
    a SequentialSampler is utilized in the dataloader.
    """

    def __init__(self, shuffle_train, *args, **kwargs):
        self.shuffle_train = shuffle_train
        super().__init__(*args, **kwargs)

    def seed_worker(self, _):
        """
        Helper function to set worker seed during Dataloader initialization.
        """
        worker_seed = torch.initial_seed() % 2 ** 32
        set_seed(worker_seed)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
            print(train_dataset)
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        if self.shuffle_train:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = SequentialSampler(self.train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=self.seed_worker,
        )


sti_args = {
    "output_dir": "./models/bert_em",
    "learning_rate": 3e-05,
    "weight_decay": 0.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 8.0,
    "logging_dir": "./models/logs/bert_em",
    "logging_strategy": "steps",
    "logging_steps": 10000,
    "eval_steps": 20000,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "save_steps": 20000,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_accuracy",
    "greater_is_better": True,
}

misc_args = {
    "model_name_or_path": "bert-base-uncased",
    "dataset_name": "gyafc_em",
    "shuffle_train": True,
}

set_seed(42)

# establish training arguments
training_args = TrainingArguments(**sti_args)

if not os.path.exists(training_args.output_dir):
    os.makedirs(training_args.output_dir)

# load classification dataset
data = construct_classification_dataset(misc_args["dataset_name"])

# load base-model and tokenizer
checkpoint = misc_args["model_name_or_path"]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


tokenized_datasets = data.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def compute_metrics(eval_preds):

    accuracy_metric = load_metric("accuracy")
    # f1_metric = load_metric("f1")

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # return {
    #     "accuracy": accuracy_metric.compute(predictions=predictions, references=labels),
    #     "f1": f1_metric.compute(predictions=predictions, references=labels),
    # }

    return accuracy_metric.compute(predictions=predictions, references=labels)


trainer = CustomTrainer(
    shuffle_train=misc_args["shuffle_train"],
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.remove_callback(MLflowCallback)

trainer.train()
