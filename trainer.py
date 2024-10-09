import argparse
import os
import torch
import wandb

import numpy as np

from torchvision import transforms as T

from accelerate import Accelerator
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import AdamW, get_scheduler
from transformers import TrainingArguments, Trainer, DefaultDataCollator

from modules.config import Config
from modules.dataset import load_dataset
from modules.scheduler import CosineAnnealingWithWarmupAndEtaMin

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_class_weights(dataset):
    labels = dataset['label']
    class_counts = np.zeros(config.num_classes)
    for label in labels:
        for l in label:
            class_counts[l] += 1

    # Avoid division by zero
    class_counts[class_counts == 0] = 1

    # Calculate class weights inversely proportional to class frequencies
    class_weights = np.sum(class_counts) / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Normalize weights to have a mean of 1
    class_weights = class_weights / class_weights.mean()

    # Clamping weights to avoid extreme values
    class_weights = torch.clamp(class_weights, min=0.05, max=5)

    return class_weights


def transforms(examples):
    global _transforms
    examples["pixel_values"] = [_transforms(image.convert('RGB')) for image in examples["image"]]

    del examples["image"]
    del examples["id"]
    del examples["idx"]

    # Converting labels to binary vectors
    labels = []
    for label in examples["label"]:
        binary_vector = torch.zeros(config.num_classes)
        for l in label:
            binary_vector[l] = 1
        labels.append(binary_vector)
    examples["labels"] = labels
    del examples["label"]

    return examples

def val_transforms(examples):
    global _val_transforms
    examples["pixel_values"] = [_val_transforms(image.convert('RGB')) for image in examples["image"]]
    del examples["image"]
    del examples["id"]
    del examples["idx"]

    labels = []
    for label in examples["label"]:
        binary_vector = torch.zeros(config.num_classes)
        for l in label:
            binary_vector[l] = 1
        labels.append(binary_vector)
    examples["labels"] = labels
    del examples["label"]

    return examples

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        global config
        # If DeepSpeed is enabled, no need to manually create optimizer or scheduler
        if self.args.deepspeed:
            return

        # Initialize the optimizer manually
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)

        # Initialize the scheduler manually
        if config.scheduler == "cosine" and config.eta_min != 0.0:
            self.lr_scheduler = CosineAnnealingWithWarmupAndEtaMin(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=config.eta_min,
                warmup_steps=self.args.warmup_steps
            )
        else:
            self.lr_scheduler = get_scheduler(
                config.scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        global loss_fn
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# if we are on windows, we need to check it, and set the torch backend to gloo
if os.name == 'nt':
    try:    
        torch.distributed.init_process_group(backend="gloo")
    except:
        pass

accelerator = Accelerator()

parser = argparse.ArgumentParser(description='Train a model for tagging images')
parser.add_argument('-c','--config', help='Path to the training config file', required=True)
parser.add_argument('-w','--wandb', help='Use wandb for logging', action='store_true')
parser.add_argument('-r','--resume', help='Resume training from the given checkpoint path', default=None)

args = parser.parse_args()

config = Config(args.config)

device = accelerator.device if torch.cuda.is_available() else 'cpu'

image_processor = AutoImageProcessor.from_pretrained(config.model, use_fast=True)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

# Defining transforms

_transforms = T.Compose([
    T.Resize(size),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.75, 1.1),
                   shear=None,
                   fill=tuple(np.array(np.array(image_processor.image_mean) * 255).astype(int).tolist())),
    T.ToTensor(),
    T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)])

_val_transforms = T.Compose([
    T.Resize(size), 
    T.ToTensor(), 
    T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)])

if __name__ == '__main__':

    # Loading the model
    if args.resume is not None:
        model = AutoModelForImageClassification.from_pretrained(args.resume)
    else:
        model = AutoModelForImageClassification.from_pretrained(config.model)
    model.to(device)
    print('Number of parameters:', model.num_parameters())
    print('Number of classes:', model.config.num_labels)

    torch.cuda.empty_cache()

    # validation transforms just resizes the image to the model's input size, without cropping
    dataset = load_dataset(config.train_dataset, 'train')
    
    train_test_split = dataset.train_test_split(
        test_size=config.eval_dataset_split_size,
        train_size=len(dataset) - config.eval_dataset_split_size,
        shuffle=True,
        seed=42        
    )
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    if accelerator.is_main_process:    
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")

    global loss_fn
    class_weights = get_class_weights(train_dataset)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    
    if accelerator.is_main_process:
        print('Class weights calculated:', class_weights)

    # Setting up Trainer

    if config.num_epochs is None:
        num_epochs = (config.max_steps * config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes) // len(train_dataset)
    else:
        num_epochs = config.num_epochs

    # Optimizer and Scheduler

    num_training_steps = num_epochs * len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    if config.scheduler == "cosine" and config.eta_min != 0.0:
        scheduler = CosineAnnealingWithWarmupAndEtaMin(
            optimizer,
            T_max=num_training_steps,
            eta_min=config.eta_min,
            warmup_steps=config.warmup_steps
        )
    else:
        scheduler = get_scheduler(
            config.scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )

    train_dataset = train_dataset.with_transform(transforms)
    eval_dataset = eval_dataset.with_transform(val_transforms)

    if accelerator.is_main_process:
        print('--- Hyperparameters ---')
        for key in config._jsonData.keys():
            print(f"{key}: {config._jsonData[key]}")
        print('-----------------------')

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        remove_unused_columns=False,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        lr_scheduler_type=config.scheduler,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        logging_dir=config.output_dir,
        save_strategy="steps" if config.save_steps is not None else "epoch",
        save_steps=config.save_steps if config.save_steps is not None else None,
        eval_strategy="steps" if config.save_steps is not None or config.eval_steps is not None else "epoch",
        eval_steps=config.eval_steps if config.eval_steps is not None else config.save_steps if config.save_steps is not None else None,
        seed=4242,
        bf16=True,
        report_to="wandb" if args.wandb else "none",
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=False,
        dataloader_num_workers=config.num_workers,
        warmup_steps=config.warmup_steps,
        resume_from_checkpoint=args.resume if args.resume is not None else None,
    )

    data_collator = DefaultDataCollator()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[],
    )

    if args.wandb and accelerator.is_main_process:
        wandb.init(project=config.wandb['project'], name=config.wandb['name'], tags=config.wandb['tags'])
        wandb.config.update(config._jsonData)
        wandb.watch(model)

    model.config.use_cache = False 

    model, optimizer, scheduler, train_dataset, eval_dataset = accelerator.prepare(
        model, optimizer, scheduler, train_dataset, eval_dataset
    )

    trainer.train()