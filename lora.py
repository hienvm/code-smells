import os
import sys
import time
import torch
import pandas as pd

from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    PeftType,
    LoraConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PeftConfig,
    PeftModel
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    get_linear_schedule_with_warmup, 
    set_seed,
)
use_cuda = True

checkpoint = "Salesforce/codet5p-110m-embedding"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_file = 'data_class/data/embedded_datasets/data_class.pkl'


# Set peft config
peft_type = PeftType.LORA

peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "k", "v"]
)

# Load tokenizer
padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def load_trainset(train_file, max_train_samples=None, seed=42):
    train_df = pd.read_csv(train_file)

    if max_train_samples != None:
        sample_df = pd.DataFrame()
        total_samples = train_df.shape[0]
        label_counts = train_df["label"].value_counts()

        for label, item in label_counts.items():
            sample_count = int(item / total_samples * max_train_samples)
            label_df = train_df[train_df["label"] == label].sample(n=sample_count, replace=False, random_state=seed, ignore_index=True)
            sample_df = pd.concat([sample_df, label_df], ignore_index=True)

        if sample_df.shape[0] < max_train_samples:
            additional_samples = max_train_samples - sample_df.shape[0]
            additional_df = train_df.sample(n=additional_samples, replace=False, random_state=seed, ignore_index=True)
            sample_df = pd.concat([sample_df, additional_df], ignore_index=True)

        print(f"Sampled {sample_df.shape[0]} samples")

        train_df = sample_df

    return train_df

# Load datasets
train_df = load_trainset(train_file, seed=42)
eval_df = train_df

trainset = Dataset.from_pandas(train_df)
evalset = Dataset.from_pandas(eval_df)

datasets = DatasetDict({
    'train': trainset,
    'validation': evalset
})


# Get the number of labels
label_list = train_df["label"].unique()
num_labels = len(label_list)


def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "text_label"],
    load_from_cache_file=False
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

def collate_fn(examples):
    outputs = tokenizer.pad(examples, return_tensors="pt", padding=True, max_length=512)
    return outputs

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

valid_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)


# Load model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Instantiate optimizer
optimizer = AdamW(model.parameters(), lr=3e-4)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * 10),
    num_training_steps=(len(train_dataloader) * 10)
)

total_steps = 0
best_validation_loss = float("inf")
peak_memory = 0
if use_cuda:
    model.cuda()

# Training
start_time = time.time()
for epoch in range(10):
    model.train()
    train_loss = 0.0

    # progress_bar_train = tqdm(
    #     total=len(train_dataloader), 
    #     desc=f"Training epoch {epoch + 1}",
    #     position=0,
    #     mininterval=1,
    #     leave=True
    # )

    for step, batch in enumerate(train_dataloader):
        total_steps += 1
        batch = {k: v.cuda() for k, v in batch.items()} if use_cuda else batch
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # if step % 5 == 0:
        #     progress_bar_train.set_postfix({"loss": loss.item()})
        #     progress_bar_train.update(5)

        current_memory = torch.cuda.max_memory_allocated()
        if current_memory > peak_memory:
            peak_memory = current_memory

    # progress_bar_train.close()

    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Training loss: {avg_train_loss}")

    # Validation
    model.eval()
    total_validation_loss = 0.0

    # progress_bar_valid = tqdm(
    #     total=len(valid_dataloader),
    #     desc=f"Validation epoch {epoch + 1}",
    #     position=0,
    #     mininterval=1,
    #     leave=True
    # )

    for step, batch in enumerate(valid_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()} if use_cuda else batch
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_validation_loss += loss.item()

    #     if step % 5 == 0:
    #         progress_bar_valid.update(5)
    # progress_bar_valid.close()

    avg_validation_loss = total_validation_loss / len(valid_dataloader)
    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        best_model_path = os.path.join('models', checkpoint, f"lora_seed_{42}", "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        model.save_pretrained(best_model_path)

    print(f"Epoch {epoch + 1} - Validation loss: {avg_validation_loss}")

    save_path = os.path.join('models', checkpoint, f"lora_seed_{42}", f"epoch_{epoch + 1}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

with open(f"models/{checkpoint}/peak_memory.txt", "a") as f:
    f.write(f"lora: {str(peak_memory)}")

end_time = time.time()

training_time = end_time - start_time

with open(f"models/{checkpoint}/training_time.txt", "a") as f:
    f.write(f"epoch: {10} lora: {str(training_time)}")