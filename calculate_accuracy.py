import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
from evaluate import load

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# File path
val_file = "tested_data/processed_val_data.csv"
model_path = "bart_model_epoch_1.pt"  # Update to the model you want to test

# Load the validation dataset
print("Loading validation dataset...")
val_dataset = load_dataset("csv", data_files={"val": val_file}, split="val")

# Rename columns to expected format
val_dataset = val_dataset.rename_columns({"bert_description": "input_text", "bart_abstract": "target_text"})

# Initialize tokenizer and model
print("Loading tokenizer and model...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# Tokenization function
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize validation dataset
print("Tokenizing validation dataset...")
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])

# Convert to PyTorch format
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# DataLoader
val_dataloader = DataLoader(tokenized_val, batch_size=1)

# Metrics
bleu_metric = load("bleu")
rouge_metric = load("rouge")

# Evaluate model
print("Starting evaluation...")
model.eval()
generated_texts = []
reference_texts = []
progress_bar = tqdm(val_dataloader, desc="Evaluating")
with torch.no_grad():
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Generate predictions
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        generated_texts.extend(decoded_preds)
        reference_texts.extend([[ref] for ref in decoded_labels])

# Calculate BLEU and ROUGE
bleu_score = bleu_metric.compute(predictions=generated_texts, references=reference_texts)
rouge_score = rouge_metric.compute(predictions=generated_texts, references=reference_texts)

print(f"BLEU Score: {bleu_score['bleu']:.4f}")
print(f"ROUGE Score: {rouge_score}")

