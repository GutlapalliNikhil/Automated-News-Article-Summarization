import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from datasets import load_dataset
from tqdm import tqdm
from evaluate import load

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# File paths
train_file = "tested_data/processed_train_data.csv"
val_file = "tested_data/processed_val_data.csv"

# Load datasets
print("Loading datasets...")
train_dataset = load_dataset("csv", data_files={"train": train_file}, split="train")
val_dataset = load_dataset("csv", data_files={"val": val_file}, split="val")

# Rename columns to expected format
train_dataset = train_dataset.rename_columns({"bert_description": "input_text", "bart_abstract": "target_text"})
val_dataset = val_dataset.rename_columns({"bert_description": "input_text", "bart_abstract": "target_text"})

# Initialize tokenizer and model
print("Loading tokenizer and model...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)

# Tokenization function
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])

# Convert to PyTorch format
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# DataLoaders
train_dataloader = DataLoader(tokenized_train, batch_size=1, shuffle=True)
val_dataloader = DataLoader(tokenized_val, batch_size=1)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Metrics
bleu_metric = load("bleu")
rouge_metric = load("rouge")

# Training loop with validation and saving
epochs = 1
save_every = 1
print("Starting training with validation...")
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_train_loss:.4f}")

    # Save model every `save_every` epochs
    if (epoch + 1) % save_every == 0:
        model_save_path = f"bart_model_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

    # Validation phase
    model.eval()
    val_loss = 0
    generated_texts = []
    reference_texts = []
    progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Validation")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            # Generate predictions
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            generated_texts.extend(decoded_preds)
            reference_texts.extend([[ref] for ref in decoded_labels])

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Validation Loss: {avg_val_loss:.4f}")

    # Calculate BLEU and ROUGE
    bleu_score = bleu_metric.compute(predictions=generated_texts, references=reference_texts)
    rouge_score = rouge_metric.compute(predictions=generated_texts, references=reference_texts)

    print(f"Epoch {epoch + 1}/{epochs} - BLEU Score: {bleu_score['bleu']:.4f}")
    print(f"Epoch {epoch + 1}/{epochs} - ROUGE Score: {rouge_score}")

print("Training and validation complete!")

