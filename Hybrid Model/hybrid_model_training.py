import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration, AdamW, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
from evaluate import load

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# File paths
train_file = "processed_train_data.csv"
val_file = "processed_val_data.csv"
log_file = "hybrid_training_log.txt"  # File to save the training logs

# Load datasets
print("Loading datasets...")
train_dataset = load_dataset("csv", data_files={"train": train_file}, split="train")
val_dataset = load_dataset("csv", data_files={"val": val_file}, split="val")

# Rename columns to expected format
train_dataset = train_dataset.rename_columns({"bert_description": "input_text", "bart_abstract": "target_text"})
val_dataset = val_dataset.rename_columns({"bert_description": "input_text", "bart_abstract": "target_text"})

# Initialize tokenizers and models
print("Loading tokenizers and models...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
bert_model = BertModel.from_pretrained("bert-large-uncased").to(device)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)

# Freeze initial layers of BERT and BART
for param in bert_model.encoder.layer[:12].parameters():
    param.requires_grad = False
for param in bart_model.model.encoder.layers[:6].parameters():
    param.requires_grad = False

# Tokenization function
def tokenize_function(examples):
    # Tokenize with BERT for input
    bert_inputs = bert_tokenizer(
        examples["input_text"], max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )

    # Tokenize with BART for target
    with bart_tokenizer.as_target_tokenizer():
        labels = bart_tokenizer(
            examples["target_text"], max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        )

    # Combine input and labels
    return {
        "input_ids": bert_inputs["input_ids"].squeeze(),
        "attention_mask": bert_inputs["attention_mask"].squeeze(),
        "labels": labels["input_ids"].squeeze(),
    }

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

# Optimizer and Scheduler
optimizer = AdamW(list(bert_model.parameters()) + list(bart_model.parameters()), lr=3e-5, weight_decay=1e-2)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=50 * len(train_dataloader))

# Metrics
bleu_metric = load("bleu")
rouge_metric = load("rouge")

# Initialize log data
log_data = []

# Training loop with validation and saving
epochs = 50
save_every = 5
print("Starting training with validation...")
for epoch in range(epochs):
    # Training phase
    bert_model.train()
    bart_model.train()
    train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Pass input through BERT encoder
        encoder_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden_states = encoder_outputs.last_hidden_state

        # Pass BERT output to BART decoder
        outputs = bart_model(
            inputs_embeds=bert_hidden_states, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bart_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_train_loss:.4f}")

    # Save model every `save_every` epochs
    if (epoch + 1) % save_every == 0:
        bert_save_path = f"bert_model_epoch_{epoch + 1}.pt"
        bart_save_path = f"bart_model_epoch_{epoch + 1}.pt"
        torch.save(bert_model.state_dict(), bert_save_path)
        torch.save(bart_model.state_dict(), bart_save_path)
        print(f"Models saved at {bert_save_path} and {bart_save_path}")

    # Validation phase
    bert_model.eval()
    bart_model.eval()
    val_loss = 0
    generated_texts = []
    reference_texts = []
    progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Validation")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Pass input through BERT encoder
            encoder_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            bert_hidden_states = encoder_outputs.last_hidden_state

            # Pass BERT output to BART decoder
            outputs = bart_model(
                inputs_embeds=bert_hidden_states, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            val_loss += loss.item()

            # Generate predictions
            generated_ids = bart_model.generate(inputs_embeds=bert_hidden_states, max_length=128)
            decoded_preds = bart_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_labels = bart_tokenizer.batch_decode(labels, skip_special_tokens=True)

            generated_texts.extend(decoded_preds)
            reference_texts.extend([[ref] for ref in decoded_labels])

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Validation Loss: {avg_val_loss:.4f}")

    # Calculate BLEU and ROUGE
    bleu_score = bleu_metric.compute(predictions=generated_texts, references=reference_texts)
    rouge_score = rouge_metric.compute(predictions=generated_texts, references=reference_texts)

    print(f"Epoch {epoch + 1}/{epochs} - BLEU Score: {bleu_score['bleu']:.4f}")
    print(f"Epoch {epoch + 1}/{epochs} - ROUGE Score: {rouge_score}")

    # Log data
    log_data.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "bleu_score": bleu_score["bleu"],
        "rouge_score": rouge_score
    })

# Write log data to a text file
with open(log_file, "w") as log_f:
    log_f.write("Epoch\tTrain Loss\tVal Loss\tBLEU Score\tROUGE Score\n")
    for entry in log_data:
        log_f.write(f"{entry['epoch']}\t{entry['train_loss']:.4f}\t{entry['val_loss']:.4f}\t"
                    f"{entry['bleu_score']:.4f}\t{entry['rouge_score']}\n")

print(f"Training log saved to {log_file}")

