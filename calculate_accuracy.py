import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
from evaluate import load

# Check for GPU availability and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define file paths for the validation dataset and the trained model
val_file = "processed_val_data.csv"
model_path = "BART/Bart_trained_model.pt"  # Update to the model you want to test

# Load the validation dataset from a CSV file
print("Loading validation dataset...")
val_dataset = load_dataset("csv", data_files={"val": val_file}, split="val")

# Rename dataset columns to match expected input and target names
val_dataset = val_dataset.rename_columns({"bert_description": "input_text", "bart_abstract": "target_text"})

# Initialize the BART tokenizer and model for conditional generation
print("Loading tokenizer and model...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Load the trained model weights from the specified path
model.load_state_dict(torch.load(model_path, map_location=device))

# Move the model to the selected device (GPU or CPU)
model = model.to(device)

# Define a tokenization function to process input and target texts
def tokenize_function(examples):
    # Tokenize the input text with padding and truncation
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize the target text (labels) similarly
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    
    # Add the tokenized labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the validation dataset
print("Tokenizing validation dataset...")
tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["input_text", "target_text"]
)

# Set the format of the tokenized dataset to PyTorch tensors
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Create a DataLoader for the validation dataset with a batch size of 1
val_dataloader = DataLoader(tokenized_val, batch_size=1)

# Load evaluation metrics for BLEU and ROUGE
bleu_metric = load("bleu")
rouge_metric = load("rouge")

# Begin the evaluation process
print("Starting evaluation...")
model.eval()  # Set the model to evaluation mode
generated_texts = []  # List to store generated texts
reference_texts = []  # List to store reference (ground truth) texts
progress_bar = tqdm(val_dataloader, desc="Evaluating")  # Progress bar for validation

with torch.no_grad():  # Disable gradient computation for efficiency
    for batch in progress_bar:
        # Move batch data to the selected device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Generate predictions using the model
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128
        )
        
        # Decode the generated token IDs to strings
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # Decode the reference token IDs to strings
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Append the decoded predictions and references to their respective lists
        generated_texts.extend(decoded_preds)
        reference_texts.extend([[ref] for ref in decoded_labels])  # ROUGE expects list of lists for references

# Calculate BLEU score using the generated and reference texts
bleu_score = bleu_metric.compute(predictions=generated_texts, references=reference_texts)

# Calculate ROUGE score using the generated and reference texts
rouge_score = rouge_metric.compute(predictions=generated_texts, references=reference_texts)

# Print the evaluation metrics
print(f"BLEU Score: {bleu_score['bleu']:.4f}")
print(f"ROUGE Score: {rouge_score}")
