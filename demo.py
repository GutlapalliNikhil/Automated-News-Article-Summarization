import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# File path for the test data
test_file = "processed_test_data.csv"

# Load test dataset
print("Loading test dataset...")
test_dataset = load_dataset("csv", data_files={"test": test_file}, split="test")

# Rename columns to expected format
test_dataset = test_dataset.rename_columns({"bert_description": "input_text", "bart_abstract": "target_text"})

# Load the saved model and tokenizer
model_load_path = "BART/bart_trained_model.pt"  # Update with the correct path if different
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")  # Initialize the tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)  # Initialize the model
model.load_state_dict(torch.load(model_load_path))
model.eval()

# Tokenize test dataset (only the first example)
test_sample = test_dataset.select([0])  # Select the first description
tokenized_test_sample = tokenizer(
    test_sample[0]["input_text"], max_length=512, truncation=True, padding="max_length", return_tensors="pt"
)

# Ensure tensors are moved to the correct device
input_ids = tokenized_test_sample["input_ids"].to(device)  # Already 2D
attention_mask = tokenized_test_sample["attention_mask"].to(device)  # Already 2D

# Generate predictions for the first test sample
with torch.no_grad():
    generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
    decoded_prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Print the input and the generated output
print("Input Description:")
print(test_sample[0]["input_text"])

print("\nGenerated Abstract:")
print(decoded_prediction)
