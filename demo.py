import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset

# Check for GPU availability and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# File path for the test data
test_file = "processed_test_data.csv"

# Load the test dataset from a CSV file using the `datasets` library
print("Loading test dataset...")
test_dataset = load_dataset("csv", data_files={"test": test_file}, split="test")

# Rename dataset columns to match expected input and target names
# This ensures consistency with the model's input requirements
test_dataset = test_dataset.rename_columns({"bert_description": "input_text", "bart_abstract": "target_text"})

# Define the path to the saved model weights
model_load_path = "BART/Bart_trained_model.pt"  # Update with the correct path if different

# Initialize the BART tokenizer from the pre-trained `facebook/bart-base` model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")  # Tokenizer converts text to tokens

# Initialize the BART model for conditional generation from the pre-trained `facebook/bart-base` model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)  # Move model to device

# Load the trained model weights from the specified path
# `map_location=device` ensures that the weights are loaded onto the correct device (GPU or CPU)
model.load_state_dict(torch.load(model_load_path, map_location=device))

# Set the model to evaluation mode to disable dropout and other training-specific layers
model.eval()

# Select the first example from the test dataset for evaluation
test_sample = test_dataset.select([0])  # Select the first description

# Tokenize the input text of the selected test sample
# This prepares the input for the model by converting text to token IDs
tokenized_test_sample = tokenizer(
    test_sample[0]["input_text"],  # Access the input_text of the first test sample
    max_length=512,               # Maximum length of the tokenized input
    truncation=True,              # Truncate inputs longer than max_length
    padding="max_length",         # Pad inputs shorter than max_length
    return_tensors="pt"           # Return PyTorch tensors
)

# Move the tokenized input tensors to the selected device (GPU or CPU)
input_ids = tokenized_test_sample["input_ids"].to(device)         # Input token IDs (shape: [1, 512])
attention_mask = tokenized_test_sample["attention_mask"].to(device)  # Attention mask (shape: [1, 512])

# Generate predictions for the first test sample using the BART model
with torch.no_grad():  # Disable gradient calculations for inference
    generated_ids = model.generate(
        input_ids=input_ids,           # Input token IDs
        attention_mask=attention_mask, # Attention mask to ignore padding tokens
        max_length=128                 # Maximum length of the generated abstract
    )
    # Decode the generated token IDs back to human-readable text
    decoded_prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Print the input description and the generated abstract
print("Input Description:")
print(test_sample[0]["input_text"])  # Display the original input text

print("\nGenerated Abstract:")
print(decoded_prediction)  # Display the model's generated abstract
