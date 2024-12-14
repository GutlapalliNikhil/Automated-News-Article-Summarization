# BigPatent Dataset Summarization

This repository contains code and resources for summarizing patent descriptions using transformer-based models. We utilized three approaches: BERT for extractive summarization, BART for abstractive summarization, and a hybrid model combining both. Below are detailed instructions on how to run the code and use the models.

## Repository Structure
```
|-- BART/
|   |-- bart_model_training.py       # Training script for BART
|   |-- bart_trained_model.pt       # Saved weights for BART model
|-- BERT/
|   |-- BERT_model.ipynb       # Training script for BERT
|   |-- BERT_saved_model.pt     # Saved weights for BERT model
|-- Hybrid Model/
|   |-- Hybrid_trained_model.pt     # Training script for Hybrid Model
|   |-- hybrid_model_training.py     # Saved weights for Hybrid model
|-- README.md               # Project instructions
|-- calculate_accuracy.py   # Script to calculate accuracy metrics
|-- demo.py                 # Script to generate test outputs
|-- extract_data.py         # Script to extract sample data from BigPatent dataset
|-- preprocess_data.py      # Script to preprocess the extracted dataset
|-- processed_test_data.csv # Preprocessed testing data
|-- processed_train_data.csv# Preprocessed training data
|-- processed_val_data.csv  # Preprocessed validation data
```

## Installation

#### Clone the repository:
```
git clone https://github.com/GutlapalliNikhil/BigPatent-Dataset-Summarization.git
cd BigPatent-Dataset-Summarization
```
#### Install the required dependencies:

Make sure you have installed GIT-LFS in your machine

```
git lfs install
```
Ensure you have Python 3.8 or higher and a compatible GPU setup for training.

## Data Preparation

#### Extract Sample Data:

Place the raw BigPatent dataset files (in .tar.gz format) in the input_path directory specified in extract_data.py.

Run the script to extract and create training, validation, and testing CSV files:
```
python3 extract_data.py
```
This will generate train_data.csv, val_data.csv, and test_data.csv in the project directory.

#### Preprocess Data:

Use preprocess_data.py to clean and tokenize the extracted dataset:
```
python3 preprocess_data.py
```
This will generate preprocessed CSV files: processed_train_data.csv, processed_val_data.csv, and processed_test_data.csv.

## Training

Train the models using the preprocessed data.

#### Train BERT Model:

Please use .piynb file

#### Train BART Model:
```
python3 BART/bart_model_training.py
```
#### Train Hybrid Model:
```
python3 Hybrid Model/hybrid_model_training.py
```
## Testing

To test the models with a single case, run:
```
python3 demo.py
```
Ensure the correct model weights (.pt files) are updated in demo.py before running.

## Accuracy Evaluation

To calculate BLEU and ROUGE scores, use:
```
python3 calculate_accuracy.py
```
