import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

# load BERT and BART tokenizers
bert_tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# --- CLEANING FUNCTIONS ---
def clean_text(text):
    """
    Clean the input text by removing extra spaces, non-ASCII characters,
    and long sequences of underscores.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra spaces, line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove long sequences of underscores (e.g., "____" or more)
    text = re.sub(r'_{5,}', ' ', text)  # Replace sequences of 5 or more underscores with a space
    
    return text

def split_into_sentences(text):
    """
    Split the text into sentences using NLTK's sentence tokenizer.
    """
    return sent_tokenize(text)

def truncate_text(text, max_length, tokenizer):
    """
    Truncate text to fit within the model's max token limit.
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.convert_tokens_to_string(tokens)

# --- PREPROCESSING PIPELINE ---
def preprocess_dataset(input_csv, output_csv, bert_tokenizer, bart_tokenizer, max_length=512):
    """
    Preprocess dataset by cleaning and tokenizing using BERT and BART tokenizers.
    """
    # Read dataset
    df = pd.read_csv(input_csv)
    cleaned_data = []

    for _, row in df.iterrows():
        # Clean text fields
        description = clean_text(row.get('description', ''))
        abstract = clean_text(row.get('abstract', ''))
        
        # Skip rows with missing or empty fields
        if not description or not abstract:
            continue
        
        # Split text into sentences
        description_sentences = split_into_sentences(description)
        abstract_sentences = split_into_sentences(abstract)

        # Tokenize each sentence and truncate to fit the max length
        bert_description = [truncate_text(sentence, max_length, bert_tokenizer) for sentence in description_sentences]
        bart_abstract = [truncate_text(sentence, max_length, bart_tokenizer) for sentence in abstract_sentences]

        # Join the sentences back together
        bert_description = " ".join(bert_description)
        bart_abstract = " ".join(bart_abstract)

        # Append cleaned and tokenized data
        cleaned_data.append({
            'bert_description': bert_description,
            'bart_abstract': bart_abstract
        })
    
    # Convert to DataFrame and save
    cleaned_df = pd.DataFrame(cleaned_data)
    cleaned_df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

# --- MAIN FUNCTION ---
if __name__ == "__main__":
    # File paths
    input_csv = "/Users/hibah/Documents/CS 6120/project/train_data.csv"
    output_csv = "/Users/hibah/Documents/CS 6120/project/processed_train_data.csv"

    # Preprocess the data
    preprocess_dataset(input_csv, output_csv, bert_tokenizer, bart_tokenizer, max_length=512)