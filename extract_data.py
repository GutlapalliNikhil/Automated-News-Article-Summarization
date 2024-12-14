import tarfile
import os
import gzip
import json
import pandas as pd

# Function to extract all .tar.gz files in the specified input directory
def extract_tar_files(input_path):
    """
    Extracts all .tar.gz files in the given input directory.

    Args:
        input_path (str): Path to the directory containing .tar.gz files.
    """
    # Iterate over all files in the input directory
    for file_name in os.listdir(input_path):
        # Check if the file has a .tar.gz extension
        if file_name.endswith('.tar.gz'):
            file_path = os.path.join(input_path, file_name)  # Full path to the .tar.gz file
            print(f"Extracting {file_path}...")  # Inform the user about the extraction process

            # Open and extract the .tar.gz file
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=input_path)  # Extract contents to the input directory
                print(f"Extracted {file_path}")  # Confirm extraction completion

# Function to extract data from .gz files within the input directory
def extract_gz_files(input_path, split_type, max_files=10, max_lines=100):
    """
    Extracts data from .gz files, parses JSON lines, and collects descriptions and abstracts.

    Args:
        input_path (str): Path to the directory containing .gz files.
        split_type (str): Type of data split (e.g., 'train', 'val', 'test') for logging purposes.
        max_files (int, optional): Maximum number of .gz files to process. Defaults to 10.
        max_lines (int, optional): Maximum number of valid JSON lines to process. Defaults to 100.

    Returns:
        pd.DataFrame: DataFrame containing 'description' and 'abstract' columns.
    """
    data = []  # List to store extracted data
    files_processed = 0  # Counter for the number of files processed
    lines_processed = 0  # Counter for the number of lines processed

    # Walk through the directory tree starting at input_path
    for root, _, files in os.walk(input_path):
        for file in files:
            # Process only .gz files and ensure we don't exceed the max_files limit
            if file.endswith(".gz") and files_processed < max_files:
                gz_path = os.path.join(root, file)  # Full path to the .gz file
                print(f"Processing .gz file: {gz_path}")  # Inform the user about the file being processed
                files_processed += 1  # Increment the file counter

                try:
                    # Open the .gz file in text mode with UTF-8 encoding
                    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            # Stop processing if we've reached the maximum number of lines
                            if lines_processed >= max_lines:
                                break

                            try:
                                # Parse the JSON object from the current line
                                json_obj = json.loads(line)
                                description = json_obj.get('description', '')  # Extract 'description'
                                abstract = json_obj.get('abstract', '')        # Extract 'abstract'

                                # If both description and abstract are present, add them to the data list
                                if description and abstract:
                                    data.append({
                                        'description': description,
                                        'abstract': abstract
                                    })
                                lines_processed += 1  # Increment the line counter
                            except json.JSONDecodeError:
                                # Skip lines that are not valid JSON
                                print(f"Skipping invalid JSON line in file: {gz_path}")

                    # Break out of the loop if we've processed enough lines
                    if lines_processed >= max_lines:
                        break
                except Exception as e:
                    # Handle any unexpected errors during file processing
                    print(f"Error processing file {gz_path}: {e}")
            
            # Break out of the loop if we've processed enough lines
            if lines_processed >= max_lines:
                break

        # Break out of the outer loop if we've processed enough lines
        if lines_processed >= max_lines:
            break

    # Convert the collected data into a pandas DataFrame
    return pd.DataFrame(data)

# Function to save a subset of the data to a CSV file for a specific data split
def save_split_data(input_path, split_type, max_files=10, max_lines=100):
    """
    Processes and saves a subset of data for a specific split (train, val, test) to a CSV file.

    Args:
        input_path (str): Path to the directory containing .gz files.
        split_type (str): Type of data split (e.g., 'train', 'val', 'test').
        max_files (int, optional): Maximum number of .gz files to process. Defaults to 10.
        max_lines (int, optional): Maximum number of valid JSON lines to process. Defaults to 100.
    """
    print(f"Processing {split_type} data...")  # Inform the user about the data split being processed
    df = extract_gz_files(input_path, split_type, max_files, max_lines)  # Extract and collect data

    # Display the shape and columns of the resulting DataFrame for verification
    print(f"DataFrame for {split_type} split has {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Columns in DataFrame: {df.columns}")

    # Define the output CSV file name based on the split type
    output_file = f"{split_type}_data.csv"
    df.to_csv(output_file, index=False)  # Save the DataFrame to a CSV file without the index
    print(f"{split_type} data saved to {output_file}")  # Confirm that the data has been saved

# Define the input directory path containing the .tar.gz and .gz files
input_path = "/Users/hibah/Documents/CS 6120/project/bigPatentData"

# Extract all .tar.gz files in the input directory
extract_tar_files(input_path)

# Save subsets of data for training, validation, and testing
# Adjust 'max_files' and 'max_lines' as needed to control the amount of data processed
save_split_data(input_path, "train", max_files=5, max_lines=100)
save_split_data(input_path, "val", max_files=5, max_lines=100)
save_split_data(input_path, "test", max_files=5, max_lines=100)
