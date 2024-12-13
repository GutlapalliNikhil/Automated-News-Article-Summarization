import tarfile
import os
import gzip
import json
import pandas as pd

# extract .tar.gz files
def extract_tar_files(input_path):
    for file_name in os.listdir(input_path):
        if file_name.endswith('.tar.gz'):
            file_path = os.path.join(input_path, file_name)
            print(f"Extracting {file_path}...")

            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=input_path)
                print(f"Extracted {file_path}")

# extract .gz files
def extract_gz_files(input_path, split_type, max_files=10, max_lines=100):
    data = []
    files_processed = 0
    lines_processed = 0

    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".gz") and files_processed < max_files:
                gz_path = os.path.join(root, file)
                print(f"Processing .gz file: {gz_path}")
                files_processed += 1

                try:
                    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if lines_processed >= max_lines:
                                break

                            try:
                                json_obj = json.loads(line)
                                description = json_obj.get('description', '')
                                abstract = json_obj.get('abstract', '')

                                if description and abstract:
                                    data.append({
                                        'description': description,
                                        'abstract': abstract
                                    })
                                lines_processed += 1
                            except json.JSONDecodeError:
                                print(f"Skipping invalid JSON line in file: {gz_path}")

                    if lines_processed >= max_lines:
                        break
                except Exception as e:
                    print(f"Error processing file {gz_path}: {e}")
            
            if lines_processed >= max_lines:
                break

        if lines_processed >= max_lines:
            break

    return pd.DataFrame(data)

# save subset of data to CSV
def save_split_data(input_path, split_type, max_files=10, max_lines=100):
    print(f"Processing {split_type} data...")
    df = extract_gz_files(input_path, split_type, max_files, max_lines)

    print(f"DataFrame for {split_type} split has {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Columns in DataFrame: {df.columns}")

    output_file = f"{split_type}_data.csv"
    df.to_csv(output_file, index=False)
    print(f"{split_type} data saved to {output_file}")

input_path = "/Users/hibah/Documents/CS 6120/project/bigPatentData"

extract_tar_files(input_path)

# save a subset of data for training, validation, and testing
save_split_data(input_path, "train", max_files=5, max_lines=100)
save_split_data(input_path, "val", max_files=5, max_lines=100)
save_split_data(input_path, "test", max_files=5, max_lines=100)