"""
Script to download the Shell Honeypot dataset from HuggingFace.
"""

import logging
from datasets import load_dataset
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main() -> None:
    """Download and save the dataset."""
    dataset_name = "Ziyang23423432/shell-attack-evolution-dataset"
    output_dir = "data/shell_dataset"
    
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    try:
        logging.info(f"Downloading dataset '{dataset_name}'...")
        ds = load_dataset(dataset_name)
        logging.info(f"Dataset downloaded successfully. Saving to {output_dir}...")
        ds.save_to_disk(output_dir)
        logging.info("Dataset saved successfully.")
    except Exception as e:
        logging.error(f"Failed to download or save dataset: {e}")

if __name__ == "__main__":
    main()
