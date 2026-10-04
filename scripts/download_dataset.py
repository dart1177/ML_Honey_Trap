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
    dataset_name = "zyw-286/shell-attack-evolution-dataset"
    base_output_dir = "data/shell_dataset"
    
    try:
        logging.info(f"Downloading request_response (curated) from '{dataset_name}'...")
        rr = load_dataset(dataset_name, "request_response", split="curated")
        rr_dir = os.path.join(base_output_dir, "request_response")
        os.makedirs(rr_dir, exist_ok=True)
        rr.save_to_disk(rr_dir)
        logging.info(f"Saved request_response to {rr_dir}")

        logging.info(f"Downloading sessions (2024) from '{dataset_name}'...")
        sess = load_dataset(dataset_name, "sessions", split="2024")
        sess_dir = os.path.join(base_output_dir, "sessions")
        os.makedirs(sess_dir, exist_ok=True)
        sess.save_to_disk(sess_dir)
        logging.info(f"Saved sessions to {sess_dir}")
        
    except Exception as e:
        logging.error(f"Failed to download or save dataset: {e}")

if __name__ == "__main__":
    main()
