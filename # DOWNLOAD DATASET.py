# DOWNLOAD DATASET

%%writefile scripts/download_data.py
# ==============================================================================
# Name: Zarar Bin Akram
# SRN: 303-221057
# File: scripts/download_data.py
# Task: 1.1 - Download the dataset into the repo structure
# ==============================================================================

import os
import urllib.request
import numpy as np

def fetch_bloodmnist_arrays(target_dir="data/raw"):
    """
    Purpose:
        Automates downloading the raw NumPy data array files for the
        BloodMNIST dataset and saves them directly to the repository structure.
    Inputs:
        target_dir (str): The local directory path where the raw .npz file
                          will be written. Defaults to "data/raw".
    Outputs:
        local_path (str): The absolute file path of the downloaded dataset file.
    Assumptions:
        Assumes an active internet connection to communicate with the Zenodo
        public medical data hosting servers.
    """
    # Public hosting URL for the standardized BloodMNIST dataset (.npz format)
    source_url = "https://zenodo.org/record/6496656/files/bloodmnist.npz?download=1"

    os.makedirs(target_dir, exist_ok=True)
    local_path = os.path.join(target_dir, "bloodmnist.npz")

    print(f"[INFO] Initiating data download from: {source_url}")
    print(f"[INFO] Targeting location: {local_path}...")

    # Execute stream download
    urllib.request.urlretrieve(source_url, local_path)

    print("[INFO] Download finalized successfully.")
    return local_path

if __name__ == "__main__":
    file_path = fetch_bloodmnist_arrays()

    # Quick sanity check to verify file size and existence
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"[SUCCESS] File verified at {file_path} ({size_mb:.2f} MB).")