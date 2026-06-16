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


#----------------------------NEXT STEP--------------------------
        # DATA VALIDATION AND CHECKS

# ==============================================================================
# Name: Zarar Bin Akram
# SRN: 303-221057
# File: Task 1.2 - Data Validation Checks
# Description: Implements comprehensive multi-point validation to check shapes,
#              detect any corrupted/NaN matrices, and plot class distributions.
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

def run_dataset_validation(file_path="data/raw/bloodmnist.npz"):
    """
    Purpose:
        Validates the dataset arrays by checking dimensions, verifying that
        no files are corrupted or unreadable (NaN/Inf checks), confirming
        the label ranges map correctly, and outputting distribution graphs.
    Inputs:
        file_path (str): The local path to the downloaded dataset array.
                         Defaults to "data/raw/bloodmnist.npz".
    Outputs:
        validated_data (dict): Dictionary holding the safely loaded train,
                               validation, and test numpy arrays.
    Assumptions:
        Assumes the dataset was successfully downloaded by Task 1.1.
        Assumes the dataset is a standard BloodMNIST archive with 8 distinct classes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Dataset file missing at {file_path}. Please execute Task 1.1 first!")

    print(f"[INFO] Initializing validation framework on target: {file_path}")
    raw_data = np.load(file_path)

    splits = ['train', 'val', 'test']
    validated_data = {}

    print("\n" + "="*60 + "\n[START] EXECUTING STRUCTURAL VALIDATION CHECKS\n" + "="*60)

    for split in splits:
        img_key = f'{split}_images'
        lbl_key = f'{split}_labels'

        images = raw_data[img_key]
        labels = raw_data[lbl_key].flatten()

        # 1. Validation Check: Shape & Structural Dimensions
        print(f"[{split.upper()} SPLIT]")
        print(f"  -> Image Array Dimensions : {images.shape}")
        print(f"  -> Label Array Dimensions : {labels.shape}")

        # 2. Validation Check: Corruption Scans (Checking for NaNs or Infinite Values)
        if np.isnan(images).any() or np.isnan(labels).any():
            raise ValueError(f"[ERROR] Corrupted images or labels detected! NaNs discovered in {split} split.")
        if np.isinf(images).any() or np.isinf(labels).any():
            raise ValueError(f"[ERROR] Corrupted elements detected! Infinite values discovered in {split} split.")
        print("  -> Corrupt/Unreadable Scan: Passed (No corrupted files or NaNs found).")

        # 3. Validation Check: Class Label Range Mapping
        unique_labels = np.unique(labels)
        print(f"  -> Found Unique Mapped Classes: {unique_labels}")
        if min(unique_labels) < 0 or max(unique_labels) > 7:
            raise ValueError(f"[ERROR] Class label mapping mismatch! Indices sit outside expected [0, 7] range.")
        print("  -> Label Range Mapping Check: Passed (All mappings match correctly).")
        print("-" * 60)

        # Cache data for verification outputs
        validated_data[f'{split}_images'] = images
        validated_data[f'{split}_labels'] = labels

    # 4. Validation Check: Class Distribution Profiles Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    blood_cell_names = ['Basophil', 'Eosinophil', 'Erythroblast', 'Immature Granulocyte',
                        'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

    for i, split in enumerate(splits):
        split_labels = validated_data[f'{split}_labels']
        classes, counts = np.unique(split_labels, return_counts=True)
        display_names = [blood_cell_names[c] for c in classes]

        axes[i].bar(display_names, counts, color='#2c3e50', edgecolor='black', alpha=0.8)
        axes[i].set_title(f"{split.upper()} Set Stratification")
        axes[i].set_ylabel("Sample Counts")
        axes[i].set_xticklabels(display_names, rotation=45, ha='right')
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Task 1.2: Structural Class Distribution Profiles", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("[SUCCESS] All multi-point data validation requirements cleared flawlessly.")
    return validated_data

# Execute checking suite
dataset_arrays = run_dataset_validation()

#----------------------------NEXT STEP--------------------------
# PREPROCESSING PIPELINE AND DATA LOADERS


# ==============================================================================
# Name: Zarar Bin Akram
# SRN: 303-221057
# File: Task 1.3 & 1.4 - Preprocessing & DataLoader Pipeline
# Description: Implements image resizing, min-max normalization, sets
#              deterministic seeds, and wraps data splits into PyTorch loaders.
# ==============================================================================

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

def set_deterministic_environment(seed=42):
    """
    Purpose:
        Ensures perfect reproducibility across runs by locking all potential
        sources of randomness in NumPy and PyTorch backends.
    Inputs:
        seed (int): The numerical seed value. Default is 42.
    Outputs:
        None
    Assumptions:
        Assumes the backend execution engine supports PyTorch deterministic flags.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Deterministic execution seed locked at: {seed}")

class BloodMNISTDataset(Dataset):
    """
    Purpose:
        Custom PyTorch Dataset class to bridge raw NumPy arrays into a tensor
        stream while dynamically applying target image preprocessing behaviors.
    Inputs:
        images (numpy.ndarray): Array containing raw images.
        labels (numpy.ndarray): Array containing ground truth target labels.
        transform (torchvision.transforms.Compose): Image manipulation pipeline.
    Outputs:
        Instance of torch.utils.data.Dataset wrapper.
    Assumptions:
        Assumes image channel ordering is standard RGB/Grayscale compatible with PIL.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Extract single sample components
        img = self.images[idx]
        label = self.labels[idx]

        # Convert NumPy array to PIL Image to enable standard torchvision transforms
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        # Convert label integer to torch tensor long format
        label = torch.tensor(label, dtype=torch.long)
        return img, label

def generate_pipeline_loaders(data_dict, batch_size=64, seed=42):
    """
    Purpose:
        Implements Task 1.3 preprocessing pipeline (Resize + Normalization)
        and Task 1.4 deterministic data structuring into PyTorch DataLoaders.
    Inputs:
        data_dict (dict): Dictionary holding validated split arrays from Task 1.2.
        batch_size (int): Training batch size. Defaults to 64.
        seed (int): Reproducibility tracker seed. Defaults to 42.
    Outputs:
        train_loader (DataLoader): PyTorch training iterator.
        val_loader (DataLoader): PyTorch validation monitoring iterator.
        test_loader (DataLoader): PyTorch final testing iterator.
    Assumptions:
        Images are resized to 224x224 to offer full out-of-the-box support for
        pretrained models (ResNet, VGG, MobileNet) in later tasks.
    """
    # Task 1.3: Define Preprocessing & Cleaning Pipeline Transformations
    # Standardize scale to 224x224 and perform min-max scaling/tensor conversion
    preprocessing_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Map ranges directly to [-1, 1]
    ])

    # Wrap underlying numpy structures into our Dataset instances
    train_dataset = BloodMNISTDataset(data_dict['train_images'], data_dict['train_labels'], transform=preprocessing_transforms)
    val_dataset = BloodMNISTDataset(data_dict['val_images'], data_dict['val_labels'], transform=preprocessing_transforms)
    test_dataset = BloodMNISTDataset(data_dict['test_images'], data_dict['test_labels'], transform=preprocessing_transforms)

    # Task 1.4: Establish Deterministic Loader Generation Seeds
    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    # Generate PyTorch DataLoader Streams
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        worker_init_fn=seed_worker, generator=g, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n" + "="*60 + "\n[SUCCESS] DATALOADER GENERATION COMPLETE\n" + "="*60)
    print(f"  -> Total Training Mini-Batches   : {len(train_loader)}")
    print(f"  -> Total Validation Mini-Batches : {len(val_loader)}")
    print(f"  -> Total Testing Mini-Batches    : {len(test_loader)}")

    return train_loader, val_loader, test_loader

# 1. Initialize random seeds for reproducibility (Task 1.4)
set_deterministic_environment(seed=42)

# 2. Extract DataLoaders using the global 'dataset_arrays' created in Task 1.2
train_loader, val_loader, test_loader = generate_pipeline_loaders(dataset_arrays, batch_size=64)