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

#----------------------------NEXT STEP--------------------------
# DEFINING THE 3, 4 AND 5 LAYER CNN ARCHITECTURE

%%writefile models/custom_cnn.py
# ==============================================================================
# Name: Zarar Bin Akram
# SRN: 303-221057
# File: models/custom_cnn.py
# Task: 2.1 - Custom Deep CNN Architectures (3, 4, and 5 layers)
# ==============================================================================

import torch
import torch.nn as nn

class CustomCNN3Layer(nn.Module):
    """
    Purpose:
        Builds a 3-layer deep Custom Convolutional Neural Network.
    Inputs:
        num_classes (int): Total number of target output categories. Default is 8.
    Outputs:
        torch.Tensor: Logits tensor of shape (batch_size, num_classes).
    Assumptions:
        Assumes input images have dimensions of 3 channels x 224 pixels x 224 pixels.
    """
    def __init__(self, num_classes=8):
        super(CustomCNN3Layer, self).__init__()

        # Block 1: Conv -> BatchNorm -> Activation -> MaxPool -> Dropout
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        ) # Output spatial shape: 112 x 112

        # Block 2: Conv -> BatchNorm -> Activation -> MaxPool -> Dropout
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        ) # Output spatial shape: 56 x 56

        # Block 3: Conv -> BatchNorm -> Activation -> MaxPool -> Dropout
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        ) # Output spatial shape: 28 x 28

        # Fully Connected Classifier Head Block
        # Flat features = 128 filters * 28 * 28 spatial pixels = 100,352 neurons
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


class CustomCNN4Layer(nn.Module):
    """
    Purpose:
        Builds a 4-layer deep Custom Convolutional Neural Network.
    Inputs:
        num_classes (int): Total number of target output categories. Default is 8.
    Outputs:
        torch.Tensor: Logits tensor of shape (batch_size, num_classes).
    Assumptions:
        Assumes input images have dimensions of 3 channels x 224 pixels x 224 pixels.
    """
    def __init__(self, num_classes=8):
        super(CustomCNN4Layer, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        ) # 112 x 112

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        ) # 56 x 56

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        ) # 28 x 28

        # Added Convolutional Block Layer 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        ) # Output spatial shape: 14 x 14

        # Classifier Head Block
        # Flat features = 256 filters * 14 * 14 spatial pixels = 50,176 neurons
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


class CustomCNN5Layer(nn.Module):
    """
    Purpose:
        Builds a 5-layer deep Custom Convolutional Neural Network.
    Inputs:
        num_classes (int): Total number of target output categories. Default is 8.
    Outputs:
        torch.Tensor: Logits tensor of shape (batch_size, num_classes).
    Assumptions:
        Assumes input images have dimensions of 3 channels x 224 pixels x 224 pixels.
    """
    def __init__(self, num_classes=8):
        super(CustomCNN5Layer, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        ) # 112 x 112

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        ) # 56 x 56

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        ) # 28 x 28

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        ) # 14 x 14

        # Added Convolutional Block Layer 5
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        ) # Output spatial shape: 7 x 7

        # Classifier Head Block
        # Flat features = 512 filters * 7 * 7 spatial pixels = 25,088 neurons
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.classifier(x)
        return x
    
    #----------------------------NEXT STEP--------------------------
    # ==============================================================================
# Name: Zarar Bin Akram
# SRN: 303-221057
# File: Task 2.2 - Self-Contained GPU Training Execution
# Description: Fully self-contained initialization and training execution
#              to bypass memory loss from Colab runtime resets.
# ==============================================================================

import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# 1. Ensure Local Directories Exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 2. Redownload Dataset if Colab completely wiped the disk during runtime switch
file_path = "data/raw/bloodmnist.npz"
if not os.path.exists(file_path):
    print("[INFO] Dataset missing from disk reset. Redownloading...")
    source_url = "https://zenodo.org/record/6496656/files/bloodmnist.npz?download=1"
    urllib.request.urlretrieve(source_url, file_path)
print("[INFO] Raw data file verified.")

# 3. Set Deterministic Seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 4. Reconstruct Dataset and DataLoaders
class BloodMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

print("[INFO] Loading arrays into memory...")
raw_data = np.load(file_path)
preprocessing_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_loader = DataLoader(BloodMNISTDataset(raw_data['train_images'], raw_data['train_labels'].flatten(), transform=preprocessing_transforms), batch_size=64, shuffle=True)
val_loader = DataLoader(BloodMNISTDataset(raw_data['val_images'], raw_data['val_labels'].flatten(), transform=preprocessing_transforms), batch_size=64, shuffle=False)
test_loader = DataLoader(BloodMNISTDataset(raw_data['test_images'], raw_data['test_labels'].flatten(), transform=preprocessing_transforms), batch_size=64, shuffle=False)

# 5. Define the Hyper-Optimized Architectures (with Global Average Pooling for speed)
class CustomCNN3Layer(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomCNN3Layer, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.3))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.classifier(self.global_pool(self.block3(self.block2(self.block1(x)))))

class CustomCNN4Layer(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomCNN4Layer, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.3))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.3))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.classifier(self.global_pool(self.block4(self.block3(self.block2(self.block1(x))))))

class CustomCNN5Layer(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomCNN5Layer, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.3))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.3))
        self.block5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.4))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        return self.classifier(self.global_pool(self.block5(self.block4(self.block3(self.block2(self.block1(x)))))))

# 6. Initialize Models on Active Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training target environment: {device}")

cnn3 = CustomCNN3Layer(num_classes=8).to(device)
cnn4 = CustomCNN4Layer(num_classes=8).to(device)
cnn5 = CustomCNN5Layer(num_classes=8).to(device)

# 7. Training and Validation Engine Block
def train_and_validate_model(model, model_name, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- Training Engine: {model_name} ---")
    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Validation
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['train_acc'].append((correct_train / total_train) * 100)
        history['val_loss'].append(running_val_loss / len(val_loader.dataset))
        history['val_acc'].append((correct_val / total_val) * 100)

        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {history['train_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.2f}%")
    return history

# Run training loops sequentially
history_3layer = train_and_validate_model(cnn3, "Custom_3_Layer_CNN", train_loader, val_loader, epochs=5)
history_4layer = train_and_validate_model(cnn4, "Custom_4_Layer_CNN", train_loader, val_loader, epochs=5)
history_5layer = train_and_validate_model(cnn5, "Custom_5_Layer_CNN", train_loader, val_loader, epochs=5)
print("\n[SUCCESS] All custom architectures trained successfully on GPU.")