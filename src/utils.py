import os
import shutil
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src import config 
import pickle


def create_dataset_structure():
    """
    Reads raw data and creates Train/Val/Test structure
    """
    # 1. Check if dataset already exists and is not empty
    if os.path.exists(config.PROCESSED_DATA_DIR):
        if len(os.listdir(config.PROCESSED_DATA_DIR)) > 0:
            print(f"Dataset already in {config.PROCESSED_DATA_DIR}, skipping the split.")
            return
        else:
            print("Output folder exists but is empty. Proceeding...")

    print(f"Starting dataset split from: {config.DATA_ROOT}...")
    
    # 2. Detect classes automatically
    found_classes = set()
    for path, dirs, files in os.walk(config.DATA_ROOT):
        # Check for images (case insensitive)
        if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
            parent = os.path.basename(path)
            
            # Handle numeric prefixes (e.g., "1-Plastic" -> "Plastic")
            if len(parent) > 2 and parent[1] == '-' and parent[0].isdigit():
                parent = parent[2:] 
            
            # Filter out system folders
            if parent not in ['TRAIN', 'TEST', 'DATASET', 'images', 'raw', 'data']:
                found_classes.add(parent)
    
    classes = sorted(list(found_classes))
    print(f"Classes Found: {classes}")

    if not classes:
        print("ERROR: No classes found! Check DATA_ROOT path.")
        return

    # 3. Perform the split
    total_images_count = 0
    
    for cls in classes:
        print(f"Processing class: '{cls}'...")
        
        # Robust search strategy: check exact name AND original folder names
        # We search recursively for files ending in common extensions (Case Insensitive logic)
        search_patterns = [
            f"{config.DATA_ROOT}/**/{cls}/*.jpg",
            f"{config.DATA_ROOT}/**/{cls}/*.JPG",
            f"{config.DATA_ROOT}/**/{cls}/*.png",
            f"{config.DATA_ROOT}/**/{cls}/*.PNG",
            f"{config.DATA_ROOT}/**/{cls}/*.jpeg"
        ]
        
        # Also search for folders with prefixes (e.g. searching for Plastic inside 1-Plastic)
        # This covers the case where we stripped the prefix in detection but need it for file path
        search_patterns.append(f"{config.DATA_ROOT}/**/?-{cls}/*.jpg") 
        
        files = []
        for pattern in search_patterns:
            found = glob.glob(pattern, recursive=True)
            files.extend(found)
        
        # Remove duplicates
        files = list(set(files))

        if not files:
            print(f"  WARNING: 0 files found for '{cls}'. Skipping...")
            continue

        total_images_count += len(files)

        # Split 70% Train - 15% Val - 15% Test
        train_files, temp = train_test_split(files, test_size=0.3, random_state=SEED)
        val_files, test_files = train_test_split(temp, test_size=0.5, random_state=SEED)

        # Physical copy
        splits = {'train': train_files, 'val': val_files, 'test': test_files}
        
        for split_name, split_files in splits.items():
            dest_dir = os.path.join(config.PROCESSED_DATA_DIR, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for f in split_files:
                try:
                    shutil.copy(f, dest_dir)
                except Exception as e:
                    print(f"Error copying {f}: {e}")
    
    if total_images_count == 0:
        print("ERROR: No images were copied. Please check your data structure.")
    else:
        print(f"Split completed! Total images processed: {total_images_count}")

def plot_results(history):
    """
    Generates and saves Loss and Accuracy plots
    """
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()

    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(config.OUTPUT_DIR, 'training_plot.png')
    plt.savefig(save_path)
    print(f"Plots saved in: {save_path}")


def save_history(history, filename):
    """
    Save the training history to a file using pickle
    """
    # Ensure the output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    filepath = os.path.join(config.OUTPUT_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    print(f"History salvata in: {filepath}")
