import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import from your src modules
from src import config, utils, data_setup, model_setup, train_model, evaluation

def run_training():
    """Executes the complete training loop."""
    print(f"STARTING TRAINING: {config.MODEL_NAME}")
    
    # 1. Load Data
    dataloaders, class_names = data_setup.get_dataloaders()
    
    # 2. Initialize Model
    model = model_setup.initialize_model(len(class_names), model_name=config.MODEL_NAME)
    model = model.to(config.DEVICE)
    
    # 3. Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
    
    try:
        model_trained, history = train_model.train_model(model, dataloaders, criterion, optimizer, config.NUM_EPOCHS)
        
        # 4. Saving Results
        out_dir = os.path.join(config.OUTPUT_DIR, f'{config.MODEL_NAME}_outputs')
        os.makedirs(out_dir, exist_ok=True)
        
        # Save Weights
        torch.save(model_trained.state_dict(), os.path.join(out_dir, f"{config.MODEL_NAME}_best.pth"))
        
        # Save History
        with open(os.path.join(out_dir, f"{config.MODEL_NAME}_history.pkl"), 'wb') as f:
            pickle.dump(history, f)
            
        # Quick Plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1,2,1); plt.plot(history['train_acc']); plt.plot(history['val_acc']); plt.title('Acc')
        plt.subplot(1,2,2); plt.plot(history['train_loss']); plt.plot(history['val_loss']); plt.title('Loss')
        plt.savefig(os.path.join(out_dir, f"{config.MODEL_NAME}_plot.png"))
        print("Training completed and saved.")
        
    except KeyboardInterrupt:
        print("Training interrupted.")

def run_testing():
    """Evaluates the model on the internal Test Set."""
    print(f"STARTING TEST: {config.MODEL_NAME}")
    
    # 1. Load Data
    dataloaders, class_names = data_setup.get_dataloaders()
    if 'test' not in dataloaders: return

    # 2. Load Model and Weights
    model = model_setup.initialize_model(len(class_names), model_name=config.MODEL_NAME)
    weights_path = os.path.join(config.OUTPUT_DIR, f'{config.MODEL_NAME}_outputs', f"{config.MODEL_NAME}_best.pth")
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
        print(f"⚖️ Weights loaded: {weights_path}")
    else:
        print("⚠️ Weights not found, impossible to test.")
        return
    
    model = model.to(config.DEVICE)

    # 3. Call the evaluation function (in src/evaluation.py)
    evaluation.evaluate_test_set(model, dataloaders['test'], class_names, config.DEVICE, config.OUTPUT_DIR)

def run_external_test(external_path):
    """Evaluates on an external folder."""
    print(f"STARTING EXTERNAL TEST: {external_path}")
    
    if not os.path.exists(external_path):
        print("Path not found.")
        return

    # 1. External Data Setup
    # Note: Logically copying validation transforms from config/data_setup
    ext_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ext_dataset = datasets.ImageFolder(external_path, transform=ext_transform)
    ext_loader = DataLoader(ext_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Load Model and Internal Classes (to map predictions)
    _, class_names = data_setup.get_dataloaders() # We only need the original class names
    
    model = model_setup.initialize_model(len(class_names), model_name=config.MODEL_NAME)
    weights_path = os.path.join(config.OUTPUT_DIR, f'{config.MODEL_NAME}_outputs', f"{config.MODEL_NAME}_best.pth")
    model.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    
    # 3. Evaluation
    evaluation.evaluate_external(model, ext_loader, class_names, config.DEVICE, config.OUTPUT_DIR)


if __name__ == "__main__":
    # --- Uncomment what you want to do ---
    
    # 1. Training
    run_training()
    
    # 2. Testing (Internal Dataset)
    run_testing()
    
    # 3. External Testing (Optional)
    # run_external_test('./path/to/new_data')