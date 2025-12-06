import torch
import torch.nn as nn
import torch.optim as optim
import os  
from src import config, utils, data_setup, model_setup, train_model

def main():
    print(f"Starting project on {config.DEVICE}")
    
    # 1. Prepare data (Split folders)
    utils.create_dataset_structure()
    
    # 2. Load DataLoaders
    dataloaders, class_names = data_setup.get_dataloaders()
    print(f"Classes: {class_names}")
    
    # 3. Prepare the model
    # Note: We use the function inside 'model_setup.py'
    model = model_setup.initialize_model(len(class_names))
    model = model.to(config.DEVICE)
    
    # 4. Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
    
    # 5. Start Training
    print(f"Training for {config.NUM_EPOCHS} epochs...")
    model, history = train_model.train_model(model, dataloaders, criterion, optimizer)
    
    # 6. Save results
    # --- SAFETY CHECK: Create output folder if it doesn't exist ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save plots
    utils.plot_results(history)
    
    # Save model
    
    save_path = os.path.join(config.OUTPUT_DIR, f"{config.MODEL_NAME}.pth")
    torch.save(model.state_dict(), save_path)
    
    print(f"Done! Model saved to {save_path}")
    print("Check the 'outputs' folder for graphs and the .pth file.")
    

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # saving the model and history with specific names
    model_filename = f"model_{config.MODEL_NAME}.pth"
    history_filename = f"history_{config.MODEL_NAME}.pkl"
    
    # Save the model
    save_path = os.path.join(config.OUTPUT_DIR, model_filename)
    torch.save(model.state_dict(), save_path)
    print(f" Model saved: {save_path}")

    # Save the history (
    utils.save_history(history, history_filename)
    
    # Plot and save training results
    utils.plot_results(history)

if __name__ == '__main__':
    main()