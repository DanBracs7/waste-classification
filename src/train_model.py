import torch
import time
import copy
from tqdm import tqdm
from src import config

def train_model(model, dataloaders, criterion, optimizer):
    """
    Main training loop.
    Args:
        model: The PyTorch model to train
        dataloaders: Dictionary containing 'train' and 'val' dataloaders
        criterion: The loss function (e.g., CrossEntropyLoss)
        optimizer: The optimizer (e.g., SGD or Adam)
    Returns:
        model: The best model (based on validation accuracy)
        history: Dictionary containing loss and accuracy metrics
    """
    
    print("Training starting...")
    start_time = time.time()

    # Initialize best model weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    
    # Dictionary to store training history for plotting
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Loop through epochs
    for epoch in range(config.NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode (enable dropout, batchnorm)
            else:
                model.eval()   # Set model to evaluate mode (freeze specific layers)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data (batches)
            # We use tqdm for a nice progress bar
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase", leave=False):
                # Move data to the configured device (GPU or CPU)
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # --- FORWARD PASS ---
                # Track gradients only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss_value = criterion(outputs, labels)

                    # --- BACKWARD PASS & OPTIMIZE ---
                    # Only perform backpropagation in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # --- STATISTICS ---
                # Multiply by batch size to get total loss for the batch
                running_loss += loss_value.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch metrics
            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            # Store metrics in history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # print("New best model found!")

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_accuracy:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history