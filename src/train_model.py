import torch
import time
import copy
from tqdm import tqdm
from src import config

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    """
    Main training function.
    """
    since = time.time()

    # Save history for final plots
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a Training and a Validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Enable dropout and batchnorm
            else:
                model.eval()   # Disable dropout (evaluation mode)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data (use tqdm for progress bar)
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}", leave=False):
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                # Zero the gradients
                optimizer.zero_grad()

                # --- FORWARD ---
                # Calculate gradients only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # --- BACKWARD + OPTIMIZE (Only in train) ---
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Save in history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy of the model if it's the best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # print("  New record! Model saved.")

        print()

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model, history