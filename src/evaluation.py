import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from . import config  # Import relative config

def evaluate_test_set(model, dataloader, class_names, device, output_dir):
    """
    Evaluates the model on the internal Test Set.
    """
    print(f"\n{'='*20} INTERNAL TEST EVALUATION {'='*20}")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 1. Prediction Loop
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 2. Save CSV
    model_out_dir = os.path.join(output_dir, f'{config.MODEL_NAME}_outputs')
    os.makedirs(model_out_dir, exist_ok=True)
    
    str_labels = [class_names[i] for i in all_labels]
    str_preds = [class_names[i] for i in all_preds]
    
    results_df = pd.DataFrame({'True': str_labels, 'Predicted': str_preds, 'Correct': [t==p for t,p in zip(str_labels, str_preds)]})
    csv_path = os.path.join(model_out_dir, f"{config.MODEL_NAME}_test_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # 3. Report & Confusion Matrix
    print("\nCLASSIFICATION REPORT")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {config.MODEL_NAME}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    
    cm_path = os.path.join(model_out_dir, f"{config.MODEL_NAME}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Matrix saved: {cm_path}")
    # plt.show() # In a pure Python script, show() often blocks execution. Better to just save.


def evaluate_external(model, external_loader, internal_class_names, device, output_dir):
    """
    Evaluates the model on an External Dataset.
    """
    print(f"\n{'='*20} EXTERNAL DATASET EVALUATION {'='*20}")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(external_loader, desc="External Test"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Name Mapping
    ext_class_names = external_loader.dataset.classes
    str_true = [ext_class_names[i] for i in all_labels]
    str_pred = [internal_class_names[i] for i in all_preds]
    
    # CSV and Matrix
    model_out_dir = os.path.join(output_dir, f'{config.MODEL_NAME}_outputs')
    
    # CSV
    results_df = pd.DataFrame({'True': str_true, 'Predicted': str_pred})
    csv_path = os.path.join(model_out_dir, f"{config.MODEL_NAME}_EXTERNAL_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"External CSV saved: {csv_path}")
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    labels_union = sorted(list(set(str_true) | set(str_pred)))
    cm = confusion_matrix(str_true, str_pred, labels=labels_union)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels_union, yticklabels=labels_union)
    plt.title(f'External Matrix - {config.MODEL_NAME}')
    
    cm_path = os.path.join(model_out_dir, f"{config.MODEL_NAME}_EXTERNAL_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"External Matrix saved: {cm_path}")