import os
import torch

# --- PERCORSI ---
# Se siamo su Kaggle o in Locale
if os.path.exists('/kaggle/input'):
    DATA_ROOT = '/kaggle/input/waste-classification'
    OUTPUT_DIR = '/kaggle/working/output'
else:
    DATA_ROOT = './data/raw' # Metti qui il tuo dataset locale
    OUTPUT_DIR = './outputs'

PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, 'processed_data')

MODEL_NAME = 'resnet18_finetuned_pre_trained'


# --- IPERPARAMETRI (STATISTICI, TOCCATE QUI!) ---
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# on colab use2 or 4 workers depending on the instance
NUM_WORKERS = 0 # Numero di workers per il DataLoader (0 = main thread) 

# --- PARAMETRI DATASET ---
IMG_SIZE = 224
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
SEED = 42 