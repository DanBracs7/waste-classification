import os
import torch
import torch.nn as nn
# --- PERCORSI ---
# Se siamo su Kaggle o in Locale
if os.path.exists('/kaggle/input'):
    DATA_ROOT = '/kaggle/input/waste-classification'
    OUTPUT_DIR = '/kaggle/working/output'

    
else:
    DATA_ROOT = './data/raw' # Metti qui il tuo dataset locale
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
    else:    
        OUTPUT_DIR = './outputs'  #

PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, 'processed_data')


# --- URAZIONE ESPERIMENTO ---
# Scegli qui quale modello allenare!
# Opzioni disponibili (basate sul tuo codice):
# 1. 'custom_resnet'     -> La tua ResNet fatta a mano (From Scratch)
# 2. 'pretrained_resnet' -> La ResNet18 di ImageNet (Transfer Learning)
# 3. 'custom_vgg'        -> La tua VGG fatta a mano (From Scratch)
 
# <--- CAMBIA QUESTO PER TESTARE DIVERSI MODELLI --->
# MODEL_NAME = 'custom_resnet' # DECOMENTA QUI PER TESTARE IL MODELLO RESNET CUSTOM PERSONALIZZATO E COMMENTA GLI ALTRI
MODEL_NAME = 'custom_vgg' # DECOMENTA QUI PER TESTARE IL MODELLO VGG PERSONALIZZATO E COMMENTA GLI ALTRI
# MODEL_NAME = 'pretrained_resnet' # DECOMENTA QUI PER TESTARE IL MODELLO RESNET PRETRAINATO O E COMMENTA GLI ALTRI

#model version = da ggiungere per nuove versioni del modello

# --- IPERPARAMETRI (STATISTICI, TOCCATE QUI!) ---
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# on colab use2 or 4 workers depending on the instance
NUM_WORKERS = 0 # Numero di workers per il DataLoader (0 = main thread) 
NUM_CLASSES = 9 # Numero di classi nel dataset
CRITERION = nn.CrossEntropyLoss() # Funzione di perdita 


# --- PARAMETRI DATASET ---
IMG_SIZE = 224
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
SEED = 42 # Per riproducibilità