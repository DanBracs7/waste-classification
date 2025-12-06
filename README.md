#  EcoSort: Waste Classification with Deep Learning

A Deep Learning project to classify waste items into **9 categories** to assist in automatic recycling processes. This project utilizes a **ResNet18** neural network pre-trained on ImageNet and fine-tuned on a waste dataset. This project was made for an university exam of Computer Science - Fundamentals of Data Science

##  Project Overview
- **Goal**: Classify images of waste (Plastic, Glass, Paper, etc.) to automate sorting.
- **Model**: ResNet18 (Transfer Learning).
- **Dataset**: [Waste Classification Data](https://www.kaggle.com/datasets/adithyachalla/waste-classification) (25k+ images).
- **Framework**: PyTorch.

## 👥 Team Members & Roles
* **Daniele Bracoloni** (Lead Engineer): Pipeline architecture, Model implementation, Code structure.
* **n** (Data Scientist): Hyperparameter tuning, Ablation studies, Experimental design.
* **n** (Data Analyst): Real-world testing, Error analysis, Business logic mapping.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone [https://github.com/TUO_USERNAME/waste-classification.git](https://github.com/DanBracs7/waste-classification.git)
cd waste-classification

2. Set up the environment

We recommend using a virtual environment (venv).
# Windowsgit
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate


3. Install dependencies

pip install -r requirements.txt


4. Download Data

    Download the dataset from Kaggle.

    Extract it into a folder named data/raw inside this project.

        Structure should look like: data/raw/DATASET/TRAIN/...


How to Run
Train the Model

To start the training pipeline (Data Split -> Training -> Saving), simply run:
python main.py



This script will:

    Automatically split data into Train/Val/Test (if not already done).

    Download the ResNet18 model.

    Train the model for the configured number of epochs.

    Save the best model (final_model.pth) and plots in the outputs/ folder



Configuration 

You don't need to change the code logic! To change Hyperparameters (Batch size, Learning Rate, Epochs), open and edit:
# Example inside src/config.py
BATCH_SIZE = 32      # Try 16 or 64
NUM_EPOCHS = 10      # Increase for better results
LEARNING_RATE = 0.001