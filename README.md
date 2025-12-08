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


📄 Technical Report: Development of CNN Architectures for Waste Classification
1. Objective and Personal Contribution

The project aims to develop and compare different Deep Learning architectures for the automatic classification of waste into 9 distinct categories. My main contribution was not limited to the application of existing models but focused on the design and implementation from scratch of complex neural architectures to analyze their behavior compared to the state of the art.
2. Data Analysis and Preprocessing Pipeline

I began with an Exploratory Data Analysis (EDA) which revealed a native image resolution of 524×524 pixels. To balance computational load with the need to preserve structural details (e.g., cardboard texture vs. plastic), I implemented a preprocessing pipeline that:

    Performs downsampling to 224×224, standardizing the input for all architectures.

    Applies statistical normalization based on ImageNet parameters.

    Introduces Data Augmentation techniques (rotations, flips) in the training set to mitigate overfitting, given the limited number of samples per class (~25k total).

    Manages the stratified division of the dataset into Train (70%), Validation (15%), and Test (15%), ensuring a statistically robust evaluation.

3. Architecture Development (Project Core)

To conduct a rigorous comparative study, I wrote the code for three distinct architectures, implementing modular logic (Factory Pattern) that allows swapping models with a single parameter:

    A. Sequential Baseline (VGG-Style): I built a classic CNN composed of 4 sequential convolutional blocks. This model serves as a baseline to quantify the performance of a network without residual connections.

    B. Custom ResNet18 (Manual Implementation): This represents the core of my research work. Instead of importing ResNet from external libraries, I implemented the architecture line-by-line:

        I coded the Residual Block (BasicBlock), manually handling the skip connections and tensor dimension adaptation.

        I assembled the 4 stages of ResNet18 following the standard [2, 2, 2, 2] topology.

        Original Contribution: I modified the standard architecture by inserting a Dropout layer (p=0.5) before the final classifier. This design choice was dictated by the need to further regularize the model, adapting it to a smaller dataset compared to the one ResNet was originally conceived for (ImageNet).

    C. Transfer Learning (The Benchmark): I used a ResNet18 pre-trained on ImageNet as a comparison term ("Champion"), performing fine-tuning only on the final Fully Connected layer.

4. Training and Evaluation Methodology

I developed a robust training engine (train_model) that includes:

    Automatic Checkpointing: The system saves model weights only when validation accuracy improves, preventing the saving of overfitting models in the final epochs.

    History Logging: Saving of Loss and Accuracy metrics for each epoch, allowing the generation of comparative plots.

The final evaluation is performed on the Test Set (unseen data) using advanced metrics:

    Classification Report: Analysis of Precision, Recall, and F1-Score for every single waste class.

    Confusion Matrix: To visually diagnose frequent errors (e.g., confusion between Glass and Plastic) and validate the effectiveness of the different architectures.