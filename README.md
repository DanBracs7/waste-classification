# ♻️ Waste Classification with Deep Learning

A Deep Learning project to classify waste items into **9 categories** to assist in automatic recycling processes. This project utilizes a multi-model approach, using a **ResNet18** neural network pre-trained on ImageNet and fine-tuned on a waste dataset, a custom (from scratch) **ResNet18**, and a VGG-style model, to maximize learning on the task.

This project was developed for a university exam: **Computer Science - Fundamentals of Data Science**.

## 📊 Project Overview

  - **Goal**: Classify images of waste (Plastic, Glass, Paper, etc.) to automate sorting.
  - **Models**: ResNet18 (Transfer Learning), Custom ResNet18, and VGG-Style model.
  - **Dataset**: [Waste Classification Data](https://www.kaggle.com/datasets/adithyachalla/waste-classification) (4k+ images).
  - **Framework**: PyTorch.

## 👥 Team Members & Roles

  * **Daniele Bracoloni** (Lead Engineer): Pipeline architecture, Model implementation, Code structure.
  * **Marco Foresi** (Data Scientist): Hyperparameter tuning, Ablation studies, Experimental design.
  * **Alice Paglia** (Data Analyst): Real-world testing, Error analysis, Business logic mapping.

-----

## 📄 Technical Report: Project Methodology & Contribution

### 1\. Objective and Personal Contribution

The project aims to develop and compare different Deep Learning architectures for the automatic classification of waste into 9 distinct categories. My main contribution was not limited to the application of existing models but focused on the **design and implementation from scratch** of complex neural architectures to analyze their behavior compared to the state of the art.

### 2\. Data Analysis and Preprocessing Pipeline

I began with an Exploratory Data Analysis (EDA) which revealed a native image resolution of $524 \times 524$ pixels. To balance computational load with the need to preserve structural details (e.g., cardboard texture vs. plastic), I implemented a preprocessing pipeline that:

  * Performs **downsampling to $224 \times 224$**, standardizing the input for all architectures.
  * Applies **statistical normalization** based on ImageNet parameters.
  * Introduces **Data Augmentation** techniques (rotations, flips) in the training set to mitigate overfitting, given the limited number of samples per class (\~25k total).
  * Manages the stratified division of the dataset into **Train (70%)**, **Validation (15%)**, and **Test (15%)**, ensuring a statistically robust evaluation.

### 3\. Architecture Development (Project Core)

To conduct a rigorous comparative study, I wrote the code for three distinct architectures, implementing modular logic (Factory Pattern) that allows swapping models with a single parameter:

  * **A. Sequential Baseline (VGG-Style):** I built a classic CNN composed of 4 sequential convolutional blocks.  This model serves as a baseline to quantify the performance of a network without residual connections.

  * **B. Custom ResNet18 (Manual Implementation):** This represents the **core of my research work**. Instead of importing ResNet from external libraries, I implemented the architecture line-by-line:

      * I coded the **Residual Block (`BasicBlock`)**, manually handling the *skip connections*  and tensor dimension adaptation.
      * I assembled the 4 stages of ResNet18 following the standard `[2, 2, 2, 2]` topology.
      * **Original Contribution:** I modified the standard architecture by inserting a **Dropout layer ($p=0.5$)** before the final classifier. This design choice was dictated by the need to further regularize the model, adapting it to a smaller dataset compared to the one ResNet was originally conceived for (ImageNet).

  * **C. Transfer Learning (The Benchmark):** I used a ResNet18 pre-trained on ImageNet as a comparison term ("Champion"), performing fine-tuning only on the final *Fully Connected* layer.

### 4\. Training and Evaluation Methodology

I developed a robust training engine (`train_model`) that includes:

  * **Automatic Checkpointing:** The system saves model weights only when validation accuracy improves, preventing the saving of overfitting models in the final epochs.
  * **History Logging:** Saving of Loss and Accuracy metrics for each epoch, allowing the generation of comparative plots.

The final evaluation is performed on the **Test Set** (unseen data) using advanced metrics:

  * **Classification Report:** Analysis of Precision, Recall, and F1-Score for every single waste class.
  * **Confusion Matrix:** To visually diagnose frequent errors (e.g., confusion between Glass and Plastic) and validate the effectiveness of the different architectures.

-----

## 🚀 Quick Start

### 1\. Clone the repository

```bash
git clone [https://github.com/DanBracs7/waste-classification.git](https://github.com/DanBracs7/waste-classification.git)
cd waste-classification
```

### 2\. Set up the environment

We recommend using a virtual environment (venv).

**Windows**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

### 4\. Prepare the Data

1.  Download the dataset (e.g., from Kaggle).
2.  Extract it into a folder named `data/raw` inside this project directory.
3.  **Crucial:** Ensure the folder structure looks like this:

<!-- end list -->

```text
data/raw/
├── Glass/
│   ├── image01.jpg
│   └── ...
├── Plastic/
│   ├── image02.jpg
│   └── ...
└── Paper/
    └── ...
```

-----

## ⚙️ Configuration

Before running the project, you can modify `src/config.py` to change the experiment settings:

  * **`MODEL_NAME`**: Choose between `'custom_resnet'`, `'custom_vgg'`, or `'pretrained'` (ResNet18).
  * **`NUM_EPOCHS`**: Set the training duration.
  * **`BATCH_SIZE`** & **`LEARNING_RATE`**: Tune hyperparameters.

-----

## ▶️ How to Run

The `main.py` file is the orchestrator for the entire pipeline. You can choose which phase to run by commenting/uncommenting the functions at the bottom of `main.py`.

### 1\. Train the Model

To start the full training pipeline (Data Split -\> Training -\> Saving):

```bash
python main.py
```

  * **What it does:**
      * Automatically splits raw data into Train (70%), Val (15%), and Test (15%).
      * Trains the selected model (defined in config).
      * Saves the best weights (`.pth`), training history (`.pkl`), and plots (`.png`) in `outputs/<MODEL_NAME>_outputs/`.

### 2\. Evaluate on Internal Test Set

To test the model on the unseen 15% of the data (created during the split):

1.  Open `main.py`.
2.  Ensure `run_testing()` is uncommented.
3.  Run:

<!-- end list -->

```bash
python main.py
```

  * **Output:** Generates a Classification Report and a Confusion Matrix (Blue) to visualize performance on the test set.

### 3\. Evaluate on External Dataset

To test the model on completely new images (e.g., downloaded from the web):

1.  Organize your new images in a folder with the same class structure (e.g., `my_new_data/Glass`, `my_new_data/Plastic`).
2.  Open `main.py`.
3.  Uncomment `run_external_test('./path/to/my_new_data')` and provide the correct path.
4.  Run:

<!-- end list -->

```bash
python main.py
```

  * **Output:** Generates a CSV file with predictions and an External Confusion Matrix (Orange).

If you don't feel confident using python files, i created a python notebook, that has the some functions and the same structure as the python files. (Notify that i used it to work with other co workers and some comments may be in italian to make the work easier)