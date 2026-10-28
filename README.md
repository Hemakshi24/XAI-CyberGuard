# 🛡️ XAI-CyberGuard
### Explainable AI-Based Cyber Threat Detection System

XAI-CyberGuard is a machine learning-based cybersecurity project designed to classify network traffic and identify potential cyber threats. It combines a Random Forest classifier with Explainable Artificial Intelligence (XAI) techniques to make model predictions more understandable and transparent.

The project uses the CICIDS2017 dataset to train and evaluate a machine learning model. SHAP (SHapley Additive exPlanations) is used to explain which features influence the model's predictions, while an interactive Streamlit dashboard provides a visual interface for exploring the system's outputs.

The primary goal is not only to classify network traffic but also to help users understand **why a particular prediction was made**.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [Machine Learning Model](#-machine-learning-model)
- [Explainable AI with SHAP](#-explainable-ai-with-shap)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Project](#-running-the-project)
- [Experimental Results](#-experimental-results)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Applications](#-applications)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 🔍 Project Overview

Cyber threats such as Denial of Service (DoS), Distributed Denial of Service (DDoS), port scanning, brute-force attacks, and other malicious activities can affect the availability, confidentiality, and integrity of digital systems.

Machine learning can help identify suspicious traffic patterns. However, a model's prediction may be difficult to interpret when it provides only a class label without explaining the factors behind its decision.

XAI-CyberGuard addresses this interpretability challenge by combining machine learning-based traffic classification with SHAP explanations.

The current prototype follows an offline analysis approach: it processes a prepared dataset, predicts traffic categories, generates explanations, and presents relevant outputs through a dashboard.

## ❗ Problem Statement

Traditional machine learning-based threat detection systems may provide predictions without sufficient insight into the reasons behind them. This lack of transparency can make it harder for users and security analysts to understand model decisions and assess their usefulness.

The project explores how Explainable AI can improve the interpretability of a machine learning-based cyber threat classification system.

## 🎯 Project Objectives

- Process and prepare network traffic data for machine learning.
- Train a model to classify benign and potentially malicious traffic.
- Evaluate the model using a held-out test dataset.
- Use SHAP to identify influential features in model predictions.
- Explain individual predictions in a more interpretable form.
- Present model outputs through an interactive Streamlit dashboard.
- Explore the role of explainability in cybersecurity analysis.

## ✨ Key Features

### 1. Dataset Preprocessing
Processes the CICIDS2017 dataset, handles missing values, removes duplicate records, and prepares data for model training.

### 2. Machine Learning-Based Classification
Uses a Random Forest classifier to categorize network traffic into 15 classes represented in the prepared dataset.

### 3. Global Feature Importance
Uses SHAP to examine which input features contribute to model predictions across the analyzed data.

### 4. Individual Prediction Explanation
Generates SHAP explanations for individual predictions to help illustrate which features influenced a particular classification.

### 5. Interactive Dashboard
Uses Streamlit to provide a graphical interface for displaying project outputs and explainability results.

### 6. Modular Python Implementation
Separates preprocessing, model training, explanation generation, and dashboard functionality into different Python scripts.

---

## 🏗️ System Architecture

The current prototype follows this workflow:

```text
             CICIDS2017 Dataset
                      |
                      v
             Data Preprocessing
                      |
                      v
              Dataset Cleaning
          (Missing Values and Duplicates)
                      |
                      v
             Prepared Dataset
                      |
                      v
             Train/Test Split
                      |
                      v
             Random Forest Model
                      |
                      v
              Traffic Prediction
                      |
             +--------+--------+
             |                 |
             v                 v
      Global SHAP         Individual SHAP
      Explanation         Explanation
             |                 |
             +--------+--------+
                      |
                      v
             Streamlit Dashboard
                      |
                      v
           Visualized Model Outputs
```

**Note:** This architecture describes the current offline prototype. It does not indicate that live network traffic is being monitored or that attacks are automatically blocked.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Random Forest | Traffic classification |
| SHAP | Model explainability |
| Matplotlib | Visualization |
| Streamlit | Interactive dashboard |
| CICIDS2017 | Network traffic dataset |
| Joblib | Saving and loading the trained model |

---

## 📊 Dataset

The project uses the **CICIDS2017 dataset**, a publicly available network intrusion detection dataset containing benign traffic and traffic associated with different attack categories.

The dataset provides traffic-flow features that can be used to train machine learning models for classification.

### Attack Categories Represented

The prepared dataset contains 15 labels, including:

- BENIGN
- DoS Hulk
- DoS GoldenEye
- DoS Slowloris
- DoS Slowhttptest
- DDoS
- PortScan
- SSH-Patator
- FTP-Patator
- Bot
- Infiltration
- Heartbleed
- Web Attack – Brute Force
- Web Attack – XSS
- Web Attack – Sql Injection

These labels represent the classes present in the prepared dataset. Their presence does not imply that the model detects every real-world variation of these attacks.

### Dataset Source

The dataset is available through the Canadian Institute for Cybersecurity:

https://www.unb.ca/cic/datasets/ids-2017.html

**Important:** The original dataset is not included in this repository by default. Users should obtain it from the official source and follow the dataset's applicable terms.

---

## 🧹 Data Preprocessing

The project uses Python scripts to inspect, preprocess, and clean the dataset.

The preprocessing workflow includes:

1. Loading the selected CICIDS2017 CSV files.
2. Inspecting dataset dimensions and class distributions.
3. Examining missing values and duplicate records.
4. Preparing the traffic-flow features and target labels.
5. Removing duplicate records during dataset cleaning.
6. Saving the cleaned dataset for model training.

### Recorded Cleaning Results

| Metric | Result |
|---|---:|
| Initial dataset records | 79,080 |
| Records after cleaning | 71,663 |
| Duplicate records removed | 7,417 |
| Missing values after cleaning | 0 |
| Input features | 78 |
| Target classes | 15 |

The cleaned dataset is saved as:

`data/processed/cicids2017_clean.csv`

The exact paths may vary depending on the local project setup.

---

## 🤖 Machine Learning Model

### Random Forest Classifier

XAI-CyberGuard uses a Random Forest classifier to learn patterns in the prepared network traffic data.

Random Forest combines predictions from multiple decision trees to produce a final classification. It is a widely used supervised learning algorithm for classification tasks.

### Training Workflow

1. Load the cleaned dataset.
2. Separate the input features from the target label.
3. Split the dataset into training and testing subsets.
4. Train the Random Forest classifier on the training subset.
5. Evaluate the trained model on the held-out test subset.
6. Save the trained model for later use.

### Why Random Forest?

- Supports multiclass classification.
- Can model nonlinear relationships between features.
- Provides a useful baseline for structured tabular data.
- Works with feature-based explanation techniques such as SHAP.

The model's predictions depend on the quality, representativeness, and preprocessing of the training data.

---

## 🔎 Explainable AI with SHAP

### What Is SHAP?

SHAP stands for **SHapley Additive exPlanations**. It is a model explanation technique based on Shapley values from cooperative game theory.

SHAP estimates how individual input features contribute to a model's prediction relative to a reference or baseline output.

In this project, SHAP is used to explore the reasoning behind the Random Forest model's outputs.

### Global Explanation

Global explanations help identify features that are influential across a collection of predictions.

The project generates feature-importance results using mean absolute SHAP values.

Recorded top features include:

- Destination Port
- Min Segment Size Forward
- Init Win Bytes Backward
- Bwd Packet Length Min
- Bwd Packets/s

These results describe the model's learned patterns in the analyzed dataset. They do not establish that a feature is inherently malicious or that it causes an attack.

### Individual Prediction Explanation

Individual explanations show how features contribute to a specific model prediction.

One recorded example from the project was:

| Field | Recorded Result |
|---|---|
| Actual label | SSH-Patator |
| Predicted label | SSH-Patator |
| Destination Port | 22 |
| Forward Packet Length Max | 640 |

For this example, the SHAP output assigned positive contributions to several features for the predicted class.

An individual explanation helps users inspect a prediction, but it does not independently prove that the traffic is malicious.

### Explanation Output

The individual explanation image is saved at:

`results/individual_prediction_shap.png`

The global feature-importance outputs are saved in the `results` directory.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard built with Streamlit.

The dashboard provides a visual interface for presenting the current prototype's functionality and outputs.

Depending on the implemented dashboard components, these may include:

- Dataset preprocessing status.
- Model information and evaluation results.
- Traffic classification outputs.
- Global feature-importance results.
- Individual SHAP explanations.

The dashboard is intended to make the model and its explanations easier to explore without requiring users to inspect every output directly in the terminal.

### Current Dashboard Status

The Streamlit application has been run successfully in the local development environment.

The current implementation is based on offline dataset analysis. Live packet capture, real-time traffic monitoring, and automatic threat blocking are not currently claimed as implemented features.

---

## 📁 Project Structure

The current project contains the following main files and directories:

```text
XAI-CyberGuard/
│
├── app.py
├── preprocess.py
├── clean_dataset.py
├── inspect_dataset.py
├── inspect_labels.py
├── inspect_processed.py
├── train_model.py
├── explain_model.py
├── explain_prediction.py
│
├── data/
│   ├── cicids2017/
│   └── processed/
│       ├── cicids2017_sample.csv
│       └── cicids2017_clean.csv
│
├── models/
│   └── random_forest_model.joblib
│
├── results/
│   └── individual_prediction_shap.png
│
├── requirements.txt
└── README.md
```

This is a guide to the project's main files. Your actual repository may differ depending on which files you include in GitHub.

**Security and repository note:** Do not commit your virtual environment, large raw datasets, private information, or unnecessary generated files. Check your `.gitignore` before pushing the project.

---

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible Python version.
- pip.
- Git (optional, for cloning the repository).
- A compatible CICIDS2017 dataset copy.

### Step 1: Clone the Repository

Replace the repository URL below if your GitHub repository uses a different address.

```bash
git clone https://github.com/Hemakshi24/XAI-CyberGuard.git
cd XAI-CyberGuard
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the Python executable inside `.venv` directly or follow the appropriate Windows Python environment instructions.

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure that `requirements.txt` contains the dependencies actually used by the project, such as:

```text
pandas
numpy
scikit-learn
shap
matplotlib
streamlit
joblib
```

These are the main dependencies; compatible versions may need to be specified for reproducible installation.

### Step 4: Prepare the Dataset

Download CICIDS2017 from the official dataset source and place the relevant CSV files in the directory expected by `preprocess.py`.

Check the input paths inside the Python scripts before running them. The directory structure used by your local copy may differ from the example.

---

## ▶️ Running the Project

Run the following commands from the project root directory, with the virtual environment activated.

### 1. Preprocess the Dataset

```bash
python preprocess.py
```

This processes the configured input CSV files and prepares the dataset for subsequent steps.

### 2. Clean the Dataset

```bash
python clean_dataset.py
```

This removes duplicate records and saves the cleaned dataset.

### 3. Train the Model

```bash
python train_model.py
```

This trains the Random Forest classifier, evaluates it on the held-out test set, and saves the trained model.

### 4. Generate Global SHAP Explanations

```bash
python explain_model.py
```

This generates feature-importance explanations for the model.

### 5. Generate an Individual Prediction Explanation

```bash
python explain_prediction.py
```

This produces an explanation for the example prediction configured in the script.

### 6. Launch the Dashboard

```bash
python -m streamlit run app.py
```

After the application starts, open the local URL shown in the terminal, usually:

`http://localhost:8501`

**Execution note:** The commands above describe the intended workflow based on the current project files. Ensure that the scripts' dataset paths, model paths, and dependencies match your local setup.

---

## 📈 Experimental Results

The current prototype has produced the following recorded results.

| Evaluation Item | Result |
|---|---|
| Cleaned dataset size | 71,663 records |
| Number of features | 78 |
| Number of classes | 15 |
| Training records | 57,330 |
| Testing records | 14,333 |
| Initial held-out test accuracy | 98.51% |
| Global SHAP feature-importance output | Generated |
| Individual SHAP explanation | Generated |
| Streamlit dashboard | Runs locally |

### Classification Evaluation

The recorded classification report shows differences in performance across attack categories.

For example, some classes have lower precision, recall, or F1-scores than others. The Web Attack classes, in particular, show weaker results in the current evaluation.

This means that overall accuracy alone is not sufficient to describe the model's effectiveness.

Further investigation is needed into class imbalance, the number of examples available for some classes, and performance on independent data.

### Interpreting the Accuracy

The recorded 98.51% accuracy applies to the current held-out test split. It should not be interpreted as a guarantee of 98.51% accuracy on live network traffic or unseen real-world attacks.

Additional validation is required before drawing conclusions about practical deployment.

---

## ⚠️ Limitations

The current prototype has several limitations:

- **Offline analysis:** It works with prepared datasets rather than live network traffic.
- **Dataset dependence:** Its predictions depend on the patterns represented in CICIDS2017.
- **Class imbalance:** Some attack categories contain substantially fewer records than others.
- **Uneven class performance:** The recorded classification report shows weaker performance for some classes.
- **Generalization:** Performance on independent datasets and real-world traffic has not yet been established.
- **Explainability:** SHAP explains the model's behavior but does not prove that a prediction is correct or that a feature causes an attack.
- **Deployment readiness:** The prototype is not a production-ready intrusion prevention system and does not automatically block traffic.

These limitations
