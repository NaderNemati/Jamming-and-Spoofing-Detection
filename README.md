# GNSS Jamming and Spoofing Detection based on ML

This repository provides a complete simulation, preprocessing, and detection pipeline for **GNSS signal spoofing and jamming** using both **XGBoost** and **Deep Learning** models. GNSS interference scenarios related to maritime environments are included along with tools for dataset generation, exploratory data analysis (EDA), training classifiers, and evaluating detection performance.

---

## 📌 Project Objectives

- ✅ Simulate realistic GNSS signal behaviors under normal, jamming, and spoofing conditions.
- ✅ Generate large synthetic datasets for supervised machine learning.
- ✅ Explore and visualize signal characteristics to understand spoofing/jamming effects.
- ✅ Train and evaluate both:
  - 🌲 XGBoost model (feature-based)
  - 🤖 Deep Learning model (vision-style / time series-based)
- ✅ Achieve high-accuracy detection of GNSS interference.

---

## 📁 Repository Structure

GNSS-Detection/
├── data_simulate.py # Simulates GNSS signals & saves dataset
├── run_xgboost.py # XGBoost training & evaluation script
├── run_deep_learning.py # Deep Learning model training (optional: CNN/LSTM/MLP)
├── EDA_ORG.ipynb # Step-by-step Exploratory Data Analysis
├── gnss_dataset.csv # Flattened CSV for ML model input
├── gnss_dataset.npy # Raw simulated array format
├── models/ # (Optional) saved models
└── README.md # Documentation file


---

## ⚙️ Requirements

Install the required Python packages via:

```bash
pip install -r requirements.txt
```

🚀 How to Use
1. Simulate the Dataset

Run the following script to generate GNSS samples and save them in CSV and .npy format:

This creates a balanced dataset with:

    20,000 samples each for normal, jamming, and spoofing

    3 channels per signal: C/N₀, Doppler shift, signal power

2. Run Exploratory Data Analysis (EDA)

Use the notebook to visualize and understand the dataset:

```bash
jupyter notebook EDA_ORG.ipynb
```

Includes:

    Class distribution

    C/N₀ patterns per class

    Correlation heatmaps

    PCA and t-SNE visualization

    Outlier detection (IQR, Isolation Forest)


3. Train and Evaluate Models
➤ XGBoost Model

```bash
python3 run_xgboost.py
```

Expected Output:

    Accuracy: ~99.5%

    Confusion Matrix and Classification Report

➤ Deep Learning Model

```bash
python3 run_deep_learning.py
```

Expected Output:

    Accuracy: ~98.1%

    Good generalization but slightly lower performance on Jamming class

📊 Example Results

XGBoost:

    Accuracy: 99.49%

    Jamming Recall: 1.00

    Spoofing Recall: 0.99

Deep Learning:

    Accuracy: 98.09%

    Jamming Recall: 0.97

    Spoofing Recall: 0.99


🔍 Detection Strategy

    Spoofing Detection: Based on abnormally high and consistent C/N₀

    Jamming Detection: Detected by sudden drop in C/N₀ across satellites

    Outlier Detection (optional): Supports unsupervised Isolation Forest fallback


🛰️ Dataset Description

Each GNSS sample contains:

    8 satellites × 30 timesteps × 3 features

    Features: C/N₀ (dB-Hz), Doppler (Hz), Signal Power (dBm)

    Labels: 0 = Normal, 1 = Spoofing, 2 = Jamming


📌 References

GNSS simulation design inspired by [AliCMU](https://github.com/alicmu2024/GNSS-Jamming-Detection-and-Classification-using-Machine-Learning-Deep-Learning-and-Computer-Vision) GNSS Jamming Detection repo
    Feature engineering and models are structured for research reproducibility
