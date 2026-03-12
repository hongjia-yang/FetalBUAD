# FetalBUAD: Simple Baselines for Fetal Brain Unsupervised Anomaly Detection

**FetalBUAD** is a simple but strong baseline for fetal brain MRI 3D unsupervised anomaly detection (UAD). It repurposes voxel-level brain age prediction for dense lesion localization and establishes an internal biological gestational age reference, robustly decoupling dating errors from structural deviations, achieving state-of-the-art performance in detecting and localizing fetal brain anomalies.


---

## 📂 Directory Structure

```
FetalBUAD/
├── data_processing/
│   └── registration.py          # Preprocessing: Registration to 38-week atlas
├── model_training/
│   ├── Train.py                 # Main training script
│   ├── dataset.py               # Custom Dataset class with data preprocessing
│   ├── net.py                   # Joint Multi-task Network Architecture
│   ├── dice.py                  # Dice Loss implementation
│   └── .gitignore
├── inference/
│   ├── 1_biascorrection.py      # Calculate linear bias correction parameters
│   ├── 2_generate_voxel_level_brain_age_gap_map.py  # MC Dropout inference & Gap Map generation
│   └── 3_generate_uncertainty_weight_gap_map_and_anomaly_score.py # Calculate final Anomaly Score
└── README.md
```

## 🛠️ Prerequisites

The code is implemented in Python using PyTorch. The main dependencies are:

- Python 3.9.19
- PyTorch 2.2.2
- Nibabel
- Numpy
- Pandas
- Scikit-learn
- SimpleITK
- antspyx
- bm4d

Install dependencies via:
```bash
conda create -n FetalBUAD python=3.9 -y
conda activate FetalBUAD
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Data Preprocessing
FetalBUAD requires all input volumes to be spatially aligned to a 38-week CRL atlas volume.
The CRL atlas: "Gholipour, A. et al. A normative spatiotemporal MRI atlas of the fetal brain for automatic segmentation and analysis of early brain growth. Sci Rep 7, 476 (2017)".
Run the registration script to register your raw T2w fetal brain volumes (reconstructed via SVR) using affine transformation.

```bash
python data_processing/registration.py
```
*   **Input:** Raw T2w volumes.
*   **Output:** Spatially normalized volumes.

### 2. Model Training
Train the Joint Multi-task Network. This network simultaneously predicts:
1.  Global Brain Age
2.  Voxel-level Brain Age Volume
3.  Brain Segmentation

```bash
python model_training/Train.py
```
*   **Configuration:** You can adjust hyperparameters (batch size, learning rate, etc.) inside `Train.py`.

### 3. Inference & Anomaly Detection
The inference process is divided into three sequential steps to ensure robust detection.

#### Step 3.1: Bias Correction
Calculate the linear bias correction parameters ($\alpha, \beta$) using the **in-distribution (normal) validation set**. This mitigates the regression dilution effect.

```bash
python inference/1_biascorrection.py
```
*   **Output:** Saves `corr_param.npy` containing slope and intercept for global age correction.

#### Step 3.2: Generate Voxel-level Gap Maps
Perform inference using **Monte Carlo (MC) Dropout** ($T$ passes). This script computes the predictive mean and epistemic uncertainty for each voxel and generates the raw Brain Age Gap Map (BAGM).

```bash
python inference/2_generate_voxel_level_brain_age_gap_map.py
```
*   **Process:** 
    1.  Loads the trained model.
    2.  Applies MC Dropout to get mean and std of voxel-level age.
    3.  Corrects the global age using parameters from Step 3.1.
    4.  Calculates `Gap = Predicted_Voxel_Age - Corrected_Global_Age`.

#### Step 3.3: Anomaly Scoring
Calculate the final **Uncertainty-Weighted Deviation Index (UWDI)** for anomaly detection and generate the final visualization maps.

```bash
python inference/3_generate_uncertainty_weight_gap_map_and_anomaly_score.py
```
*   **Method:** This script amplifies the gap map using the uncertainty weights and calculates the dispersion (std) of the positive deviations to derive the subject-level anomaly score.

---

