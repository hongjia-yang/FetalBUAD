# FetalBUAD: Simple Baselines for Fetal Brain Unsupervised Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![MICCAI](https://img.shields.io/badge/Conference-MICCAI-blue.svg)](http://www.miccai.org/)

This repository contains the official PyTorch implementation for the paper: **"Simple Baselines for Fetal Brain Unsupervised Anomaly Detection"**.

**FetalBUAD** is a novel unsupervised anomaly detection (UAD) framework that repurposes voxel-level brain age prediction for dense lesion localization. By establishing an internal biological baseline via uncertainty weighting and bias correction, it robustly decouples dating errors from structural deviations, achieving state-of-the-art performance in detecting and localizing fetal brain anomalies.

---

## 📂 Directory Structure

```
FetalBUAD/
├── data_processing/
│   └── registration.py          # Preprocessing: Registration to 38-week atlas & Normalization
├── model_training/
│   ├── Train.py                 # Main training script
│   ├── dataset.py               # Custom Dataset class with Target Randomization
│   ├── unet.py                  # Joint Multi-task Network Architecture
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

- Python 3.8+
- PyTorch 1.10+
- Nibabel
- Numpy
- Pandas
- Scikit-learn
- SimpleITK

Install dependencies via:
```bash
pip install torch nibabel numpy pandas scikit-learn simpleitk
```

---

## 🚀 Usage

### 1. Data Preprocessing
FetalBUAD requires all input volumes to be spatially aligned to a canonical space.
Run the registration script to register your raw T2w fetal brain volumes (reconstructed via SVR) to the 38-week fetal brain atlas.

```bash
python data_processing/registration.py
```
*   **Input:** Raw T2w volumes & Brain masks.
*   **Output:** Spatially normalized volumes (128x160x128) and segmentation maps aligned to the 38-week template.

### 2. Model Training
Train the Joint Multi-task Network. This network simultaneously predicts:
1.  Global Brain Age
2.  Voxel-level Brain Age Volume
3.  Brain Segmentation

```bash
python model_training/Train.py
```
*   **Configuration:** You can adjust hyperparameters (batch size, learning rate, etc.) inside `Train.py`.
*   **Target Randomization:** The `dataset.py` implements the stochastic label perturbation strategy to prevent mode collapse during voxel-wise regression training.

### 3. Inference & Anomaly Detection
The inference process is divided into three sequential steps to ensure robust detection.

#### Step 3.1: Bias Correction
Calculate the linear bias correction parameters ($\alpha, \beta$) using the **in-distribution (normal) validation set**. This mitigates the regression dilution effect.

```bash
python inference/1_biascorrection.py
```
*   **Output:** Saves `corr_param.npy` containing slope and intercept for global age correction.

#### Step 3.2: Generate Voxel-level Gap Maps
Perform inference using **Monte Carlo (MC) Dropout** ($T=36$ passes). This script computes the predictive mean and epistemic uncertainty for each voxel and generates the raw Brain Age Gap Map (BAGM).

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

## 📊 Results

FetalBUAD achieves state-of-the-art performance on both internal large-scale datasets and external benchmarks (FeTA dataset).

| Method | AUROC | AUPR | Inference Speed |
| :--- | :---: | :---: | :---: |
| IterMask3D | 0.905 | 0.779 | Slow (Iterative) |
| **FetalBUAD (Ours)** | **0.905** | **0.764** | **Fast (Single-pass)** |

*For detailed results on lesion localization and generalization, please refer to the paper.*

---

## 📝 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{FetalBUAD2026,
  title={Simple Baselines for Fetal Brain Unsupervised Anomaly Detection},
  author={Anonymized Authors},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year={2026}
}
```

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
