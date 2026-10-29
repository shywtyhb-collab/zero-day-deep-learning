# 🛡️ Zero-Day Network Intrusion Detection System (Autoencoder)

An Unsupervised Deep Learning framework designed to detect **Zero-Day network attacks** and novel behavioral anomalies without relying on pre-existing signatures.

---

## 📌 Overview
Traditional Intrusion Detection Systems (IDS) rely heavily on known signatures, leaving systems vulnerable to Zero-Day exploits. This project addresses this critical gap by training an **Autoencoder Neural Network** strictly on baseline benign network traffic to learn normal flow representations. Unknown attacks are captured when the network reconstruction error exceeds a statistically defined threshold.

---

## 🏗️ Architecture & Methodology
- **Encoder:** Compresses high-dimensional network flow metrics into a compressed latent bottleneck.
- **Bottleneck:** Captures core benign network transmission characteristics.
- **Decoder:** Attempts to reconstruct input features from the latent space.
- **Anomaly Scoring:** Evaluates Mean Squared Error (MSE) reconstruction loss:
  $$\text{Reconstruction Error} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2$$
- **Thresholding Strategy:** Statistical anomaly boundary set as:
  $$\text{Threshold} = \mu_{\text{benign}} + 3\sigma_{\text{benign}}$$

---

## 📊 Key Evaluation Results
- **Dataset Paradigm:** Trained on normal traffic only; evaluated against unseen zero-day attacks.
- **Detection Accuracy:** ~99%
- **Zero-Day Attack Recall:** 100% (1,500 / 1,500 detected)
- **Mean Normal Reconstruction Error:** ~0.018
- **Mean Zero-Day Reconstruction Error:** ~136.72

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment
\`\`\`bash
git clone https://github.com/shywtyhb-collab/zero-day-deep-learning.git
cd zero-day-deep-learning
python3 -m venv venv
source venv/bin/activate
pip install torch pandas numpy scikit-learn streamlit
\`\`\`

### 2. Prepare Data & Train Model
\`\`\`bash
python3 prepare_data.py
python3 train_model.py
python3 evaluate_zeroday.py
\`\`\`

### 3. Launch Interactive SOC Dashboard
\`\`\`bash
streamlit run app.py
\`\`\`
