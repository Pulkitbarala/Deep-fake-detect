# 🚀 Deepfake Detection with Hybrid CNN + Transformer

A complete **Deepfake Detection System** that combines **ResNet50 (CNN)** for spatial feature extraction and a **Transformer Encoder** for global context understanding.

This repository includes:
- 🧠 Full **training pipeline (Jupyter Notebook + scripts)**
- 🌐 **Ready-to-use web application** (no training required)
- ⚡ Hybrid deep learning architecture for improved accuracy

---

## 📌 Project Overview

Deepfake media is becoming increasingly realistic and dangerous. This project tackles the problem using a **hybrid architecture**:

- **CNN (ResNet50)** → captures local facial features  
- **Transformer Encoder** → models global dependencies  

👉 The system predicts whether an image is:
- ✅ **Real**
- ❌ **Fake**

---

## 🗂️ Project Structure

```
Deep-fake-detect/
│
├── webapp/
│   ├── static/
│   ├── templates/
│   ├── app.py
│   └── trained model files
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── app.py
│
├── data_raw/
├── data_processed/
├── training.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
git clone <your-repo-link>
cd Deep-fake-detect
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Web App

```bash
cd webapp
streamlit run app.py
```

Open: http://127.0.0.1:5000

---



### Train Model

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

---

## 🧠 Model Architecture

- ResNet50 (CNN backbone)
- Transformer Encoder (2 layers, 8 heads)
- Binary Classification (Sigmoid)

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

Pulkit
