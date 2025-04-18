# American Sign Language Gesture Classification 🤟

This project is the final assignment for **EE541 – A Computational Introduction to Deep Learning** at USC.

We built and compared several models for American Sign Language (ASL) gesture classification, including:

- 📌 **Support Vector Machine (SVM)**  
- 📌 **Baseline Convolutional Neural Network (CNN)**  
- 📌 **Pre-trained ResNet34**

---

## 📂 Dataset

- **Source**: [Kaggle ASL dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)  
- **Total Samples**: 87,000 images  
- **Classes**: 29 (A–Z, SPACE, DELETE, NOTHING)  
- **Image Size**: 200x200 → Resized to 224x224  
- **Split**: 80% training / 10% validation / 10% test  

We applied standard preprocessing:
- Random horizontal flip and rotation (data augmentation)
- Normalization (mean 0, std 1)
- Class-balanced sampling

---

## 🧠 Models & Techniques

### 1️⃣ Support Vector Machine (SVM)
- Input: Grayscale pixel values
- Hyperparameter tuning: C, kernel, gamma (random search + cross-validation)
- **Best accuracy**: 79%

### 2️⃣ Baseline CNN (implemented from scratch with PyTorch)
- 1 or 2 convolutional layers
- BatchNorm, ReLU, MaxPooling, Fully Connected
- **Best accuracy**: 90.35% (2-layer CNN)

### 3️⃣ ResNet34 (Pretrained, fine-tuned)
- Modified final classification layer (output: 29 classes)
- Gradual unfreezing strategy
- Hyperparameter tuning (batch size, learning rate)
- **Best accuracy**: 🥇 97.59%

---

## 📈 Results

| Model      | Accuracy  |
|------------|-----------|
| SVM        | 79.00%    |
| CNN (2-layer) | 90.35% |
| ResNet34   | **97.59%** |

---

## 🛠️ Tools & Frameworks

- Python 3.8+
- PyTorch
- NumPy, Matplotlib
- Scikit-learn

---

## 📜 Report

Read the full report [here](./Final%20Project.pdf)

---

## 👨‍💻 Authors

- Jerry Chen – [LinkedIn](https://www.linkedin.com/in/jerry-chenyw)  
- Ping-Hsi Hsu
