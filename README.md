# 🧠 Advanced Deepfake Detection Using Hybrid Diffusion-Based Techniques

This repository contains the implementation of a deepfake detection system that combines a hybrid diffusion-based GAN for generating realistic fake images, Vision Transformer (ViT) for powerful feature extraction, and traditional machine learning classifiers (SVM, Random Forest, Logistic Regression) for high-accuracy classification.


## 🧩 Key Components

### 📷 Dataset Collection & Preprocessing
- **Real Images**: Sourced from [FlickrFacesHQ](https://github.com/NVlabs/ffhq-dataset).
- **Fake Images**: Generated using Stable Diffusion.
- **Preprocessing**: Resizing, normalization, and augmentation for better generalization.

### 🌀 Deepfake Generation
- Trained a **hybrid diffusion-based GAN** for 50 epochs.
- Achieved **Generator Loss = 2.0882**, **Discriminator Loss = 0.3253**, ensuring stable adversarial training.

### 🔍 Feature Extraction
- Used **Vision Transformer (ViT)** to extract hierarchical global features from input images.

### 🧮 Classification
- Trained three ML classifiers on ViT features:
  - **Support Vector Machine (SVM)**: `99.61%`
  - **Random Forest**: `98.92%`
  - **Logistic Regression**: `99.46%`
- Tested additional CNN-based models:
  - **InceptionResNetV2**: `46.0%`
  - **EfficientNetB3**: `49.0%`
  - **ConvNeXt-Tiny**: `47.0%`

### 📊 Performance Evaluation
- Evaluation metrics: `Accuracy`, `Precision`, `Recall`, `F1-Score`, `Confusion Matrix`
- ViT + ML pipeline outperformed all other baselines
- Loss curves validate the stability and quality of fake image generation


## 📈 Results Summary

| Model                | Accuracy (%) |
|---------------------|--------------|
| SVM (ViT)           | 99.61        |
| Random Forest (ViT) | 98.92        |
| Logistic Regression | 99.46        |
| InceptionResNetV2   | 46.00        |
| EfficientNetB3      | 49.00        |
| ConvNeXt-Tiny       | 47.00        |



