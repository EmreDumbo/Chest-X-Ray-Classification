# Lung Disease Classification from Chest X-Rays

## Introduction

Lung diseases, such as pneumothorax, atelectasis, and cardiomegaly, are significant global health challenges. Early and accurate diagnosis of these conditions is critical to improving patient outcomes. However, traditional methods of diagnosis, which rely on expert radiological interpretation, can be time-consuming and resource-intensive, especially in areas with limited healthcare access.

This project leverages the power of deep learning to classify lung diseases using chest X-rays. By automating the diagnostic process, this project aims to assist healthcare professionals in providing faster and more accurate diagnoses.

---

## Project Overview

This project uses the NIH Chest X-rays dataset and focuses on developing a deep learning model to classify grayscale chest X-ray images into six disease categories.

###  Key Features:
- **Dataset**: A subset of the NIH Chest X-rays dataset, preprocessed to include 19,561 images across six classes.
- **Deep Learning Architecture**: A fine-tuned ResNet50 architecture optimized for grayscale medical image classification.
- **Data Preprocessing**: Grayscale normalization, augmentation, and class weight balancing.


---

## Technologies Used
- Python 3.10
- TensorFlow and Keras
- NVIDIA GeForce RTX 3060 GPU for training
- Libraries: NumPy, Matplotlib, and scikit-learn

---

##  Getting Started

###  Prerequisites

Ensure you have the following installed:
- Python 3.10
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

###  Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/EmreDumbo/Lung-Disease-Classification.git
   cd Lung-Disease-Classification
   ```

2. **Prepare the Dataset**:
   - Download the NIH Chest X-rays dataset from [this link](https://www.kaggle.com/datasets/nih-chest-xrays/data).
   - Filter and preprocess the dataset to include six target classes.
   - Organize the dataset into `train`, `val`, and `test` directories.

3. **Set Up the Environment**:
   ```bash
   python -m venv lung_disease_env
   source lung_disease_env/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the Model**:
   ```bash
   python train.py
   ```

6. **Evaluate the Model**:
   ```bash
   python evaluate.py
   ```

---

## Results

The model achieved the following performance metrics:
- **Accuracy**: 55%
- **ROC-AUC (Macro)**: 0.87
- **Class-wise Highlights**:
  - Cardiomegaly: Sensitivity 0.85, Specificity 0.96
  - Pneumothorax: Sensitivity 0.77, Specificity 0.86

While the overall accuracy is modest, the high AUC scores for critical conditions demonstrate the model’s potential for real-world diagnostic applications. Challenges, such as dataset imbalance and underrepresented classes, remain and highlight areas for further research.

---

##  Contributing

Contributions are welcome! To contribute:
- Fork the repository.
- Create a feature branch.
- Submit a pull request with detailed comments.

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

##  Acknowledgments

- The creators of the NIH Chest X-rays dataset.
- Open-source frameworks like TensorFlow and Keras.
