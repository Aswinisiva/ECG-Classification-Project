# ECG Image Classification Project

This project involves analyzing and classifying ECG (Electrocardiogram) images using Convolutional Neural Networks (CNNs). It is structured into two key scripts: one for data exploration and preprocessing, and another for model training and evaluation.

## Dataset

- **Source:** [Kaggle ECG Image Data](https://www.kaggle.com/datasets/erhmrai/ecg-image-data/data)
- The dataset contains images of ECG signals categorized into multiple classes.

---

## Project Structure

1. **`ecg_eda.py`**
    - Purpose: Perform exploratory data analysis (EDA) and preprocess the ECG dataset.
    - Key Features:
        - Data visualization (e.g., class distributions and sample images).
        - Data normalization and balancing.
        - Generates insights into the dataset for effective model training.

2. **`ECG_model_training.py`**
    - Purpose: Train and evaluate deep learning models on the ECG dataset.
    - Models Implemented:
        - Simple CNN
        - VGG16 (pretrained)
        - ResNet50 (pretrained)
    - Key Features:
        - Support for data augmentation.
        - Implementation of focal loss for handling class imbalance.
        - Visualization of training metrics and confusion matrices.

---

## Key Features

- **Data Handling:**
    - The scripts download and preprocess the dataset.
    - Balances classes by limiting maximum samples per class.
    - Supports data augmentation techniques for better generalization.

- **Model Training:**
    - Simple CNN model is built from scratch.
    - VGG16 and ResNet50 are fine-tuned using transfer learning.
    - Training uses callbacks like Early Stopping and ReduceLROnPlateau for optimized training.

- **Metrics:**
    - Accuracy, Precision, Recall, and AUC metrics are calculated.
    - Training and validation curves are plotted for comparison.
    - Confusion matrices provide a visual representation of predictions.

---

## Dependencies

Ensure the following Python libraries are installed:
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Kaggle API
- Pillow

Install dependencies with:
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn kaggle pillow
