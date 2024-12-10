# MNIST Handwritten Digits Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Implemented Methods](#implemented-methods)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Parameter Tuning](#parameter-tuning)

---

## Introduction
This project performs multi-class classification on the MNIST Handwritten Digits dataset using various machine learning and deep learning classifiers. The goal is to classify each image into one of ten classes (0-9) and compare their performances using k-fold cross-validation, parameter tuning, and evaluation metrics.

---

## Dataset
The **MNIST dataset** is a collection of 70,000 grayscale images of handwritten digits, each of size `28 x 28`. It includes:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

Each image represents a single digit (0-9), with pixel values normalized to the range `[0, 1]`.

---

## Implemented Methods
1. **Decision Tree**
2. **Random Forest**
3. **Naïve Bayes Classifier**
4. **K-Nearest Neighbors (KNN)**
5. **Neural Network Classifier**

### Implementation Details
Each classifier is trained and evaluated using k-fold cross-validation, and hyperparameters are tuned using grid search or other parameter tuning techniques.

---

## Evaluation Metrics
The classifiers are evaluated using the following metrics:
1. **Accuracy**: The ratio of correctly predicted instances to total instances.
2. **Precision, Recall, and F1-Score**: For class-wise evaluation.

---

## Parameter Tuning
We employ **grid search** and **cross-validation** to fine-tune hyperparameters for each model:
1. **Decision Tree**:
   - `max_depth`
   - `min_samples_split`
2. **Random Forest**:
   - `n_estimators`
   - `max_features`
   - `max_depth`
3. **Naïve Bayes**:
   - No tunable parameters (Gaussian NB used).
4. **KNN**:
   - `n_neighbors`
   - `weights`
   - `metric`
5. **Neural Network**:
   - Number of layers and neurons
   - Activation functions
   - Optimizer and learning rate
   - Batch size and epochs

---