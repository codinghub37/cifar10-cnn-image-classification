#  CIFAR-10 Image Classification Using Convolutional Neural Networks (CNN)

##  Project Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The objective is to train a deep learning model capable of recognizing and classifying images into one of ten predefined categories.

The project covers the complete deep learning workflow, including:

- Data Loading
- Data Preprocessing
- Data Visualization
- CNN Model Development
- Model Training
- Model Evaluation
- Image Prediction

The implementation is developed using TensorFlow and Keras in Python.

---

#  Objectives

The primary objectives of this project are:

- Understand the fundamentals of Convolutional Neural Networks.
- Learn image preprocessing techniques.
- Train a CNN model on a real-world image dataset.
- Evaluate model performance using test data.
- Perform image classification on unseen samples.
- Gain practical experience with TensorFlow and Keras.

---

#  About CIFAR-10 Dataset

CIFAR-10 is one of the most widely used benchmark datasets in Computer Vision.

### Dataset Statistics

| Property | Value |
|-----------|---------|
| Total Images | 60,000 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Image Size | 32 × 32 |
| Channels | RGB (3 Channels) |
| Classes | 10 |

### Classes

The dataset contains images from the following categories:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

---

#  Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab / Jupyter Notebook

---

#  Project Workflow

## Step 1: Import Required Libraries

Required libraries are imported for:

- Deep Learning
- Data Processing
- Data Visualization

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

---

## Step 2: Load CIFAR-10 Dataset

The dataset is loaded directly from TensorFlow/Keras.

```python
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

---

## Step 3: Data Preprocessing

### Normalize Images

Pixel values are converted from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

### Why Normalization?

- Faster convergence
- Better model performance
- Stable training process

---

## Step 4: Data Visualization

Sample images are displayed to understand the dataset structure and class distribution.

Visualization helps verify:

- Correct image loading
- Label assignments
- Dataset quality

---

## Step 5: CNN Model Development

A Convolutional Neural Network is developed using the Keras Sequential API.

### CNN Architecture

```text
Input Layer (32×32×3)
        ↓
Conv2D Layer
        ↓
ReLU Activation
        ↓
MaxPooling2D
        ↓
Conv2D Layer
        ↓
ReLU Activation
        ↓
MaxPooling2D
        ↓
Flatten Layer
        ↓
Dense Layer
        ↓
Output Layer (10 Classes)
```

### Layer Functions

#### Convolution Layer

Extracts important features such as:

- Edges
- Shapes
- Textures
- Patterns

#### Max Pooling Layer

Reduces dimensionality while preserving useful information.

#### Flatten Layer

Converts feature maps into a one-dimensional vector.

#### Dense Layer

Performs classification using extracted features.

#### Output Layer

Predicts probabilities for all 10 classes.

---

#  Model Compilation

The CNN model is compiled using:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

#  Model Training

The model is trained using the training dataset.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10
)
```

During training:

- Forward propagation
- Loss calculation
- Backpropagation
- Weight updates

are performed automatically.

---

# Model Evaluation

The trained model is evaluated on test data.

```python
test_loss, test_acc = model.evaluate(
    X_test,
    y_test
)
```

Evaluation helps determine:

- Accuracy
- Generalization capability
- Performance on unseen data

---

#  Image Prediction

The trained model can classify unseen images.

```python
prediction = model.predict(X_test)
```

The model predicts one of the following classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

#  Results

The CNN model successfully learns image features and performs image classification across all ten categories.

### Example

Input Image:

```text
Airplane
```

Predicted Class:

```text
Airplane
```

---

#  Project Structure

```text
cifar10-cnn-image-classification/
│
├── cnn_cifar10_dataset.ipynb
├── README.md
├── requirements.txt
└── images/
```

---

#  How to Run

### Clone Repository

```bash
git clone https://github.com/yourusername/cifar10-cnn-image-classification.git
```

### Install Dependencies

```bash
pip install tensorflow
pip install numpy
pip install matplotlib
```

### Open Notebook

```bash
jupyter notebook cnn_cifar10_dataset.ipynb
```

---

#  Real-World Applications

CNN-based image classification can be used in:

- Autonomous Vehicles
- Medical Imaging
- Face Recognition
- Smart Surveillance
- Security Systems
- Industrial Automation
- Object Detection Systems

---

#  Learning Outcomes

This project helps understand:

- Deep Learning Fundamentals
- Convolutional Neural Networks
- Image Classification
- Data Preprocessing
- Model Training
- Model Evaluation
- TensorFlow and Keras

---

#  Future Improvements

Possible future enhancements include:

- Data Augmentation
- Transfer Learning
- ResNet Implementation
- VGG16 Implementation
- Hyperparameter Tuning
- Higher Accuracy Models
- Web Application Deployment

---

#  Conclusion

This project successfully demonstrates image classification using a Convolutional Neural Network trained on the CIFAR-10 dataset. The implementation provides practical experience with deep learning, computer vision, and image recognition techniques using TensorFlow and Keras.

---

#  Author

Developed as a Deep Learning and Computer Vision project for learning and research purposes.
