# Machine Learning Task 1

Classification on MNIST handwritten digits.

# Task 1: 🔢 MNIST Digit Classification

A machine learning model to classify handwritten digits (0–9) using the MNIST dataset.

---

## 📌 Dataset

This task focuses on building a classification model to recognize handwritten digits from images. The MNIST dataset contains **70,000 grayscale images** of digits, each **28×28 pixels**, flattened into **784 features**.

**Dataset Features:**

* Each sample is an image of a handwritten digit.
* Each image is represented as **784 numerical features** corresponding to pixel intensity values (0–255).
* No categorical features; all features are numeric.

**Target Variable:**

* Digit label (0–9)
* Represents the actual digit depicted in the image.
* This is a **multiclass classification problem** with 10 classes.

---

## 📂 Dataset Loading

The MNIST dataset is loaded using `scikit-learn`’s `fetch_openml`. Students can use the following code to load the dataset:

```python
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist["data"], mnist["target"].astype(int)
```

---

## 🔁 Workflow Overview

* Load the MNIST dataset
* Split into training and testing sets
* Train the model
* Evaluate using accuracy and classification report
