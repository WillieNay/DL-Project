# Handwritten Digit Recognition with TensorFlow

This project implements a neural network model using TensorFlow to classify handwritten digits from the MNIST dataset. The model is trained on normalized MNIST images and can be used to classify new digit images.

---

## Features

1. **Training a Neural Network**:
   - Model structure includes:
     - Input layer: Flattens 28x28 grayscale images.
     - Two Dense layers: One with 128 ReLU neurons and another with 10 Softmax neurons for classification.
   - Trained using `adam` optimizer and `sparse_categorical_crossentropy` loss.

2. **Digit Classification**:
   - The model predicts the digit from grayscale images.
   - Supports uploading custom handwritten digit images for classification.

3. **Pre-trained Model**:
   - Saves the trained model as `handwrittendigit.keras`.
   - Loads the model for reuse and predictions.

---

## Directory Structure
**train_model.py**: Script for training the model on MNIST.
**classify_digit.py**: Script for classifying a custom digit image.
**handwrittendigit.keras**: Pre-trained model saved after training.
**sample_data/**: Directory containing custom test images.

---

## Output

**Training Results**:
Displays training loss and accuracy for each epoch.
Outputs test loss and accuracy.

**Digit Classification**:
Resizes and normalizes the input image.

**Predicts the digit and prints the result**:
The digit is probably a <digit>

---

## Key Functions

**Model Building**:
Uses the Sequential API with the following layers:
Flatten (input layer).
Dense layer with 128 neurons and ReLU activation.
Dense layer with 10 neurons and Softmax activation.

**Digit Classification**:
Converts the input image to grayscale.
Resizes to 28x28 pixels and normalizes to [0, 1].
Uses the pre-trained model to predict the digit.

**Image Processing**:
Resizes input images for display and prediction.

