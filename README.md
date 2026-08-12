# Happy-Sad Emotion Detection

A Deep Learning project that detects **Happy** and **Sad** facial emotions using a webcam in real time.

## Project Overview

This project uses facial images to train a Convolutional Neural Network (CNN) model. The trained model is then used to recognize whether a detected facial expression is **Happy** or **Sad**.

## Technologies Used

* Python
* OpenCV
* TensorFlow
* Keras
* DeepFace
* NumPy
* CNN (Convolutional Neural Network)

## Project Workflow

```text
Webcam
   ↓
Face Detection
   ↓
Image Processing
   ↓
Deep Learning Model
   ↓
Emotion Classification
   ↓
Happy / Sad
```

## Project Files

* `capture.py` – Captures Happy and Sad images using the webcam.
* `prepare_dataset.py` – Detects faces and prepares the dataset.
* `train_model.py` – Trains the CNN model using the prepared dataset.
* `live_emotion.py` – Performs real-time emotion detection using the webcam.
* `happy_sad_model.keras` – Trained Deep Learning model.
* `datausage` – Project-related data/resource file.

## Dataset

The dataset contains two emotion classes:

* Happy
* Sad

Face images are collected using a webcam and processed before training.

## Model Training

The CNN model is trained using TensorFlow and Keras.

The model learns facial features from the Happy and Sad datasets and classifies the detected expression into one of the two categories.

## How to Run

### 1. Install required packages

```bash
pip install tensorflow opencv-python numpy pillow deepface
```

### 2. Prepare the dataset

```bash
python prepare_dataset.py
```

### 3. Train the model

```bash
python train_model.py
```

### 4. Run live emotion detection

```bash
python live_emotion.py
```

A webcam window will open and the system will detect the facial emotion in real time.

## Features

* Real-time webcam detection
* Face detection
* Happy/Sad emotion classification
* Deep Learning based prediction
* CNN model training

## Future Improvements

* Add more emotions such as Angry, Fear, Surprise and Neutral.
* Increase the size and diversity of the dataset.
* Improve model accuracy using transfer learning.
* Build a web or desktop application for easier usage.

## Author

**Akilarasan P**

B.Tech Artificial Intelligence and Data Science
