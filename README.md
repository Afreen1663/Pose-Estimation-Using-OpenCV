# Pose Estimation using OpenCV and Mediapipe

This project performs human pose estimation using OpenCV and Mediapipe. It identifies keypoints of the human body in both static images and real-time webcam feeds, allowing for applications such as fitness tracking, gesture recognition, and movement analysis.

<img width="391" alt="Image" src="https://github.com/user-attachments/assets/ffd6e943-6af6-461e-909c-7b1ace0b1268" />

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Pose Detection Process](#pose-detection-process)
- [Implementation](#implementation)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

Pose estimation is a computer vision technique used to detect and track keypoints on the human body. This project leverages Mediapipe’s Pose model, which uses deep learning to detect and track body landmarks in real time. The objective is to create a system that can identify various human poses accurately in both images and live webcam feeds.

## Features

- Detects and tracks human body keypoints
- Supports static images and real-time webcam processing
- Uses Mediapipe’s pre-trained pose estimation model
- Visualizes keypoints and skeletal structure
- Fast and lightweight inference

## Technologies Used

- **Python** – Programming language
- **OpenCV** – Image processing and real-time webcam support
- **Mediapipe** – Pre-trained model for human pose estimation
- **NumPy** – Efficient mathematical operations
- **Matplotlib** – Visualization of keypoints and pose structures

## Dataset

- This project processes static images stored in a local folder.
- The webcam is used for real-time pose estimation.
- Mediapipe’s Pose model is used as the core detection algorithm.

## Pose Detection Process

1. Load an image or capture a frame from the webcam.
2. Convert the image to RGB format (Mediapipe processes RGB images).
3. Use the Mediapipe Pose model to detect keypoints.
4. Draw skeletal connections and keypoints on the image.
5. Display the processed image with detected landmarks.

## Implementation

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Afreen1663/Pose-Estimation-Using-OpenCV.git
   ```
2. Install dependencies:
   ```sh
   pip install opencv-python mediapipe numpy matplotlib
   ```
3. Run the Jupyter Notebook:
   ```sh
   jupyter notebook "pose_estimation.ipynb"
   ```

### Sample Code
```python
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Pose Estimation', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Evaluation

The performance of the pose estimation system is evaluated based on:

- **Detection Accuracy** – How well the keypoints align with actual human body parts.
- **Processing Speed** – Frame rate achieved for real-time applications.
- **Robustness** – Ability to handle different lighting conditions and occlusions.

## Results

- Successfully detects and visualizes body keypoints.
- Real-time estimation provides smooth tracking of movements.
- Accurate keypoint detection even in varying lighting conditions.

## Future Improvements

- Implement action recognition based on detected poses.
- Improve performance on low-resolution images.
- Integrate with other AI models for advanced gesture recognition.
- Deploy as a web or mobile application.

