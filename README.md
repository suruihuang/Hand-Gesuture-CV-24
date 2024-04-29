# Hand-Gesture-CV-24

## Overview
This is the repo for the computer vision final project. The datasets used in this project are not uploaded due to their volume. However, you can access them through the following links:
- Rock-Paper-Scissors dataset: [https://storage.googleapis.com/mediapipetasks/gesture_recognizer/rps_data_sample.zip](https://storage.googleapis.com/mediapipetasks/gesture_recognizer/rps_data_sample.zip)
- ASL Alphabet dataset: [https://www.kaggle.com/dsv/29550](https://www.kaggle.com/dsv/29550)

## Project Directory

### `data`
- Contains the datasets and related data files used for model training and evaluation. The dataset can be downloaded from: ASL Alphabet dataset: [https://www.kaggle.com/dsv/29550](https://www.kaggle.com/dsv/29550)
- Since the file is too large, the processed landmark data can be found on google drive: [https://drive.google.com/file/d/1fuDxxlbtOqP-sYCmQHtybUu0iGoXUxOT/view?usp=sharing]

### `output`
- Stores the model for gui and the classification report for the model.
### `deprecated models`
- Stores experimental models and report. Not in use


### `src`
Contains the source code for the project.

- `connect_model.py`: Script to connect and load models to GUI.
- `fivelayercnn.py`: Python script for a 5-layer CNN model. Needed for model to run
- `gui.py`: Graphical User Interface for the application. Main driver code. 
- `landmark_stream.py`, `landmark.py`: Script for landmark detection. 
- `image_processing.py`: used for blob detection. 

### `model`
- `asl_cnn.ipynb`: Jupyter notebook for the American Sign Language CNN model with result.

#### `deprecated`
- `threelayerCNN.py`: Python script for the 3-layer CNN model. Not in use
- `cnn_model.py`: Python script for a the initial CNN model. Not in use

### `ASL_chart.jpg`
- An image chart for American Sign Language generated from dataset. Used for GUI. 

### `hand_landmarker.task`
- for landmark detection 


### `requirements.txt`
- necessary package for running the scirpt. 



