# NeuroScan - Brain Tumor Detection using Transfer Learning and Custom Model

## Project Overview

NeuroScan is a deep learning-based project aimed at detecting brain tumors from MRI scans using transfer learning techniques. The goal is to build an efficient model using pre-trained neural networks (e.g., VGG16, ResNet) to classify MRI images into categories: **Tumor** and **No Tumor**. This project leverages the power of Convolutional Neural Networks (CNN) to perform image classification.

## Table of Contents

- [Project Setup](#project-setup)
- [Dependencies](#dependencies)
- [Data](#data)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Project Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/prathameshbhavsar19/NeuroScan.git
   cd NeuroScan
Create and activate a virtual environment:
python3 -m venv env
source env/bin/activate
Install the dependencies:
pip install -r requirements.txt
Dependencies

Python 3.x
Keras
TensorFlow
Keras-Tuner
NumPy
Matplotlib
OpenCV
scikit-learn
You can install the required dependencies by running:

pip install -r requirements.txt
Data

This project uses the Brain Tumor MRI Dataset available on Kaggle, which consists of MRI images classified into four categories:

Glioma
Meningioma
Pituitary Tumor
No Tumor
The dataset is split into two main folders: Training and Testing, each containing subfolders for the respective tumor categories.

Model

The model leverages Transfer Learning, using pre-trained CNN architectures such as VGG16 or ResNet as feature extractors, and fine-tuning them for the task of classifying MRI images.

The pre-trained model is modified to suit the problem by adding custom dense layers on top.
The model is trained using sparse categorical cross-entropy loss and an optimizer like RMSprop.
Training

To train the model, run the following script:

python train.py
Hyperparameter tuning (optional)
If you'd like to perform hyperparameter tuning, the keras_tuner module is available. It allows you to search for optimal parameters (e.g., learning rate, batch size).

python tune_model.py
Evaluation

To evaluate the trained model on the test dataset, you can run:

python evaluate_model.py
Results

Once the model is trained and evaluated, the results such as accuracy, loss, and confusion matrix will be displayed.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Kaggle for providing the dataset.
Keras and TensorFlow for their pre-trained models and powerful APIs.
OpenCV for image pre-processing tasks.