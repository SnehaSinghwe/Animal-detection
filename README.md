# Animal-detection
## Overview
This project showcases a deep learning-powered image classification model capable of identifying 15 distinct animal species from images. Leveraging Convolutional Neural Networks (CNNs) and the strength of transfer learning, the model learns to recognize each animal based on its unique visual patterns and textures.
It lays the groundwork for applications such as real-time wildlife monitoring, educational AI tools, and automated animal identification systems.

## Animal Classes
The model is trained to detect and classify the following animals:
Cat, Dog, Elephant, Horse, Lion, Tiger, Bear, Zebra, Deer, Monkey, Giraffe, Cow, Sheep, Chicken, Fox

## Dataset Details
Image Shape: 224x224x3 (RGB)
Dataset Format: Organized in ImageNet-style directory structure:
dataset/
├── train/
│   ├── Cat/
│   ├── Dog/
│   └── ...
└── test/
    ├── Cat/
    ├── Dog/
    └── ...
Dataset Source: Kaggle

## Tech Stack
Programming Language: Python

## Libraries & Tools:
TensorFlow / Keras
NumPy & Pandas
Matplotlib & Seaborn
Scikit-learn
OpenCV (for preprocessing and future real-time extensions)

## Model Architecture
Base Model: MobileNetV2 or ResNet50 (Transfer Learning)
Custom Classifier:
GlobalAveragePooling → Dense(128) → Dropout → Dense(15, Softmax)
Loss Function: Categorical Crossentropy
Optimizer: Adam
Evaluation Metric: Accuracy

## Getting Started
1. Clone the Repository
git clone https://github.com/SnehaSinghwe/animal-detection.git
cd animal-detection
2. Create a Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate     # For Windows
3. Install Dependencies
pip install -r requirements.txt

## Model Performance

Accuracy: ~91–94%
Loss: Low and Stable
Inference Time:  ~20–30ms per image (GPU)
Includes training/validation curves, confusion matrix, and visualized predictions.

## Project Structure

animal-detection/

├── dataset/

│   ├── train/

│   └── test/

├── models/

│   └── animal_classifier.h5

├── images/

│   └── sample_predictions/

├── requirements.txt

└── README.md

## Future Enhancements
 Integrate real-time animal detection using OpenCV and live webcam feed
 Deploy via Streamlit or Flask as an interactive web app
 Convert to TensorFlow Lite (TFLite) for mobile compatibility
 Expand dataset with rare or exotic animal species
