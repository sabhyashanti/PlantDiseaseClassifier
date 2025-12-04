🌿 Plant Disease Classifier

A Deep Learning web application that detects and classifies diseases in plant leaves using computer vision. Built with Python, TensorFlow, and Streamlit.

🚀 Overview

This project is an AI-powered solution designed to help identify plant diseases early. Users can upload an image of a plant leaf, and the system processes it using a trained Convolutional Neural Network (CNN) to predict the specific disease or confirm if the plant is healthy.

The model was trained on Google Colab using a dataset from Kaggle and achieves high accuracy in classifying various plant ailments.

✨ Features

Image Upload: Simple interface to upload leaf images (JPG/PNG).

Real-time Prediction: Instant analysis and disease classification.

Confidence Scores: Displays the model's confidence in its prediction.

User-Friendly UI: Clean and responsive dashboard powered by Streamlit.

🛠️ Tech Stack

Language: Python

Framework: Streamlit (for Web UI)

Machine Learning: TensorFlow / Keras

Data Processing: NumPy, Pandas, PIL

Training Platform: Google Colab

📂 Project Structure

PlantDiseaseClassifier/
├── app/
│   ├── trained_model/      # Folder for the .h5 model file
│   └── ...
├── main.py                 # Main Streamlit application file
├── requirements.txt        # List of python dependencies
├── .gitignore              # Files to exclude from Git
└── README.md               # Project documentation


⚙️ Installation & Setup

Follow these steps to run the project locally on your machine.

1. Clone the Repository

git clone [https://github.com/sabhyashanti/PlantDiseaseClassifier.git](https://github.com/sabhyashanti/PlantDiseaseClassifier.git)
cd PlantDiseaseClassifier


2. Create a Virtual Environment

# Windows
python -m venv venv
.\venv\Scripts\Activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. ⚠️ Important: Model Setup

Due to GitHub's file size limits, the pre-trained model file (plant_disease_prediction_model.h5) is not included in this repository.

To run the app, you must either:

Train the model yourself using the provided dataset/notebook (if applicable).

Place your own .h5 model file inside the app/trained_model/ directory.

Ensure the file is named: plant_disease_prediction_model.h5

5. Run the Application

streamlit run main.py


📊 Dataset

The model was trained using the PlantVillage dataset available on Kaggle. It includes thousands of images of healthy and diseased leaves across multiple crop species.
