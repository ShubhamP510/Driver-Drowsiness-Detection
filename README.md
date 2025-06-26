Driver Drowsiness Detection System
Overview
A cutting-edge solution aimed at enhancing road safety by monitoring driver fatigue in real time. Employs deep learning to analyze eye movements and detect signs of drowsiness, triggering an alert to prevent accidents.

✨ Key Features
Real-time Monitoring: Continuously captures and processes webcam video to assess eye state.

Deep Learning Model: A CNN is trained to distinguish open vs closed eyes.

Alert Mechanism: An audible alarm sounds if drowsiness is detected.
⚙️ Installation Guide
Prerequisites
Python 3.7+

Webcam

Setup Instructions
Clone the repository
git clone https://github.com/Shubham-Singla259/Driver-Drowsiness-Detection-System.git
cd Driver-Drowsiness-Detection-System
Install dependencies

pip install -r requirements.txt

Download the MRL Eye Dataset (from Kaggle), extract into MRL Eye Dataset/mrlEyes_2018_01.

Run the Application
Open main.ipynb in Jupyter, connect to your webcam, and execute the notebook.

🚦 Usage
Data Preparation: Run Data Preparation.ipynb to preprocess images: resizing, normalization, train/test split.

Model Training: Use Model Training.ipynb to train the CNN; the best model is saved as best_model.h5.

Real-time Detection: Launch main.ipynb for live drowsiness monitoring and alert triggering.

Acknowledgements
MRL Eye Dataset by MRL Lab at VSB-TUO

Haar Cascade Classifiers from OpenCV
