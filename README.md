Plant Disease Detection
A web application using Streamlit to detect plant diseases from leaf images.

Description
This application allows users to upload an image of a plant leaf or take a camera input to detect the presence of any diseases. The model uses a combination of machine learning algorithms and natural language processing techniques to provide accurate predictions.

Features
Upload an image file of a plant leaf
Take a camera input to capture a live photo of a plant leaf
Receive a prediction with a confidence score indicating the likelihood of disease presence
Suggestions for curing or reducing the disease are provided if the prediction is not "healthy"
Requirements
Streamlit library (install using pip install streamlit)
TensorFlow and Keras libraries for model development
Google Drive credentials to download pre-trained models ( stored in dockerfile)
Installation
Clone this repository: git clone https://github.com/your-username/plant-disease-detector.git
Install the required dependencies: pip install -r requirements.txt
Build the Docker image using docker build .
Run the application using streamlit run app.py
Usage
Open a web browser and navigate to http://localhost:8501
Choose an input method (File Uploader or Camera Input)
Upload an image file or take a camera input
Wait for the model to process the image and provide a prediction
Models
The application uses two pre-trained models:

Model 1: Downloaded from Google Drive using gdown
Model 2: Downloaded from Google Drive using gdown
These models are used for disease detection and suggest potential cures.
