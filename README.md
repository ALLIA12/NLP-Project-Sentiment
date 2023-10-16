
# Sentiment Analysis Application

This application performs sentiment analysis using a Deep CNN model built with TensorFlow. It provides a GUI where users can enter text and see whether the sentiment is positive or negative.

## Dataset
The model was trained on the [Sentiment140 dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140). Training statistics: 
```
loss: 0.1550 - accuracy: 0.9379
```

## Requirements
To run this application, you need to have the following packages installed:
- TensorFlow
- NumPy

You can install them using `pip`:
```
pip install tensorflow numpy
```

## Running the Application
After ensuring you have the required packages, simply run the provided main.py script. The GUI will show up, allowing you to enter a text and analyze its sentiment.

## Application Overview
1. The main window is titled "Sentimental Analysis" and provides an input field where users can enter their text.
2. After entering the text, click on the "Submit" button to get the sentiment result.
3. A message box will pop up displaying whether the entered text has a positive or negative sentiment, along with a confidence score.

## Model Overview
The `DeepCNN` class in the code defines the architecture of the Convolutional Neural Network. It utilizes embeddings followed by multiple convolutional layers with varying kernel sizes to capture bi-grams, tri-grams, and quad-grams from the input text. The final layers include dense layers and a dropout layer for regularization.

## Credits
Application and Model created by Ali Mohammad and Dawi Alotaibi.

