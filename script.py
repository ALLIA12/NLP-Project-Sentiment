import json
import numpy as np
import os
import pickle
import tensorflow as tf
from azureml.core.model import Model


def init():
    global model
    global tokenizer

    # Load the TensorFlow model
    model_path = Model.get_model_path('sentiment_analysis_model')
    model = tf.saved_model.load(model_path)

    # Load the tokenizer
    tokenizer_path = Model.get_model_path(
        'my_tokenizer')  # Assuming you've registered the tokenizer as a model in Azure ML
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)


def preprocess_text(texts):
    # Assuming 'texts' is a list of text entries to be processed
    # Tokenize the texts
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad the sequences (if necessary)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                                     maxlen=52)  # Specify `your_max_length` based on your model's expected input

    return padded_sequences


def run(raw_data):
    try:
        # Extract texts from the input JSON
        data = json.loads(raw_data)['data']

        # Preprocess the text data
        processed_data = preprocess_text(data)

        # Perform prediction
        predictions = model(processed_data)

        # Convert predictions to list (if necessary) and return
        return json.dumps({"result": predictions.numpy().tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
