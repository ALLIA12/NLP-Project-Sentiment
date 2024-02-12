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


def run(raw_data):
    try:
        # Extract texts from the input JSON
        data = json.loads(raw_data)['data']

        # Tokenize the texts
        sequence = tokenizer.texts_to_sequences([data])

        # Pad the sequences (if necessary)
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=52, padding='post',
                                                                        truncating='post')
        # Perform prediction
        predictions = model(padded_sequence)

        # Convert predictions to list (if necessary) and return
        return json.dumps({"result": predictions.numpy().tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
