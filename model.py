import tensorflow as tf
import logging
import numpy as np

try:
    model = tf.keras.models.load_model('mnist.h5')
except FileNotFoundError as e:
    logging.error(f"Model file are not Found: {e}")


def preprocessing(input):
    return input.reshape(1, 28, 28, 1)

def predict(input):
    predict = model.predict(input)
    return predict

