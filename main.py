import tensorflow as tf
import numpy as np
import IPython.display

if "X0" not in globals():
    (X0, L0), (_, _) = tf.keras.datasets.mnist.load_data()
    X0 = tf.cast(X0, tf.float32) / 255.0
    X0 = tf.expand_dims(X0, -1)

