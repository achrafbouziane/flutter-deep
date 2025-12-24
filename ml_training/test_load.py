import tensorflow as tf
import numpy as np
try:
    print("Numpy version:", np.__version__)
    print("TF version:", tf.__version__)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("MNIST loaded successfully")
except Exception as e:
    print("FAILED:", e)
