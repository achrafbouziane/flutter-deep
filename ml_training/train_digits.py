import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Create assets directory if it doesn't exist
if not os.path.exists('../assets'):
    os.makedirs('../assets')

def train_digits():
    print("Loading MNIST data...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN (28, 28, 1)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    print("Building Model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training Model...")
    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('../assets/digits.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Save labels
    with open('../assets/digits.txt', 'w') as f:
        for i in range(10):
            f.write(f'{i}\n')
            
    print("Done! Model saved to ../assets/digits.tflite")

if __name__ == '__main__':
    train_digits()
