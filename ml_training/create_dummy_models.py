import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

if not os.path.exists('../assets'):
    os.makedirs('../assets')

def create_dummy_model(name, num_classes, labels):
    print(f"Creating dummy model for {name}...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(f'../assets/{name}.tflite', 'wb') as f:
        f.write(tflite_model)
        
    with open(f'../assets/{name}.txt', 'w') as f:
        for l in labels:
            f.write(f'{l}\n')
            
    print(f"Saved ../assets/{name}.tflite")

def main():
    # Digits
    create_dummy_model('digits', 10, [str(i) for i in range(10)])
    
    # Letters
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    create_dummy_model('letters', 26, list(alphabet))
    
    print("Dummy models created. Use these for UI testing if training fails.")

if __name__ == '__main__':
    main()
