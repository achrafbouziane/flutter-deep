import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Define assets directory relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
assets_dir = os.path.join(project_root, 'assets')
os.makedirs(assets_dir, exist_ok=True)

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
    
    # DEBUG: Save a grid of images
    print("Saving debug image grid...")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
            plt.title(f"Label: {y_train[i]}")
            plt.axis('off')
        plt.savefig(os.path.join(assets_dir, 'debug_digits.png'))
        plt.close()
    except Exception as e:
        print(f"Debug image failed: {e}")

    print("Building Deeper Model with Augmentation...")
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), # Increased dropout slightly
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training Model (15 epochs)...")
    model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), batch_size=64)

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(assets_dir, 'digits.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Save labels
    txt_path = os.path.join(assets_dir, 'digits.txt')
    with open(txt_path, 'w') as f:
        for i in range(10):
            f.write(f'{i}\n')
            
    print(f"Done! Model saved to {tflite_path}")

if __name__ == '__main__':
    train_digits()
