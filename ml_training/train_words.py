import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random

# Create assets directory if it doesn't exist
if not os.path.exists('../assets'):
    os.makedirs('../assets')

WORDS = ["CAT", "DOG", "HI", "YES", "NO"]
NUM_SAMPLES_PER_WORD = 1000

def create_dataset():
    x_data = []
    y_data = []

    print("Generating synthetic dataset...")
    # Try to load a font, fallback to default
    try:
        # Check for common fonts on Windows/Linux
        font_path = "arial.ttf" 
        font = ImageFont.truetype(font_path, 20)
    except:
        print("Warning: arial.ttf not found, using default font.")
        font = ImageFont.load_default()

    for label_idx, word in enumerate(WORDS):
        for _ in range(NUM_SAMPLES_PER_WORD):
            # Create black background image
            img = Image.new('L', (28, 28), color=0)
            draw = ImageDraw.Draw(img)
            
            # Randomize position and font size slightly if possible (with default font strict, but let's try)
            # For simplicity with default font, we just center it.
            # With truetype we could vary size.
            
            # Draw text in white
            # We want to fit it in 28x28. 
            # If word is long, it will be tiny. 
            # But the requirement is 28x28 input.
            
            w = 28
            h = 28
            
            # Rough centering
            # This is a very basic generator
            draw.text((2, 8), word, fill=255, font=font)
            
            # Add some noise/distortion? 
            # Rotation
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, fillcolor=0)
            
            # Convert to numpy
            img_arr = np.array(img)
            
            # Normalize
            img_arr = img_arr / 255.0
            
            x_data.append(img_arr)
            y_data.append(label_idx)

    x_data = np.array(x_data).reshape((-1, 28, 28, 1))
    y_data = np.array(y_data)
    
    return x_data, y_data

def train_words():
    x_train, y_train = create_dataset()
    
    # Shuffle
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    print("Building Model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(WORDS), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training Model...")
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('../assets/words.tflite', 'wb') as f:
        f.write(tflite_model)
        
    # Save labels
    with open('../assets/words.txt', 'w') as f:
        for word in WORDS:
            f.write(f'{word}\n')

    print("Done! Model saved to ../assets/words.tflite")

if __name__ == '__main__':
    train_words()
