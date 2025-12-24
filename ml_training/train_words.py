import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random

# Define assets directory relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
assets_dir = os.path.join(project_root, 'assets')
os.makedirs(assets_dir, exist_ok=True)

WORDS = ["CAT", "DOG", "HI", "YES", "NO"]
NUM_SAMPLES_PER_WORD = 1000

def create_dataset():
    x_data = []
    y_data = []

    print("Generating synthetic dataset (Matched to App Preprocessing)...")
    try:
        font_path = "arial.ttf" 
        font = ImageFont.truetype(font_path, 40) # Larger size for clear rendering before downsampling
    except:
        print("Warning: arial.ttf not found, using default font.")
        font = ImageFont.load_default()

    for label_idx, word in enumerate(WORDS):
        for _ in range(NUM_SAMPLES_PER_WORD):
            # 1. Draw word on large temporary canvas
            temp_img = Image.new('L', (200, 100), color=0)
            draw = ImageDraw.Draw(temp_img)
            draw.text((10, 10), word, fill=255, font=font)
            
            # 2. Find Bounding Box and Crop
            bbox = temp_img.getbbox()
            if bbox:
                cropped = temp_img.crop(bbox)
            else:
                cropped = temp_img # Should not happen if text drawn

            # 3. Square (Aspect Ratio Preservation)
            w, h = cropped.size
            max_dim = max(w, h)
            square_img = Image.new('L', (max_dim, max_dim), color=0)
            square_img.paste(cropped, ((max_dim - w) // 2, (max_dim - h) // 2))

            # 4. Resize to 20x20
            scaled_content = square_img.resize((20, 20), Image.Resampling.LANCZOS)

            # 5. Paste into 28x28 (Centered -> 4px margin)
            final_img = Image.new('L', (28, 28), color=0)
            
            # Apply slight rotation/offset before final paste? 
            # Actually, standard MNIST is centered. 
            # We can apply perturbations here.
            
            # Augmentation: Rotation / Offset
            angle = random.uniform(-10, 10)
            scaled_content = scaled_content.rotate(angle, fillcolor=0)
            
            final_img.paste(scaled_content, (4, 4))
            
            # Add Noise
            img_arr = np.array(final_img)
            noise = np.random.randint(0, 50, img_arr.shape, dtype='uint8')
            img_arr = np.clip(img_arr + (img_arr > 0) * noise, 0, 255) # Add noise only to strokes? Or background?
            # Let's add slight background noise too for robustness
            bg_noise = np.random.randint(0, 10, img_arr.shape, dtype='uint8')
            img_arr = np.clip(img_arr + bg_noise, 0, 255)

            # Normalize
            img_arr = img_arr.astype('float32') / 255.0
            
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
        layers.Dropout(0.2), # Add dropout
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(WORDS), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training Model...")
    model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(assets_dir, 'words.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    # Save labels
    txt_path = os.path.join(assets_dir, 'words.txt')
    with open(txt_path, 'w') as f:
        for word in WORDS:
            f.write(f'{word}\n')

    print(f"Done! Model saved to {tflite_path}")

if __name__ == '__main__':
    train_words()
