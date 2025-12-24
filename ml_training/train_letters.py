import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# Define assets directory relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
assets_dir = os.path.join(project_root, 'assets')
os.makedirs(assets_dir, exist_ok=True)

def train_letters():
    print("Loading EMNIST Letters dataset via tensorflow_datasets...")

    (ds_train, ds_test), info = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        with_info=True
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        # EMNIST images are rotated 90 degrees and flipped (transposed).
        # We need to swap width and height: (W, H, C) -> (H, W, C)
        image = tf.transpose(image, perm=[1, 0, 2])
        label = label - 1               # 1–26 → 0–25
        return image, label

    ds_train = ds_train.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    
    # DEBUG: Save a grid of images to verify orientation
    print("Saving debug image grid to verify orientation...")
    try:
        import matplotlib.pyplot as plt
        
        # Take one batch
        for images, labels in ds_train.take(1):
            plt.figure(figsize=(10, 10))
            for i in range(25):
                plt.subplot(5, 5, i + 1)
                # Reshape to 28x28 for display
                img_data = images[i].numpy().reshape(28, 28)
                plt.imshow(img_data, cmap='gray')
                plt.title(f"Label: {labels[i].numpy()}")
                plt.axis('off')
            
            debug_path = os.path.join(assets_dir, 'debug_emnist.png')
            plt.savefig(debug_path)
            print(f"Debug image saved to {debug_path}")
            plt.close()
    except ImportError:
        print("Matplotlib not found, skipping debug image.")
    except Exception as e:
        print(f"Failed to save debug image: {e}")

    print("Building model...")
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(26, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training model...")
    model.fit(ds_train, epochs=3, validation_data=ds_test)

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(assets_dir, 'letters.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    txt_path = os.path.join(assets_dir, 'letters.txt')
    with open(txt_path, 'w') as f:
        for c in alphabet:
            f.write(c + '\n')

    print(f"✅ DONE: Model saved to {tflite_path}")

if __name__ == "__main__":
    train_letters()
