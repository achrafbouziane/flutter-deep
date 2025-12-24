# Handwritten Digits, Letters, and Words Recognition (Flutter + TFLite)

This is an academic project that demonstrates how to integrate Deep Learning (CNNs) with a Flutter mobile application for offline recognition of handwritten content.

## Features
- **Digits Mode**: Recognize 0-9 (Trained on MNIST).
- **Letters Mode**: Recognize A-Z (Trained on EMNIST).
- **Words Mode**: Recognize a small set of vocabulary items (Trained on Synthetic Data).
- **Drawing Canvas**: Interactive canvas for input.
- **Offline Inference**: Uses TensorFlow Lite for on-device prediction.

## Project Structure
- `lib/`: Flutter source code.
  - `screens/`: UI screens (Home, Recognition).
  - `canvas/`: Custom painting implementation.
  - `ml/`: Classifier service and TFLite wrapper.
  - `utils/`: Image processing (bitmap capture, resizing, normalization).
- `ml_training/`: Python scripts to train models.
- `assets/`: Stores the `.tflite` models and labels.

## Setup Instructions

### 1. Train the Models
Before running the app, you MUST generate the TFLite models. The app expects them in the `assets/` directory.

1.  Ensure you have Python installed.
2.  Install requirements:
    ```bash
    pip install -r ml_training/requirements.txt
    ```
3.  Run the training scripts:
    ```bash
    cd ml_training
    python train_digits.py
    python train_letters.py
    python train_words.py
    ```
    This will generate `digits.tflite`, `letters.tflite`, and `words.tflite` (and their corresponding `.txt` label files) in the `assets/` folder.

### 2. Run the App
1.  Ensure you have Flutter installed.
2.  Run the app:
    ```bash
    flutter pub get
    flutter run
    ```

### Troubleshooting
If you encounter errors running the training scripts locally (often due to Python/Numpy version mismatches on Windows), you can:

1.  **Use Dummy Models**: Run `python ml_training/create_dummy_models.py` to generate functional (but non-intelligent) models. This allows you to test the App UI and flow.
2.  **Train on Google Colab**: Upload the `ml_training/` scripts to Google Colab, run them, and download the `.tflite` files to your `assets/` folder.

## Architecture
- **Input**: User draws on canvas -> `List<Offset>`.
- **Processing**: `List<Offset>` -> `Image` -> Resized 28x28 Grayscale -> `List<double>` (Normalized 0-1).
- **Model**: CNN (Conv2D -> MaxPool -> Dense) converts 28x28 input to Class Probabilities.
- **Output**: Highest probability class displayed to user.

## Dependencies
- `tflite_flutter`: For running the models.
- `image`: For image manipulation (resizing, pixel access).
- `path_provider`: For temporary file handling (if needed).
