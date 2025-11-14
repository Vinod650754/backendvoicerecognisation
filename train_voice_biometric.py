#!/usr/bin/env python3
"""
Voice Biometric Training Script
Trains a model to recognize the owner's voice using MFCC features.
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json

# ============================================================================
# Step 1: Extract MFCC Features from Audio
# ============================================================================

def extract_mfcc(file_path, n_mfcc=13, sr=16000):
    """Extract MFCC features from an audio file."""
    try:
        y, sr_loaded = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)  # Average over time frames
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ============================================================================
# Step 2: Prepare Dataset
# ============================================================================

def load_dataset(data_folder):
    """Load audio dataset organized as data_folder/owner/wakeword/*.wav and /command/*.wav"""
    X, y = [], []
    labels_map = {}
    label_index = 0

    for user_folder in os.listdir(data_folder):
        user_path = os.path.join(data_folder, user_folder)
        if os.path.isdir(user_path):
            if user_folder not in labels_map:
                labels_map[user_folder] = label_index
                label_index += 1

            # Recursively load all .wav files under user_path (supports nested dirs like wakeword/, command/)
            for root, dirs, files in os.walk(user_path):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        feature = extract_mfcc(file_path)
                        if feature is not None:
                            X.append(feature)
                            y.append(labels_map[user_folder])
                            print(f"Loaded: {file_path}")

    if len(X) == 0:
        print("ERROR: No audio files found in dataset. Please check data folder structure.")
        return None, None, None

    return np.array(X), np.array(y), labels_map


# ============================================================================
# Step 3: Build and Train Model
# ============================================================================

def train_voice_biometric_model(data_folder='./data', epochs=50, batch_size=8):
    """Train a voice biometric model on owner's voice."""
    print("Loading dataset from:", data_folder)
    X, y, labels_map = load_dataset(data_folder)

    if X is None or len(X) == 0:
        print("No data found. Cannot train.")
        return False

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Labels map: {labels_map}")
    print(f"Number of samples: {len(X)}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Build model
    n_mfcc = X.shape[1]
    num_classes = len(labels_map)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_mfcc,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/voice_biometric_model.h5')
    print("Saved: models/voice_biometric_model.h5")

    # Save labels map
    with open('models/voice_biometric_labels.json', 'w') as f:
        json.dump(labels_map, f)
    print("Saved: models/voice_biometric_labels.json")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('models/voice_biometric_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Saved: models/voice_biometric_model.tflite")

    return True


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Voice Biometric Training")
    print("=" * 70)

    data_folder = './data'
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]

    success = train_voice_biometric_model(data_folder=data_folder, epochs=50, batch_size=8)
    if success:
        print("\nTraining complete! Use the model in backend/app.py for voice verification.")
    else:
        print("\nTraining failed. Ensure you have audio samples in the data folder.")
        sys.exit(1)
