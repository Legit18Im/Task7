import os

# Paths
DATASET_PATH = "TESS"
PROCESSED_DATA_PATH = "saved_models/processed_data_2.pkl"
MODEL_SAVE_PATH = "saved_models"

# Emotion Labels (ensure correct mapping)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]

# Audio Processing
SAMPLE_RATE = 22050  # Standard sampling rate
N_MFCC = 40  # Number of MFCC features
MAX_PAD_LEN = 200  # Padding length for uniform input

# Model Training
BATCH_SIZE = 64  # Increased batch size for better gradient updates
EPOCHS = 40  # More epochs for better convergence
LEARNING_RATE = 0.0001  # Lowered for stable training
TEST_SIZE = 0.2  # Train-test split
