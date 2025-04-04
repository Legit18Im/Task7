import pickle
import numpy as np
import tensorflow as tf
import config
from model import build_model
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Function to add noise for data augmentation
def add_noise(X, noise_factor=0.003):
    noise = np.random.randn(*X.shape)
    return X + noise_factor * noise

# Load preprocessed data
with open(f"{config.PROCESSED_DATA_PATH}/processed_data.pkl", "rb") as f:
    X, y = pickle.load(f)

print(f"Loaded data: {X.shape} samples")

# Reshape input for Conv1D + LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))
input_shape = (X.shape[1], 1)

# Data augmentation (add noise)
X_augmented = add_noise(X)
y_augmented = y.copy()

# Combine original + augmented data
X_combined = np.concatenate((X, X_augmented))
y_combined = np.concatenate((y, y_augmented))

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_combined)

# Ensure save directory
os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

# Save label encoder
joblib.dump(label_encoder, os.path.join(config.MODEL_SAVE_PATH, "label_encoder.pkl"))

# Build model
num_classes = len(np.unique(y_encoded))
model = build_model(input_shape, num_classes)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train model
history = model.fit(
    X_combined, y_encoded,
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

# Save model in recommended format
model.save(os.path.join(config.MODEL_SAVE_PATH, "emotion_model.keras"))
print("Model and label encoder saved successfully with augmentation!")