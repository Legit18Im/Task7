import numpy as np
from model import build_model
import tensorflow as tf

# Load preprocessed data
data = np.load("/kaggle/working/processed_data.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# Build and train the model
model = build_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))  # Increased epochs to 50
model.save("/kaggle/working/emotion_model.keras")
print("Model training completed and saved.")