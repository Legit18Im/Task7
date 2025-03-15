import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Load test data
data = np.load("/kaggle/working/processed_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Load trained model
model = tf.keras.models.load_model("/kaggle/working/emotion_model.keras")

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
print(classification_report(y_true_classes, y_pred_classes))