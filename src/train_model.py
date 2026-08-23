import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
# Load preprocessed datasets
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_validation = np.load("X_validation.npy")
y_validation = np.load("y_validation.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Dataset loaded successfully!")

print(f"Training images: {len(X_train)}")
print(f"Validation images: {len(X_validation)}")
print(f"Testing images: {len(X_test)}")


# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid")
])


# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Display model architecture
model.summary()

# Train the CNN model
print("\nStarting model training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=10,
    batch_size=32,
    verbose=1
)

print("\nModel training completed!")
# Save the trained model
model.save("models/face_mask_detector.keras")

print("Model saved successfully!")