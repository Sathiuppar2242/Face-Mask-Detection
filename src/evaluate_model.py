import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/face_mask_detector.keras"

MASK_DIR = "data/with_mask"
NO_MASK_DIR = "data/without_mask"

IMAGE_SIZE = (128, 128)

TEST_SIZE = 0.10

RANDOM_STATE = 42


# ==========================================
# Load images
# ==========================================

print("Loading test dataset...")

images = []
labels = []


# ==========================================
# Load masked images
# ==========================================

print("Loading masked images...")

for filename in os.listdir(MASK_DIR):

    filepath = os.path.join(MASK_DIR, filename)

    image = cv2.imread(filepath)

    if image is None:
        continue

    image = cv2.resize(image, IMAGE_SIZE)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images.append(image)

    labels.append(1)


# ==========================================
# Load unmasked images
# ==========================================

print("Loading unmasked images...")

for filename in os.listdir(NO_MASK_DIR):

    filepath = os.path.join(NO_MASK_DIR, filename)

    image = cv2.imread(filepath)

    if image is None:
        continue

    image = cv2.resize(image, IMAGE_SIZE)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images.append(image)

    labels.append(0)


# ==========================================
# Convert to NumPy arrays
# ==========================================

images = np.array(images, dtype="float32")

labels = np.array(labels, dtype="int32")


print("\nTotal images:", len(images))

print("With mask:", np.sum(labels == 1))

print("Without mask:", np.sum(labels == 0))


# ==========================================
# Normalize images
# ==========================================

images = images / 255.0


# ==========================================
# Create same 90% / 10% split
# ==========================================

_, test_images, _, test_labels = train_test_split(
    images,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)


print("\nTest dataset created!")

print("Testing images:", len(test_images))


# ==========================================
# Load trained model
# ==========================================

print("\nLoading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# Evaluate model
# ==========================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    test_images,
    test_labels,
    verbose=1
)


# ==========================================
# Display results
# ==========================================

print("\n===================================")
print("       MODEL EVALUATION RESULTS")
print("===================================")

print(f"Test Loss:     {loss:.4f}")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("===================================")