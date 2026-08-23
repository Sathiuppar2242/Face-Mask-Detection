import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


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

print("Loading dataset...")

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
# Normalize
# ==========================================

images = images / 255.0


# ==========================================
# Create test split
# ==========================================

_, X_test, _, y_test = train_test_split(
    images,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)


print("\nTesting images:", len(X_test))


# ==========================================
# Load trained model
# ==========================================

print("\nLoading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# Generate predictions
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)


# Convert probabilities to classes

y_pred = (predictions >= 0.5).astype(int).flatten()


# ==========================================
# Classification report
# ==========================================

print("\n===================================")
print("       CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Without Mask",
            "With Mask"
        ]
    )
)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n===================================")
print("       CONFUSION MATRIX")
print("===================================")

print(cm)


# ==========================================
# Create reports directory
# ==========================================

os.makedirs("reports", exist_ok=True)


# ==========================================
# Save confusion matrix
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Without Mask",
        "With Mask"
    ]
)


display.plot()

plt.title("Face Mask Detection - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300
)

plt.show()


# ==========================================
# Calculate accuracy
# ==========================================

accuracy = np.mean(
    y_pred == y_test
)


# ==========================================
# Create accuracy chart
# ==========================================

plt.figure(figsize=(7, 5))

plt.bar(
    ["Test Accuracy"],
    [accuracy * 100]
)

plt.ylim(0, 100)

plt.ylabel("Accuracy (%)")

plt.title("Face Mask Detection Model Accuracy")

plt.text(
    0,
    accuracy * 100 + 2,
    f"{accuracy * 100:.2f}%",
    ha="center",
    fontsize=12
)

plt.tight_layout()

plt.savefig(
    "reports/test_accuracy.png",
    dpi=300
)

plt.show()


# ==========================================
# Final result
# ==========================================

print("\n===================================")
print("       PERFORMANCE SUMMARY")
print("===================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nReports saved in:")

print("reports/confusion_matrix.png")

print("reports/test_accuracy.png")

print("===================================")