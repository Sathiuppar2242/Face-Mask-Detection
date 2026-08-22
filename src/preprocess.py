import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset paths
DATA_DIR = "data"

WITH_MASK_DIR = os.path.join(DATA_DIR, "with_mask")
WITHOUT_MASK_DIR = os.path.join(DATA_DIR, "without_mask")

# Image configuration
IMG_SIZE = 128

images = []
labels = []

# Load images from a folder
def load_images(folder_path, label):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        image = cv2.imread(file_path)

        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image)
        labels.append(label)


print("Loading masked images...")
load_images(WITH_MASK_DIR, 1)

print("Loading unmasked images...")
load_images(WITHOUT_MASK_DIR, 0)

# Convert to NumPy arrays
images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Normalize pixel values
images = images / 255.0

print(f"\nTotal images: {len(images)}")
print(f"With mask: {np.sum(labels == 1)}")
print(f"Without mask: {np.sum(labels == 0)}")

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    images,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")
print(f"Training:   {len(X_train)}")
print(f"Validation: {len(X_validation)}")
print(f"Testing:    {len(X_test)}")

# Save processed datasets
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_validation.npy", X_validation)
np.save("y_validation.npy", y_validation)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\nPreprocessing completed successfully!")