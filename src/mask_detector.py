import cv2
import numpy as np
from tensorflow.keras.models import load_model


# ==========================================
# Load trained mask detection model
# ==========================================

MODEL_PATH = "models/face_mask_detector.keras"

model = load_model(MODEL_PATH)

print("Mask detection model loaded successfully!")


# ==========================================
# Load Haar Cascade face detector
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Haar Cascade loaded successfully!")


# ==========================================
# Open webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started!")
print("Press 'q' to quit.")


# ==========================================
# Real-time detection
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80)
    )

    # Process every detected face
    for (x, y, w, h) in faces:

        # Extract face
        face = frame[y:y + h, x:x + w]

        # Resize to CNN input size
        face = cv2.resize(face, (128, 128))

        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize pixel values
        face = face.astype("float32") / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Make prediction
        prediction = model.predict(
            face,
            verbose=0
        )[0][0]

        # ==================================
        # Mask classification
        # ==================================

        if prediction >= 0.5:

            label = "With Mask"

            confidence = prediction * 100

            box_color = (0, 255, 0)

        else:

            label = "Without Mask"

            confidence = (1 - prediction) * 100

            box_color = (0, 0, 255)

        # ==================================
        # Draw face rectangle
        # ==================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            3
        )

        # ==================================
        # Display prediction
        # ==================================

        text = f"{label}: {confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )

    # ======================================
    # Display face count
    # ======================================

    cv2.putText(
        frame,
        f"Faces Detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ======================================
    # Display webcam
    # ======================================

    cv2.imshow(
        "Face Mask Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Release resources
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Face Mask Detection stopped.")