import cv2
import numpy as np

from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model


# ==========================================
# Flask application
# ==========================================

app = Flask(__name__)


# ==========================================
# Load trained AI model
# ==========================================

MODEL_PATH = "models/face_mask_detector.keras"

model = load_model(MODEL_PATH)

print("Mask detection model loaded successfully!")


# ==========================================
# Load Haar Cascade face detector
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

print("Haar Cascade loaded successfully!")


# ==========================================
# Webcam
# ==========================================

camera = cv2.VideoCapture(0)


# ==========================================
# Generate webcam frames
# ==========================================

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

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
            face = frame[
                y:y + h,
                x:x + w
            ]

            # Resize
            face = cv2.resize(
                face,
                (128, 128)
            )

            # Convert BGR → RGB
            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            # Normalize
            face = face.astype(
                "float32"
            ) / 255.0

            # Add batch dimension
            face = np.expand_dims(
                face,
                axis=0
            )

            # AI prediction
            prediction = model.predict(
                face,
                verbose=0
            )[0][0]

            # ==================================
            # Classification
            # ==================================

            if prediction >= 0.5:

                label = "With Mask"

                confidence = prediction * 100

                color = (0, 255, 0)

            else:

                label = "Without Mask"

                confidence = (1 - prediction) * 100

                color = (0, 0, 255)

            # ==================================
            # Draw face rectangle
            # ==================================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                3
            )

            # ==================================
            # Display prediction
            # ==================================

            text = (
                f"{label}: "
                f"{confidence:.1f}%"
            )

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # ==================================
        # Encode frame
        # ==================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame_bytes = buffer.tobytes()

        # ==================================
        # Send frame to browser
        # ==================================

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ==========================================
# Home page
# ==========================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ==========================================
# Video stream
# ==========================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ==========================================
# Run application
# ==========================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("   FACE MASK DETECTION WEB APP")
    print("======================================")
    print("Open: http://127.0.0.1:5000/")
    print("Press CTRL+C to stop the server.")
    print("======================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )