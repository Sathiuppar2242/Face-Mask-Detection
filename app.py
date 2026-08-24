import cv2
import numpy as np

from datetime import datetime
from flask import Flask, render_template, Response, jsonify
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
# Latest detection information
# ==========================================

latest_detection = {
    "status": "No Face Detected",
    "confidence": 0,
    "faces": 0
}


# ==========================================
# Detection history
# ==========================================

detection_history = []

MAX_HISTORY = 20


# ==========================================
# Add detection to history
# ==========================================

def add_detection_history(
    status,
    confidence,
    faces
):

    detection = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "status": status,
        "confidence": round(
            float(confidence),
            1
        ),
        "faces": faces
    }

    detection_history.insert(
        0,
        detection
    )

    if len(detection_history) > MAX_HISTORY:
        detection_history.pop()


# ==========================================
# Generate webcam frames
# ==========================================

def generate_frames():

    global latest_detection

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

        # Default status
        latest_detection = {
            "status": "No Face Detected",
            "confidence": 0,
            "faces": len(faces)
        }

        # No face detected
        if len(faces) == 0:

            # Add no-face event occasionally
            add_detection_history(
                "No Face Detected",
                0,
                0
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

            # Convert BGR to RGB
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

            # Classification
            if prediction >= 0.5:

                label = "With Mask"

                confidence = prediction * 100

                color = (0, 255, 0)

            else:

                label = "Without Mask"

                confidence = (
                    1 - prediction
                ) * 100

                color = (0, 0, 255)

            # Update latest detection
            latest_detection = {
                "status": label,
                "confidence": round(
                    float(confidence),
                    1
                ),
                "faces": len(faces)
            }

            # Add detection to history
            add_detection_history(
                label,
                confidence,
                len(faces)
            )

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                3
            )

            # Display prediction
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

        # Encode frame
        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Send frame to browser
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
# Detection status API
# ==========================================

@app.route("/detection_status")
def detection_status():

    return jsonify(
        latest_detection
    )


# ==========================================
# Detection history API
# ==========================================

@app.route("/detection_history")
def detection_history_api():

    return jsonify(
        detection_history
    )


# ==========================================
# Clear detection history
# ==========================================

@app.route("/clear_history", methods=["POST"])
def clear_history():

    detection_history.clear()

    return jsonify({
        "success": True,
        "message": "Detection history cleared successfully."
    })


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
        debug=False,
        threaded=True
    )