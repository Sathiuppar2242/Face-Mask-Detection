# \# 🛡️ AI Face Mask Detection System

# 

# A real-time AI-powered Face Mask Detection System built using Python, TensorFlow, OpenCV, and Flask.

# 

# The system uses a trained Convolutional Neural Network (CNN) to classify whether a detected face is wearing a mask or not.

# 

# \## 🚀 Features

# 

# \- Real-time webcam face detection

# \- AI-based face mask classification

# \- CNN deep learning model

# \- Haar Cascade face detection

# \- Detection confidence score

# \- Multiple face detection

# \- Real-time detection status

# \- Detection history

# \- Detection history dashboard

# \- Responsive web interface

# \- Flask-based web application

# \- Model performance reporting

# 

# \## 🧠 Technologies Used

# 

# \- Python

# \- TensorFlow / Keras

# \- OpenCV

# \- NumPy

# \- Flask

# \- HTML

# \- CSS

# \- JavaScript

# 

# \## 📊 Model Performance

# 

# \*\*Test Accuracy: 96.30%\*\*

# 

# The trained CNN model is used to classify detected faces into:

# 

# \- 😷 With Mask

# \- ⚠️ Without Mask

# 

# \## 📁 Project Structure

# 

# ```text

# Face-Mask-Detection/

# │

# ├── data/

# ├── models/

# │   └── face\_mask\_detector.keras

# │

# ├── reports/

# ├── screenshots/

# │

# ├── src/

# │   ├── face\_detector.py

# │   ├── mask\_detector.py

# │   ├── preprocess.py

# │   ├── train\_model.py

# │   ├── evaluate\_model.py

# │   └── performance\_report.py

# │

# ├── static/

# │

# ├── templates/

# │   ├── index.html

# │   └── history.html

# │

# ├── app.py

# ├── requirements.txt

# ├── .gitignore

# └── README.md



Installation



Clone the repository:



git clone <YOUR\_GITHUB\_REPOSITORY\_URL>



Move into the project directory:



cd Face-Mask-Detection



Create a virtual environment:



python -m venv venv



Activate the virtual environment on Windows:



venv\\Scripts\\Activate.ps1



Install dependencies:



pip install -r requirements.txt

▶️ Run the Application



Start the Flask application:



python app.py



Open the application in your browser:



http://127.0.0.1:5000/

📊 Detection History



The application provides a dedicated detection history dashboard.



Open:



http://127.0.0.1:5000/history



The history page displays:



Detection time

Detection status

Confidence

Number of detected faces

🔍 How It Works

The webcam captures live video.

OpenCV detects faces using Haar Cascade.

Each detected face is extracted from the frame.

The face is resized to the model input size.

The image is normalized and passed to the CNN model.

The model predicts whether the person is wearing a mask.

The prediction confidence is calculated.

The result is displayed on the live camera feed.

Detection information is available through the web dashboard.

Detection records are maintained in the history system.

🎯 Project Objective



The objective of this project is to demonstrate how computer vision and deep learning can be combined to build a real-time safety monitoring application.



🔮 Future Enhancements

Automatic detection alerts

Detection statistics and charts

Database-based history storage

Face recognition integration

Multiple camera support

Email or notification alerts

Cloud deployment

Improved object detection models

👨‍💻 Author



Sathish R



Computer Science Engineering Student



📄 License



This project is developed for educational and portfolio purposes.





Save the file.



Then run:



```powershell

git add README.md

git commit -m "Improve project documentation"

git push origin master

