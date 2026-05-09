## Deepfake Detection System

An AI-powered web application that detects manipulated (deepfake) images and videos using a Convolutional Neural Network (CNN). The system allows users to upload media files through a simple web interface and receive real-time authenticity predictions with confidence scores.

---

## Features

- Deepfake detection for both images and videos
- CNN-based media classification
- Login-free and user-friendly interface
- Automated frame extraction from uploaded videos
- Analysis of up to 80 frames per video
- Fast inference pipeline (5–10 seconds)
- Confidence score visualization
- Built using Django backend with HTML/CSS/JavaScript frontend

---

## Tech Stack

### Backend
- Python
- Django
- OpenCV
- TensorFlow / Keras
- NumPy

### Frontend
- HTML
- CSS
- JavaScript

### AI/ML
- Convolutional Neural Networks (CNNs)
- Frame-based video analysis

---

## System Workflow

1. User uploads an image or video.
2. Backend preprocesses the media.
3. For videos:
   - Frames are extracted automatically.
   - Up to 80 frames are selected for analysis.
4. CNN model performs prediction on extracted frames/images.
5. Confidence scores are generated.
6. Final result is displayed as:
   - Real
   - Fake (Deepfake)

---

## Project Structure

Deepfake-Detection-System/
│
├── backend/
│   ├── models/
│   ├── views.py
│   ├── urls.py
│   └── inference.py
│
├── frontend/
│   ├── templates/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│
├── media/
├── uploads/
├── requirements.txt
├── manage.py
└── README.md
---

## Installation

### 1. Clone the Repository


git clone https://github.com/your-username/deepfake-detection-system.git
cd deepfake-detection-system

### 2. Create Virtual Environment
python -m venv venv
### 3. Activate Virtual Environment
#### Windows
venv\Scripts\activate
#### Linux/macOS
source venv/bin/activate
### 4. Install Dependencies
pip install -r requirements.txt

### 5. Run the Server
python manage.py runserver
---
## Usage

1. Open the web application in your browser.
2. Upload an image or video file.
3. Wait for processing and inference.
4. View prediction results and confidence score.

---

## Future Improvements

- Real-time webcam deepfake detection
- Transformer-based detection models
- Explainable AI visualizations
- Mobile application integration
- Cloud deployment and scalable inference

---

## Disclaimer

This project is designed for educational, research, and awareness purposes only. Detection results may not always be fully accurate and should not be used as sole evidence for verification decisions.

---

## Author

**Archishman Das**  
B.Tech CSE (AIML) — Institute of Engineering and Management, Kolkata
