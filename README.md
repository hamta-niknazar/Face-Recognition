# Face Detection and Recognition
A simple Python-based face recognition system that captures faces from a webcam, stores them in a dataset, and trains a local model using OpenCV’s LBPHFaceRecognizer. Includes user management and allows viewing registered users.

---

## Features

- Capture face images with live webcam feed.
- Store captured faces in a dataset folder.
- Train a local LBPH face recognition model.
- Manage registered users (add/view).
- Customizable number of pictures per user.

---

## Usage

Clone the repository:

git clone https://github.com/hamta-niknazar/Face-Recognition-and-Detection

cd FaceDetection


Run the program:

    python main.py

  Select from the menu:
  
  0 → Setup: Change number of pictures per user.
  
  1 → Capture new face(s).
  
  2 → Train the model.
  
  3 → View registered users.
  
  4 → Exit.

    python recognition.py

---

## Requirements

- Python 
- OpenCV (`opencv-python`, `opencv-contrib-python`)  
- NumPy  
- Pillow  

Install dependencies:
```bash
pip install opencv-python opencv-contrib-python numpy pillow

