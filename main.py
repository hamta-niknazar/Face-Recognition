import cv2
import os
import numpy as np
from PIL import Image

# ----------------------------
# Paths and Setup
# ----------------------------

dataset_path = "dataset"
trainer_path = "trainer"
names_file = "names.txt"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
num_pictures_per_user = 50  # Default

# ----------------------------
# Helper Functions
# ----------------------------

def load_names():
    """Load names from file safely"""
    if os.path.exists(names_file):
        with open(names_file, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return []

def save_names(names):
    """Save names to file"""
    with open(names_file, 'w') as f:
        for name in names:
            f.write(f"{name}\n")
    print(f"[INFO] Names saved to {names_file}")

def get_next_id():
    """Return the next available ID safely"""
    files = [f for f in os.listdir(dataset_path) if f.startswith("User.") and f.endswith(".jpg")]
    if not files:
        return 1
    try:
        ids = [int(f.split(".")[1]) for f in files if f.count(".") >= 2]
        return max(ids) + 1 if ids else 1
    except Exception as e:
        print(f"[WARNING] Skipping invalid file in dataset: {e}")
        return 1
    
# ----------------------------
# Functions
# ----------------------------

def setup():
    global num_pictures_per_user
    num = input("Enter number of pictures per user (default 50): ").strip()
    if num.isdigit() and int(num) > 0:
        num_pictures_per_user = int(num)
    print(f"[INFO] Pictures per user set to: {num_pictures_per_user}")

def capture_face():
    names = load_names()
    
    face_name = input("\nEnter user name: ").strip()

    if face_name not in names:
        names.append(face_name)
        face_id = len(names)
        save_names(names)
        print(f"[INFO] New user '{face_name}' added with ID {face_id}")
    else:
        face_id = names.index(face_name) + 1
        print(f"[INFO] Existing user '{face_name}' with ID {face_id}")

    cam = cv2.VideoCapture(0)
    count = 0

    print(f"[INFO] Capturing {num_pictures_per_user} pictures. Press ESC to stop early.")

    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Camera error.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f"{dataset_path}/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.putText(img, f"Captured: {count}/{num_pictures_per_user}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capturing Face", img)
        if cv2.waitKey(1) & 0xFF == 27 or count >= num_pictures_per_user:
            break

    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(100)
    print(f"[INFO] Face capture complete. {count} pictures saved for '{face_name}'.")

# ----------------------------
# Training Function 
# ----------------------------

def train_model():
    names = load_names()
    if len(names) == 0:
        print("[ERROR] No users found. Please capture faces first.")
        return
    
    print("\n[INFO] Training faces...")
    imagePaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]
    
    if len(imagePaths) == 0:
        print("[ERROR] No images found in dataset folder.")
        return
    
    faces, ids = [], []
    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert("L")
            img_numpy = np.array(PIL_img, "uint8")
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            
            faces_detected = face_detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces_detected:
                faces.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        except Exception as e:
            print(f"[WARNING] Skipping invalid file {imagePath}: {e}")
    
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        recognizer.write(f"{trainer_path}/trainer.yml")
        print(f"[INFO] Training complete! {len(set(ids))} user(s) trained with {len(faces)} face samples.")
    else:
        print("[ERROR] No valid faces detected in dataset.")

def view_users():
    names = load_names()
    if len(names) == 0:
        print("\n[INFO] No users registered yet.")
    else:
        print("\n--- Registered Users ---")
        for i, name in enumerate(names, 1):
            print(f"ID {i}: {name}")

# ----------------------------
# Main Menu
# ----------------------------

while True:
    print("\n=== Face Recognition - Capture & Train ===")
    print("0. Setup (Change pictures per user)")
    print("1. Capture New Face")
    print("2. Train Model")
    print("3. View Registered Users")
    print("4. Exit")

    choice = input("Enter choice: ").strip()
    if choice == "0":
        setup()
    elif choice == "1":
        capture_face()
    elif choice == "2":
        train_model()
    elif choice == "3":
        view_users()
    elif choice == "4":
        print("[INFO] Exiting...")
        break
    else:
        print("[ERROR] Invalid choice. Try again.")
