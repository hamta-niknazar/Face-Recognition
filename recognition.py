import cv2
import os

# ----------------------------
# Paths and Setup
# ----------------------------

CONFIDENCE_THRESHOLD = 60  # tune this value

trainer_path = "trainer"
names_file = "names.txt"
trainer_file = f"{trainer_path}/trainer.yml"

# Check if required files exist
if not os.path.exists(names_file):
    print(f"[ERROR] {names_file} not found. Please capture and train faces first.")
    exit()

if not os.path.exists(trainer_file):
    print(f"[ERROR] {trainer_file} not found. Please train the model first.")
    exit()

# Load names
with open(names_file, 'r') as f:
    names = [line.strip() for line in f.readlines() if line.strip()]

if len(names) == 0:
    print("[ERROR] No names found in names.txt. Please capture faces first.")
    exit()

print(f"[INFO] Loaded {len(names)} user(s): {', '.join(names)}")

# Initialize recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_file)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----------------------------
# Face Recognition
# ----------------------------

cam = cv2.VideoCapture(0)

# Try setting lower resolution to speed up both capture & detection. (My camera rejects it so I commented these lines.)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

print("\n[INFO] Starting face recognition...")
print("[INFO] Press 'ESC' to exit or close the window.")

try:
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Camera error.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.2, 
            minNeighbors=5, 
            minSize=(int(minW), int(minH))
        )

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < CONFIDENCE_THRESHOLD:
                name = names[id-1] if 0 < id <= len(names) else "Unknown"
                confidence_text = f"{round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"{round(100 - confidence)}%"

            # Draw rectangle and labels
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, str(name), (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence_text), (x+5, y+h-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        cv2.imshow('Face Recognition', img)
        
        # Exit on ESC key or window close
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break
        
        # Check if window was closed
        if cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    # Clean up
    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Face recognition stopped. Goodbye!")