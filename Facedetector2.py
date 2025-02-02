import cv2
import numpy as np
from deepface import DeepFace
from feat import Detector

# Initialize Face Detector (OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Py-Feat detector
feat_detector = Detector()

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert frame to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # If at least one face is detected
    if len(faces) > 0:
        print("Face detected!")  

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract face region
            face_roi = frame[y:y+h, x:x+w]

            # Convert to RGB (DeepFace expects RGB format)
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            try:
                # Analyze emotions using DeepFace
                result = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)

                # Extract dominant emotion and confidence
                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion]

                # Display emotion text on frame
                text = f"{emotion} ({confidence:.2f})"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"Detected Emotion: {emotion}, Confidence: {confidence:.2f}")

            except Exception as e:
                print(f"Emotion Detection Error: {str(e)}")

    # Show the frame with face detection and emotion analysis
    cv2.imshow("Live Face & Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()