import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# --- Eye Aspect Ratio (EAR) function ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Drawing utils
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices (from MediaPipe Face Mesh: 468 landmarks)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # around left eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # around right eye

# EAR Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# Counter
counter = 0

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye coordinates
            h, w, _ = frame.shape
            left_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w),
                          int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            # Compute EAR
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eyes
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Drowsiness detection
            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                counter = 0

            # Show EAR value
            cv2.putText(frame, f"EAR: {ear:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
