import cv2
import math
import pyautogui
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

current_letter = 'A'
selected_letter = 'A'
letters = [chr(i) for i in range(65, 91)]  # A-Z
letter_index = 0
angle_threshold = 45

cap = cv2.VideoCapture(0)

def calculate_angle(p1, p2):
    """
    Calculate the angle between the vertical line and the line formed by two points.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return abs(angle) if angle < 0 else 360 - angle

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            forehead = face_landmarks.landmark[10]

            h, w, _ = frame.shape
            nose_coords = (int(nose_tip.x * w), int(nose_tip.y * h))
            forehead_coords = (int(forehead.x * w), int(forehead.y * h))

            cv2.circle(frame, nose_coords, 5, (255, 0, 0), -1)
            cv2.circle(frame, forehead_coords, 5, (0, 255, 0), -1)
            cv2.line(frame, nose_coords, forehead_coords, (0, 255, 255), 2)

            angle = calculate_angle(nose_coords, forehead_coords)
            cv2.putText(frame, f"Angle: {int(angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if angle > 90 + angle_threshold:
                letter_index = (letter_index + 1) % len(letters)
                selected_letter = letters[letter_index]
            elif angle < 90 - angle_threshold:
                pyautogui.typewrite(selected_letter)
    cv2.putText(frame, f"Selected Letter: {selected_letter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Face Detection Typing', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
