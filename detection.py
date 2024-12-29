import cv2
import mediapipe as mp
import random
import time
from utilities import create_landolt_c_image, calculate_hand_direction

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def run_detection():
    orientations = ['right', 'left', 'up', 'down']
    initial_size = 120
    min_size = 20
    size_decrement = 10
    count_correct = 0
    total_c_created = 0
    current_orientation = random.choice(orientations)
    current_size = initial_size
    min_size_reached = False
    extra_rounds_after_min = 3

    previous_nose_x = None
    shake_threshold = 0.05
    shake_count = 0
    required_shakes = 3

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
            mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose, \
            mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as face:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landolt_c = create_landolt_c_image(current_size, thickness=15, orientation=current_orientation)
            h, w, _ = landolt_c.shape
            y_offset = (frame.shape[0] - h) // 2
            x_offset = (frame.shape[1] - w) // 2
            frame[y_offset:y_offset + h, x_offset:x_offset + w] = landolt_c

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(image_rgb)
            pose_results = pose.process(image_rgb)
            face_results = face.process(image_rgb)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            detected_orientation = None

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    detected_orientation = calculate_hand_direction(wrist, index_finger_tip)

                    if detected_orientation == current_orientation:
                        count_correct += 1
                        total_c_created += 1
                        current_size = max(current_size - size_decrement, min_size)
                        if current_size <= min_size:
                            min_size_reached = True
                            extra_rounds_after_min -= 1

                        current_orientation = random.choice(orientations)
                        start_time = time.time()

            cv2.putText(frame, f"Correct: {count_correct} / {total_c_created}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gesture Detection", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()