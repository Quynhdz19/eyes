import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
# from ultralytics import YOLO

# Khởi tạo MediaPipe cho nhận diện bàn tay, cơ thể và khuôn mặt
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# model = YOLO("ppe.pt")


# Hàm tạo hình ảnh chữ "C" với hướng và kích thước cụ thể
def create_landolt_c_image(radius, thickness, orientation, canvas_size=(300, 300)):
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    center = (canvas_size[0] // 2, canvas_size[1] // 2)

    # Xác định góc mở của chữ "C" dựa trên hướng
    if orientation == 'right':
        start_angle, end_angle = 45, 315
    elif orientation == 'left':
        start_angle, end_angle = 225, 495
    elif orientation == 'up':
        start_angle, end_angle = -45, 225
    elif orientation == 'down':
        start_angle, end_angle = 135, 405

    cv2.ellipse(canvas, center, (radius, radius), 0, start_angle, end_angle, (0, 0, 0), thickness)
    return canvas


# Cài đặt ban đầu
orientations = ['right', 'left', 'up', 'down']
initial_size = 120
min_size = 20
size_decrement = 10
count_correct = 0
total_c_created = 0
current_orientation = random.choice(orientations)
current_size = initial_size
min_size_reached = False  # Biến để kiểm tra khi đạt kích thước tối thiểu
extra_rounds_after_min = 3  # Số lần sau khi đạt min_size

# Tham số cho phát hiện lắc đầu
previous_nose_x = None
shake_threshold = 0.05  # Ngưỡng để phát hiện lắc đầu
shake_count = 0
required_shakes = 3  # Số lần lắc đầu để xác định "không biết"


# Hàm tính toán hướng chỉ tay dựa trên góc giữa cổ tay và ngón trỏ
def calculate_hand_direction(wrist, index_finger_tip):
    dx = index_finger_tip.x - wrist.x
    dy = index_finger_tip.y - wrist.y
    angle = math.degrees(math.atan2(dy, dx))

    # Chuyển đổi góc thành hướng chỉ tay
    if -45 <= angle <= 45:
        return 'right'
    elif 135 < angle or angle < -135:
        return 'left'
    elif 45 < angle <= 135:
        return 'down'
    elif -135 <= angle < -45:
        return 'up'


# Thời gian khởi tạo chữ "C" mới
start_time = time.time()

# Mở webcam
cap = cv2.VideoCapture(0)

# Khởi tạo nhận diện bàn tay, cơ thể và khuôn mặt
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
        mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose, \
        mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as face:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Lật khung hình để giống với gương
        frame = cv2.flip(frame, 1)

        # Kiểm tra nếu đạt đủ số lần sau khi kích thước chữ "C" đạt min_size thì hiển thị kết quả
        if min_size_reached and extra_rounds_after_min <= 0:
            # Hiển thị thông báo kết quả trên màn hình
            cv2.putText(frame, "Kết thúc test", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, f"Final Score: {count_correct} / {total_c_created}", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gesture Direction Detection with Pose and Hands", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
                break
            continue

        # Tạo chữ "C" với hướng và kích thước hiện tại
        landolt_c = create_landolt_c_image(current_size, thickness=15, orientation=current_orientation)
        h, w, _ = landolt_c.shape
        y_offset = (frame.shape[0] - h) // 2
        x_offset = (frame.shape[1] - w) // 2
        frame[y_offset:y_offset + h, x_offset:x_offset + w] = landolt_c


        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Nhận diện bàn tay, cơ thể và khuôn mặt
        hand_results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)
        face_results = face.process(image_rgb)
        image_rgb.flags.writeable = True

        # Vẽ các điểm mốc của vai và mắt nếu nhận diện được cơ thể
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_eye = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

            # Vẽ các điểm mốc vai và mắt
            cv2.circle(frame, (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0])), 5,
                       (255, 0, 0), -1)
            cv2.circle(frame, (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])), 5,
                       (255, 0, 0), -1)
            cv2.circle(frame, (int(left_eye.x * frame.shape[1]), int(left_eye.y * frame.shape[0])), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_eye.x * frame.shape[1]), int(right_eye.y * frame.shape[0])), 5, (0, 255, 0),
                       -1)

        # Kiểm tra sự thay đổi vị trí của mũi để phát hiện lắc đầu
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]  # Mũi
                nose_x = nose.x

                # Kiểm tra lắc đầu
                if previous_nose_x is not None and abs(nose_x - previous_nose_x) > shake_threshold:
                    shake_count += 1
                else:
                    shake_count = 0  # Đặt lại đếm nếu không lắc đầu

                previous_nose_x = nose_x

                # Nếu lắc đầu đạt ngưỡng, chuyển sang chữ "C" tiếp theo
                if shake_count >= required_shakes:
                    shake_count = 0
                    current_orientation = random.choice(orientations)
                    current_size = max(current_size - size_decrement, min_size)
                    total_c_created += 1

                    # Kiểm tra giới hạn min_size
                    if current_size <= min_size:
                        min_size_reached = True
                        extra_rounds_after_min -= 1
                    start_time = time.time()  # Đặt lại thời gian cho chữ "C" mới

        detected_orientation = None  # Biến để lưu hướng của tay

        # Kiểm tra hướng chỉ tay
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Xác định cổ tay và đầu ngón trỏ
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Tính toán hướng chỉ tay
                detected_orientation = calculate_hand_direction(wrist, index_finger_tip)

                # Nếu hướng chỉ tay trùng với hướng của chữ "C"
                if detected_orientation == current_orientation:
                    count_correct += 1
                    total_c_created += 1

                    # Nếu đạt kích thước tối thiểu, giảm số lần còn lại
                    if current_size <= min_size:
                        min_size_reached = True
                        extra_rounds_after_min -= 1
                    else:
                        current_size = max(current_size - size_decrement, min_size)  # Giảm kích thước chữ "C"

                    # Đổi hướng ngẫu nhiên cho chữ "C" và đặt lại thời gian
                    current_orientation = random.choice(orientations)
                    start_time = time.time()

        # Hiển thị hướng chỉ tay lên màn hình
        if detected_orientation:
            cv2.putText(frame, f"Hand Direction: {detected_orientation}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị hướng của chữ "C" lên màn hình
        cv2.putText(frame, f"C Direction: {current_orientation}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị số lần chỉ tay đúng / tổng số chữ "C" được tạo ra
        cv2.putText(frame, f"Correct: {count_correct} / {total_c_created}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị khung hình
        cv2.imshow("Gesture Direction Detection with Pose and Hands", frame)

        # Thoát chương trình khi nhấn 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()