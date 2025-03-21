from collections import deque

from flask import Flask, render_template, Response, Flask, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import random
import math
from ultralytics import YOLO  # Thư viện YOLOv8
import torch

app = Flask(__name__)


model = YOLO("yolov8n.pt")  # YOLOv8 nhận diện người


# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

MIN_FACE_DISTANCE = 50   # Cận (pixels)
MAX_FACE_DISTANCE = 250  # Xa (pixels)

def calculate_hand_direction(wrist, index_finger_tip):
    dx = index_finger_tip.x - wrist.x
    dy = index_finger_tip.y - wrist.y
    angle = math.degrees(math.atan2(dy, dx))

    if -45 <= angle <= 45:
        return 'right'
    elif 135 < angle or angle < -135:
        return 'left'
    elif 45 < angle <= 135:
        return 'down'
    elif -135 <= angle < -45:
        return 'up'

# Hàm tính khoảng cách giữa hai điểm
def calculate_distance(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

# Hàm phát hiện lắc đầu với Face Mesh
def detect_head_shake(face_landmarks, prev_head_pos, frame_count, shake_threshold=0.8, frame_window=20, smoothing_window=20):
    if face_landmarks is None or frame_count < frame_window:
        return False, None

    # Lấy các điểm chính trên khuôn mặt
    left_ear = face_landmarks.landmark[33]    # Tai trái
    right_ear = face_landmarks.landmark[263]  # Tai phải
    nose_tip = face_landmarks.landmark[1]     # Đỉnh mũi
    left_eye = face_landmarks.landmark[159]   # Mắt trái
    right_eye = face_landmarks.landmark[386]  # Mắt phải

    # Tính góc yaw (nghiêng ngang)
    dx_yaw = right_ear.x - left_ear.x
    dy_yaw = right_ear.y - left_ear.y
    current_yaw = math.degrees(math.atan2(dy_yaw, dx_yaw))

    # Tính góc pitch (ngửa/úp)
    eye_center_x = (left_eye.x + right_eye.x) / 2
    eye_center_y = (left_eye.y + right_eye.y) / 2
    eye_center_z = (left_eye.z + right_eye.z) / 2
    dx_pitch = nose_tip.x - eye_center_x
    dy_pitch = nose_tip.y - eye_center_y
    current_pitch = math.degrees(math.atan2(dy_pitch, -nose_tip.z))

    # Tính góc roll (nghiêng dọc)
    dy_roll = right_eye.y - left_eye.y
    dx_roll = right_eye.x - left_eye.x
    current_roll = math.degrees(math.atan2(dy_roll, dx_roll))

    # Tạo bộ nhớ tạm cho smoothing
    if not hasattr(detect_head_shake, 'yaw_history'):
        detect_head_shake.yaw_history = deque(maxlen=smoothing_window)
        detect_head_shake.pitch_history = deque(maxlen=smoothing_window)
        detect_head_shake.roll_history = deque(maxlen=smoothing_window)

    detect_head_shake.yaw_history.append(current_yaw)
    detect_head_shake.pitch_history.append(current_pitch)
    detect_head_shake.roll_history.append(current_roll)

    # Tính giá trị trung bình để giảm nhiễu
    smoothed_yaw = np.mean(detect_head_shake.yaw_history)
    smoothed_pitch = np.mean(detect_head_shake.pitch_history)
    smoothed_roll = np.mean(detect_head_shake.roll_history)

    if prev_head_pos is None:
        return False, (smoothed_yaw, smoothed_pitch, smoothed_roll)

    prev_yaw, prev_pitch, prev_roll = prev_head_pos

    # Tính độ thay đổi của từng góc
    yaw_change = abs(smoothed_yaw - prev_yaw)
    pitch_change = abs(smoothed_pitch - prev_pitch)
    roll_change = abs(smoothed_roll - prev_roll)

    # Debug để kiểm tra giá trị thay đổi
    print(f"Yaw change: {yaw_change:.2f}, Pitch change: {pitch_change:.2f}, Roll change: {roll_change:.2f}")

    # Phát hiện lắc đầu với ngưỡng cao hơn
    if yaw_change > shake_threshold or pitch_change > shake_threshold or roll_change > shake_threshold:
        return True, (smoothed_yaw, smoothed_pitch, smoothed_roll)

    return False, (smoothed_yaw, smoothed_pitch, smoothed_roll)

# Hàm tạo khung hình video
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    # Cài đặt ban đầu
    orientations = ['right', 'left', 'up', 'down']
    initial_size = 120
    min_size = 20
    size_decrement = 10
    count_correct = 0
    total_c_created = 0
    current_orientation = random.choice(orientations)
    current_size = initial_size

    # Biến kiểm tra
    prev_head_pos = None
    frame_count = 0
    shake_cooldown = 0
    face_missing_cooldown = 0
    action_locked = False
    prev_detected_orientation = None
    head_shaking = False
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
            mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose, \
            mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection, \
            mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_faces=1) as face_mesh:

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Kết thúc sau 30 chữ C
            if total_c_created >= 30:
                cv2.putText(frame, "Da ket thuc!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.putText(frame, f"Diem dung: {count_correct}/{total_c_created}", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue

            # Chuyển đổi sang RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(image_rgb)
            pose_results = pose.process(image_rgb)
            face_results = face_detection.process(image_rgb)
            face_mesh_results = face_mesh.process(image_rgb)

            # Kiểm tra nhận diện khuôn mặt
            face_detected = False
            if face_results.detections:
                face_detected = True
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (255, 0, 0), 2)

            # Nếu không phát hiện khuôn mặt
            if not face_detected:
                face_missing_cooldown = 30
                cv2.putText(frame, "No face detected - Skipping C", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Giảm thời gian cooldown
            if face_missing_cooldown > 0:
                face_missing_cooldown -= 1
            if shake_cooldown > 0:
                shake_cooldown -= 1

            # Kiểm tra lắc đầu
            is_shaking = False
            if face_mesh_results.multi_face_landmarks and face_detected and face_missing_cooldown == 0:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    is_shaking, prev_head_pos = detect_head_shake(face_landmarks, prev_head_pos, frame_count)
                    if is_shaking and shake_cooldown == 0 and not head_shaking:
                        # Chỉ kích hoạt khi bắt đầu lắc đầu (head_shaking = False)
                        head_shaking = True
                        shake_cooldown = 60  # Tăng cooldown lên 2 giây (60 frame ở 30 FPS)
                        total_c_created += 1
                        current_size = max(current_size - size_decrement, min_size)
                        available_orientations = [o for o in orientations if o != current_orientation]
                        current_orientation = random.choice(available_orientations)
                        cv2.putText(frame, "Head shake - Next C", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    elif not is_shaking and head_shaking and shake_cooldown > 0:
                        # Giữ trạng thái head_shaking cho đến khi cooldown hết
                        pass
                    elif not is_shaking and shake_cooldown == 0:
                        # Reset trạng thái khi đầu ổn định và cooldown hết
                        head_shaking = False

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Vẽ chữ "C" nếu có khuôn mặt
            if face_missing_cooldown == 0:
                canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255
                center = (150, 150)
                start_angle, end_angle = {
                    'right': (45, 315),
                    'left': (225, 495),
                    'up': (-45, 225),
                    'down': (135, 405)
                }[current_orientation]

                cv2.ellipse(canvas, center, (current_size, current_size), 0, start_angle, end_angle, (0, 0, 0), 15)
                h, w, _ = canvas.shape
                y_offset = (frame.shape[0] - h) // 2
                x_offset = (frame.shape[1] - w) // 2
                frame[y_offset:y_offset + h, x_offset:x_offset + w] = canvas

            # Xử lý nhận diện bàn tay
            detected_orientation = None
            if hand_results.multi_hand_landmarks and face_missing_cooldown == 0:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    detected_orientation = calculate_hand_direction(wrist, index_finger_tip)
                    break

            # Xử lý hành động chỉ tay
            if detected_orientation is not None and not action_locked and shake_cooldown == 0:
                if detected_orientation != prev_detected_orientation:
                    prev_detected_orientation = detected_orientation
                    action_locked = True
                    print(f"Detected Orientation: {detected_orientation}, Current C Orientation: {current_orientation}")

                    if detected_orientation == current_orientation:
                        count_correct += 1
                        total_c_created += 1
                        current_size = max(current_size - size_decrement, min_size)
                        current_orientation = random.choice(orientations)
                        cv2.putText(frame, "Correct!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        current_size = max(current_size - size_decrement, min_size)
                        current_orientation = random.choice(orientations)
                        total_c_created += 1
                        cv2.putText(frame, "Wrong!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Mở khóa khi không phát hiện tay
            if detected_orientation is None:
                action_locked = False
                prev_detected_orientation = None

            # Hiển thị điểm số
            cv2.putText(frame, f"Correct: {count_correct} / {total_c_created}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
# **Khởi tạo MediaPipe Pose**
yolo_pose = YOLO("yolov8n-pose.pt")  # Load mô hình YOLOv8 Pose
MAX_SPINE_ANGLE = 3
MAX_HEAD_ANGLE = 10
posture_data = {"status": "Analyzing...", "spine_angle": "--", "head_angle": "--", "confidence": "--"}
def calculate_spine_angle(shoulder, hip):
    """Tính góc giữa vector cột sống và trục thẳng đứng"""
    spine_vector = np.array([hip[0] - shoulder[0], hip[1] - shoulder[1]])
    vertical_vector = np.array([0, 1])

    dot_product = np.dot(spine_vector, vertical_vector)
    norm_spine = np.linalg.norm(spine_vector)
    norm_vertical = np.linalg.norm(vertical_vector)

    if norm_spine == 0:
        return None

    cos_theta = dot_product / (norm_spine * norm_vertical)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def calculate_head_angle(eye, ear):
    """Tính góc giữa vector mắt-tai và trục ngang"""
    head_vector = np.array([ear[0] - eye[0], ear[1] - eye[1]])
    horizontal_vector = np.array([1, 0])  # Trục ngang

    dot_product = np.dot(head_vector, horizontal_vector)
    norm_head = np.linalg.norm(head_vector)
    norm_horizontal = np.linalg.norm(horizontal_vector)

    if norm_head == 0:
        return None

    cos_theta = dot_product / (norm_head * norm_horizontal)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def get_point(keypoints, index):
    """Get (x, y) coordinates from keypoints if valid"""
    x, y = keypoints[index]
    if x > 0 and y > 0:
        return (int(x), int(y))
    return None

def detect_posture(frame):
    """Detect posture using YOLO Pose and return processed frame with data"""
    global posture_data
    results = yolo_pose(frame)
    output_data = []

    if len(results) == 0 or results[0].keypoints is None:
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        posture_data = {"status": "Not detected", "spine_angle": "--", "head_angle": "--", "confidence": "--"}
        output_data.append(posture_data)
        return frame, output_data

    person_count = 0
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        for i, (kps, box) in enumerate(zip(keypoints, boxes)):
            person_count += 1
            if len(kps) < 17:  # YOLOv8 Pose cần 17 keypoints
                continue

            # Lấy các điểm cho cột sống
            left_shoulder = get_point(kps, 5)
            right_shoulder = get_point(kps, 6)
            left_hip = get_point(kps, 11)
            right_hip = get_point(kps, 12)

            # Lấy các điểm cho đầu
            left_eye = get_point(kps, 1)
            left_ear = get_point(kps, 3)
            right_eye = get_point(kps, 2)
            right_ear = get_point(kps, 4)

            if all([left_shoulder, right_shoulder, left_hip, right_hip]) and any([left_eye and left_ear, right_eye and right_ear]):
                # Tính điểm giữa vai và hông
                mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                                (left_shoulder[1] + right_shoulder[1]) / 2)
                mid_hip = ((left_hip[0] + right_hip[0]) / 2,
                           (left_hip[1] + right_hip[1]) / 2)

                # Tính góc cột sống
                spine_angle = calculate_spine_angle(mid_shoulder, mid_hip)

                # Tính góc đầu (dùng bên trái hoặc bên phải tùy dữ liệu có sẵn)
                head_angle = None
                if left_eye and left_ear:
                    head_angle = calculate_head_angle(left_eye, left_ear)
                elif right_eye and right_ear:
                    head_angle = calculate_head_angle(right_eye, right_ear)

                if spine_angle is not None and head_angle is not None:
                    # Tư thế đúng: cả góc cột sống và đầu đều trong ngưỡng
                    is_spine_correct = spine_angle < MAX_SPINE_ANGLE
                    is_head_correct = head_angle < MAX_HEAD_ANGLE
                    status = "Correct" if is_spine_correct and is_head_correct else "Incorrect"
                    color = (0, 255, 0) if status == "Correct" else (0, 0, 255)

                    # Vẽ hộp và keypoints
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    for kp in kps:
                        if kp[0] > 0 and kp[1] > 0:
                            cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)

                    # Hiển thị thông tin
                    cv2.putText(frame, f"Person {person_count}: {status} (Spine: {spine_angle:.2f}°, Head: {head_angle:.2f}°)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    confidence = float(result.boxes.conf[i].cpu().numpy()) * 100 if result.boxes.conf is not None else 100
                    posture_data = {
                        "status": status,
                        "spine_angle": round(spine_angle, 2),
                        "head_angle": round(head_angle, 2),
                        "confidence": round(confidence, 2)
                    }
                    output_data.append(posture_data)

    return frame, output_data

def detect_body_with_opencv(frame):
    """Nhận diện người bằng OpenCV DNN khi MediaPipe không hoạt động"""
    net = cv2.dnn.readNetFromCaffe(
        "models/deploy.prototxt",
        "models/res10_300x300_ssd_iter_140000.caffemodel"
    )

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # Chỉ lấy những phát hiện có độ tin cậy cao
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Vẽ khung nhận diện
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {round(confidence * 100, 1)}%", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame



def generate_frames_index():
    """Luồng video phát hiện tư thế"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = detect_posture(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()




# nhận diện lớp học
def calculate_angle(a, b, c):
    """Tính góc giữa 3 điểm (b là điểm giữa)"""
    if any(p is None for p in [a, b, c]):
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def process_video(video_path):
    """Xử lý video để nhận diện tư thế"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video!")
        return None

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # Nhận diện người bằng YOLO
            results = model(frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]

                    # Crop phần người để nhận diện tư thế
                    person_img = frame[y1:y2, x1:x2]

                    # Nhận diện tư thế bằng MediaPipe Pose
                    image_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(image_rgb)

                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        landmarks = pose_results.pose_landmarks.landmark

                        def get_point(landmark):
                            if landmark.visibility > 0.5:
                                return int(landmark.x * (x2 - x1)) + x1, int(landmark.y * (y2 - y1)) + y1
                            return None

                        left_shoulder = get_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
                        right_shoulder = get_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                        left_hip = get_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
                        right_hip = get_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])

                        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                                        (left_shoulder[1] + right_shoulder[1]) // 2) if left_shoulder and right_shoulder else None
                        mid_hip = ((left_hip[0] + right_hip[0]) // 2,
                                   (left_hip[1] + right_hip[1]) // 2) if left_hip and right_hip else None

                        # Tính góc cột sống
                        if mid_shoulder and mid_hip and left_hip:
                            spine_angle = calculate_angle(mid_shoulder, mid_hip, left_hip)

                            # Đánh giá tư thế
                            status_text = "✅ Đúng" if spine_angle < 3 else "❌ Sai"
                            color = (0, 255, 0) if spine_angle < 3 else (0, 0, 255)

                            cv2.putText(frame, f"{status_text} - {round(spine_angle, 2)}°",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Chạy kiểm tra tư thế


# Route xử lý gửi email
# @app.route('/send-warning', methods=['POST'])
# def send_warning():
#     try:
#         # Thông tin email
#         # sender_email = "bkstarstudy@gmail.com"
#         # sender_password = "yipwdmjnoffovpbb"
#         # receiver_email = "20020339@vnu.edu.vn"
#
#         # Nội dung email
#         # subject = "Cảnh báo: Ngồi sai tư thế"
#         # body = "Bạn đã ngồi sai tư thế hơn 1 giờ trong ngày. Vui lòng điều chỉnh tư thế để bảo vệ sức khỏe!"
#         #
#         # # Tạo email
#         # msg = MIMEText(body)
#         # msg['Subject'] = subject
#         # msg['From'] = sender_email
#         # msg['To'] = receiver_email
#         #
#         # # Gửi email
#         # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#         #     server.login(sender_email, sender_password)
#         #     server.sendmail(sender_email, receiver_email, msg.as_string())
#         #
#         # return "Email cảnh báo đã được gửi.", 200
#     except Exception as e:
#         print(e)
#         return "Lỗi khi gửi email.", 500




device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_pose = YOLO("yolov8n-pose.pt").to(device)
if device == "cuda":
    yolo_pose = yolo_pose.half()

MAX_SPINE_ANGLE = 15

def calculate_angle(a, b, c):
    if any(p is None for p in [a, b, c]):
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_point(landmarks, i):
    if i < len(landmarks):
        return landmarks[i][0], landmarks[i][1]
    return None

def analyze_posture(frame):
    results = yolo_pose(frame)
    if len(results) == 0 or results[0].keypoints is None:
        cv2.putText(frame, "No person detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    person_count = 0
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        for i, (kps, box) in enumerate(zip(keypoints, boxes)):
            person_count += 1
            if len(kps) < 13:
                continue

            left_shoulder = get_point(kps, 5)
            right_shoulder = get_point(kps, 6)
            left_hip = get_point(kps, 11)
            right_hip = get_point(kps, 12)

            if all([left_shoulder, right_shoulder, left_hip, right_hip]):
                mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                              (left_shoulder[1] + right_shoulder[1]) / 2)
                mid_hip = ((left_hip[0] + right_hip[0]) / 2,
                         (left_hip[1] + right_hip[1]) / 2)

                vertical_point = (mid_shoulder[0], mid_shoulder[1] + 100)
                spine_angle = calculate_angle(mid_shoulder, mid_hip, vertical_point)

                if spine_angle is not None:
                    status = "correct" if spine_angle < MAX_SPINE_ANGLE else "incorrect"
                    color = (0, 255, 0) if spine_angle < MAX_SPINE_ANGLE else (0, 0, 255)

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    for kp in kps:
                        if kp[0] > 0 and kp[1] > 0:
                            cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)

                    cv2.putText(frame, f"Person {person_count}: {status} ({spine_angle:.2f}°)",
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def generate_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = analyze_posture(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def generate_uploaded_feed():
    global uploaded_video_path
    if not uploaded_video_path:
        return

    cap = cv2.VideoCapture(uploaded_video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = analyze_posture(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

uploaded_video_path = None


@app.route('/')
def index():
    return render_template('home.html')

@app.route("/posture_data")
def get_posture_status():
    """API trả về trạng thái tư thế"""
    return jsonify(posture_data)

@app.route('/predict')
def predic():
    return render_template('index.html')

@app.route('/news')
def news():
    return render_template('new.html')


@app.route('/class_predict')
def class_index():
    return render_template('class.html')

def generate_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = analyze_posture(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def generate_uploaded_feed():
    global uploaded_video_path
    if not uploaded_video_path:
        return

    cap = cv2.VideoCapture(uploaded_video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = analyze_posture(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

uploaded_video_path = None

@app.route('/class_feed')
def class_feed():
    return render_template('class_feed.html')

@app.route('/video_class_feed')
def video_class_feed():
    return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global uploaded_video_path
    video = request.files['video']
    uploaded_video_path = f"uploads/{video.filename}"
    video.save(uploaded_video_path)
    return "Video uploaded successfully", 200


@app.route('/video_feed_index')
def video_feed_index():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera!")
            return

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame, data = detect_posture(frame)
            global posture_data
            posture_data = data[0] if data else {"status": "Not detected", "spine_angle": "--", "confidence": "--"}

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_class_feed_upload')
def video_class_feed_upload():
    return Response(generate_uploaded_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
