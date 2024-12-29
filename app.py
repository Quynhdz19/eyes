from flask import Flask, render_template, Response, Flask
import smtplib
from email.mime.text import MIMEText
import cv2
import mediapipe as mp
import numpy as np
import random
import math

app = Flask(__name__)

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# Hàm tính toán hướng chỉ tay
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

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
            mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            if total_c_created >= 30:  # Kiểm tra nếu đạt 30 chữ "C"
                cv2.putText(frame, "Da ket thuc!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.putText(frame, f"Diem dung: {count_correct}/{total_c_created}", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue

            # Tạo hình ảnh chữ "C"
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

            # Chuyển đổi sang RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(image_rgb)
            pose_results = pose.process(image_rgb)

            # Xử lý nhận diện bàn tay và cơ thể
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
                        current_orientation = random.choice(orientations)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"Correct: {count_correct} / {total_c_created}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route xử lý gửi email
@app.route('/send-warning', methods=['POST'])
def send_warning():
    try:
        # Thông tin email
        sender_email = "bkstarstudy@gmail.com"
        sender_password = "yipwdmjnoffovpbb"
        receiver_email = "20020339@vnu.edu.vn"

        # Nội dung email
        subject = "Cảnh báo: Ngồi sai tư thế"
        body = "Bạn đã ngồi sai tư thế hơn 1 giờ trong ngày. Vui lòng điều chỉnh tư thế để bảo vệ sức khỏe!"

        # Tạo email
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        # Gửi email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())

        return "Email cảnh báo đã được gửi.", 200
    except Exception as e:
        print(e)
        return "Lỗi khi gửi email.", 500

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict')
def predic():
    return render_template('index.html')

@app.route('/news')
def news():
    return render_template('new.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)