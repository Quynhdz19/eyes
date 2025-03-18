import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_pose = YOLO("yolov8n-pose.pt")  # Model nhẹ nhất cho pose
if device == "cuda":
    yolo_pose = yolo_pose.half()  # FP16 trên GPU

# Ngưỡng để đánh giá tư thế
MAX_SPINE_ANGLE = 15  # Góc tối đa cho cột sống (độ)

def calculate_angle(a, b, c):
    """Tính góc giữa 3 điểm (b là điểm giữa)"""
    if any(p is None for p in [a, b, c]):
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_point(landmarks, i):
    """Lấy tọa độ từ keypoints"""
    if i < len(landmarks):
        return landmarks[i][0], landmarks[i][1]
    return None

def analyze_posture(frame):
    """Phân tích tư thế ngồi cho tất cả người trong frame"""
    results = yolo_pose(frame)
    if len(results) == 0 or results[0].keypoints is None:
        cv2.putText(frame, "", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    # Lặp qua tất cả người được phát hiện
    person_count = 0
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # Keypoints của tất cả người trong frame
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes của tất cả người

        for i, (kps, box) in enumerate(zip(keypoints, boxes)):
            person_count += 1
            if len(kps) < 13:
                cv2.putText(frame, f" {person_count}: ",
                            (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            # Lấy các điểm mốc
            left_shoulder = get_point(kps, 5)  # Vai trái
            right_shoulder = get_point(kps, 6)  # Vai phải
            left_hip = get_point(kps, 11)      # Hông trái
            right_hip = get_point(kps, 12)     # Hông phải

            if all([left_shoulder, right_shoulder, left_hip, right_hip]):
                # Tính điểm giữa vai và hông
                mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                                (left_shoulder[1] + right_shoulder[1]) / 2)
                mid_hip = ((left_hip[0] + right_hip[0]) / 2,
                           (left_hip[1] + right_hip[1]) / 2)

                # Tính góc cột sống
                vertical_point = (mid_shoulder[0], mid_shoulder[1] + 100)  # Điểm tham chiếu thẳng đứng
                spine_angle = calculate_angle(mid_shoulder, mid_hip, vertical_point)

                if spine_angle is not None:
                    status = "correct" if spine_angle < MAX_SPINE_ANGLE else "in correct"
                    color = (0, 255, 0) if spine_angle < MAX_SPINE_ANGLE else (0, 0, 255)

                    # Vẽ bounding box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Vẽ keypoints
                    for kp in kps:
                        if kp[0] > 0 and kp[1] > 0:
                            cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)

                    # Ghi trạng thái và góc lên frame
                    cv2.putText(frame, f" {person_count}: {status} ({spine_angle:.2f} degree)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    print(f" {person_count}: {status} ({spine_angle:.2f} degree)")

    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Phân tích tư thế cho tất cả người
        frame = analyze_posture(frame)

        # Hiển thị kết quả
        cv2.imshow("Phân tích tư thế ngồi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy với video của bạn
video_path = "datas/istockphoto-2199170782-640_adpp_is.mp4"
process_video(video_path)