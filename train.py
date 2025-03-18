import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# 1. Hàm tiền xử lý ảnh
def preprocess_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Chuẩn hóa về [0, 1]
    return img


# 2. Load dữ liệu thật
def load_real_data(image_dir, label_file):
    images = []
    keypoints = []

    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(',')
            image_path = os.path.join(image_dir, data[0])
            if os.path.exists(image_path):
                img = preprocess_image(image_path)
                kp = np.array([float(x) for x in data[1:]]).reshape(-1, 2)  # keypoints: [x, y]
                # Chuẩn hóa keypoints về [0, 1] dựa trên kích thước ảnh gốc
                kp[:, 0] = kp[:, 0] / cv2.imread(image_path).shape[1]  # x / width
                kp[:, 1] = kp[:, 1] / cv2.imread(image_path).shape[0]  # y / height
                images.append(img)
                keypoints.append(kp.flatten())

    return np.array(images), np.array(keypoints)


# 3. Xây dựng mô hình CNN đơn giản
def build_pose_model(input_shape=(64, 64, 3), num_keypoints=6):  # 3 joints x 2 (x, y)
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_keypoints)  # Output: tọa độ keypoints
    ])
    return model


# 4. Train mô hình
def train_model(model, X_train, y_train, epochs=20, batch_size=4):
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return history


# 5. Hiển thị kết quả training
def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.show()


# 6. Dự đoán và hiển thị keypoints
def predict_and_visualize(model, image, image_path):
    pred_keypoints = model.predict(np.expand_dims(image, axis=0))[0]
    pred_keypoints = pred_keypoints.reshape(-1, 2)  # Chuyển về [x, y]

    # Đọc lại ảnh gốc để hiển thị với kích thước gốc
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    height, width = orig_img.shape[:2]

    # Scale keypoints từ [0, 1] về kích thước ảnh gốc
    pred_keypoints[:, 0] = pred_keypoints[:, 0] * width
    pred_keypoints[:, 1] = pred_keypoints[:, 1] * height

    plt.imshow(orig_img)
    plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c='r', s=50)
    plt.title("Predicted Keypoints")
    plt.show()


# 7. Main execution
if __name__ == "__main__":
    # Đường dẫn tới thư mục ảnh và file nhãn
    image_dir = "path/to/your/images"  # Thay bằng đường dẫn thư mục chứa ảnh
    label_file = "path/to/your/labels.txt"  # Thay bằng đường dẫn file nhãn

    # Load dữ liệu
    X, y = load_real_data(image_dir, label_file)

    # Xây dựng và train mô hình
    model = build_pose_model(num_keypoints=y.shape[1])  # Số keypoints dựa trên dữ liệu
    history = train_model(model, X, y)

    # Hiển thị kết quả
    plot_history(history)

    # Dự đoán trên một ảnh mẫu
    sample_idx = 0  # Chọn ảnh đầu tiên để demo
    sample_image = X[sample_idx]
    sample_image_path = os.path.join(image_dir, os.listdir(image_dir)[sample_idx])
    predict_and_visualize(model, sample_image, sample_image_path)

    # Lưu mô hình
    model.save("simple_pose_model.h5")