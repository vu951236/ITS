import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Khởi tạo biến
data, labels = [], []
num_classes = 43  # Số lượng lớp
img_size = (30, 30)  # Kích thước ảnh
cur_path = os.getcwd()

# Đọc dữ liệu ảnh
for i in range(num_classes):
    path = os.path.join(cur_path, 'train', str(i))
    if not os.path.exists(path):
        print(f"Warning: Folder {path} does not exist!")
        continue
    images = os.listdir(path)

    for img_name in images:
        img_path = os.path.join(path, img_name)
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {img_name}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
            image = cv2.resize(image, img_size)  # Resize ảnh
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_name}: {e}")

# Chuyển đổi thành numpy arrays và chuẩn hóa dữ liệu
data = np.array(data, dtype=np.float32) / 255.0  # Chuẩn hóa pixel về khoảng [0,1]
labels = np.array(labels, dtype=np.int32)

print(f"Dataset shape: {data.shape}, Labels shape: {labels.shape}")