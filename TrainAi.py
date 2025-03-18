import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Khởi tạo biến
data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Đọc dữ liệu ảnh
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = cv2.imread(os.path.join(path, a))
            image = cv2.resize(image, (30, 30))
            data.append(image)
            labels.append(i)
        except:
            print(f"Error loading image: {a}")

# Chuyển đổi thành numpy arrays và chuẩn hóa dữ liệu
data = np.array(data) / 255.0  # Chuẩn hóa pixel về khoảng [0,1]
labels = np.array(labels)

print(data.shape, labels.shape)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=43)

# One-hot encoding labels
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Lưu dữ liệu để sử dụng sau này
np.savez_compressed("dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
