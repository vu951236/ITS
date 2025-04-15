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

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=43, stratify=labels)

# One-hot encoding labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Tăng cường dữ liệu để cải thiện tổng quát hóa
data_augmentation = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:], padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
csv_logger = CSVLogger("training_log.csv")
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Huấn luyện mô hình
epochs = 25
batch_size = 32

history = model.fit(
    data_augmentation.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr, csv_logger, checkpoint]
)

# Lưu mô hình cuối cùng
model.save("final_model.h5")

# Vẽ biểu đồ huấn luyện
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.show()