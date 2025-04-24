import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Thiết lập thông số
num_classes = 43
img_size = (30, 30)
data = []
labels = []

# Lấy đường dẫn gốc
base_path = os.getcwd()

# Đọc dữ liệu ảnh từ thư mục
for class_id in range(num_classes):
    folder_path = os.path.join(base_path, 'train', str(class_id))
    if not os.path.isdir(folder_path):
        print(f"Không tìm thấy thư mục: {folder_path}")
        continue

    for image_file in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, image_file)
            image = cv2.imread(img_path)
            if image is None:
                raise Exception("Không thể đọc ảnh")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, img_size)
            data.append(image)
            labels.append(class_id)
        except Exception as err:
            print(f"Lỗi ở {img_path}: {err}")

# Chuyển sang mảng numpy và chuẩn hóa
data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)

print(f"Đã tải: {len(data)} ảnh")

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Khởi tạo bộ tạo ảnh tăng cường
augmentor = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Xây dựng mô hình CNN
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Biên dịch mô hình
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Các callback trong quá trình huấn luyện
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1),
    CSVLogger("log.csv"),
    ModelCheckpoint("model_best.h5", monitor='val_accuracy', save_best_only=True)
]

# Huấn luyện mô hình
EPOCHS = 25
BATCH_SIZE = 32

history = model.fit(
    augmentor.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=callbacks
)

# Lưu mô hình
model.save("final_model.h5")

# Vẽ biểu đồ kết quả
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Biểu đồ độ chính xác')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Biểu đồ hàm mất mát')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
