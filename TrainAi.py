import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import seaborn as sns
import tensorflow as tf
from utils import preprocess_image

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, img_size)
    return image

# Khởi tạo biến
data, labels = [], []
num_classes = 43
img_size = (30, 30)
cur_path = os.getcwd()
class_counts = Counter()

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
            if image is None or image.shape[0] < 10 or image.shape[1] < 10:
                raise ValueError(f"Ảnh không hợp lệ: {img_name}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess_image(image)
            data.append(image)
            labels.append(i)
            class_counts[i] += 1
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_name}: {e}")

# Kiểm tra cân bằng lớp
print("Số lượng ảnh mỗi lớp:", class_counts)
if len(set(class_counts.values())) > 1:
    print("Cảnh báo: Dữ liệu không cân bằng!")

# Chuyển đổi thành numpy arrays và chuẩn hóa
data = np.array(data, dtype=np.float32) / 255.0
labels = np.array(labels, dtype=np.int32)

print(f"Dataset shape: {data.shape}, Labels shape: {labels.shape}")

# Chia tập dữ liệu
X_temp, X_test, y_temp, y_test = train_test_split(data, labels, test_size=0.15, random_state=43, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=43, stratify=y_temp)

# One-hot encoding
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# Tăng cường dữ liệu
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Xây dựng mô hình
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3), padding='same'),
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
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
csv_logger = CSVLogger("training_log.csv")
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

def lr_schedule(epoch):
    lr = 0.001
    if epoch > 20:
        lr *= 0.5
    if epoch > 40:
        lr *= 0.2
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Huấn luyện mô hình
epochs = 50
batch_size = 64

# Tính class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

history = model.fit(
    data_augmentation.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, csv_logger, checkpoint, lr_scheduler],
    class_weight=class_weights_dict
)

# Đánh giá mô hình
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report
print(classification_report(y_true, y_pred_classes, target_names=[classes[i] for i in range(1, num_classes+1)]))

# Lưu mô hình
model.save("final_model.h5")

# Chuyển sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('final_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Vẽ biểu đồ
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.show()