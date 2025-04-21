import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
import threading
import sys
import psutil
from gtts import gTTS
import pygame
import tempfile
sys.stdout.reconfigure(encoding='utf-8')

# Tắt cảnh báo oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Danh sách các loại biển báo
classes = {
    1: 'Giới hạn tốc độ 20km/h',
    2: 'Giới hạn tốc độ 30km/h',
    3: 'Giới hạn tốc độ 50km/h',
    4: 'Giới hạn tốc độ 60km/h',
    5: 'Giới hạn tốc độ 70km/h',
    6: 'Giới hạn tốc độ 80km/h',
    7: 'Hết giới hạn tốc độ 80km/h',
    8: 'Giới hạn tốc độ 100km/h',
    9: 'Giới hạn tốc độ 120km/h',
    10: 'Cấm vượt',
    11: 'Cấm vượt xe trên 3.5 tấn',
    12: 'Được ưu tiên tại ngã tư',
    13: 'Đường ưu tiên',
    14: 'Nhường đường',
    15: 'Dừng lại',
    16: 'Cấm phương tiện',
    17: 'Cấm xe trên 3.5 tấn',
    18: 'Cấm vào',
    19: 'Chú ý nguy hiểm',
    20: 'Đường cong nguy hiểm bên trái',
    21: 'Đường cong nguy hiểm bên phải',
    22: 'Đường cong đôi',
    23: 'Đường xóc',
    24: 'Đường trơn trượt',
    25: 'Đường hẹp bên phải',
    26: 'Công trình đang thi công',
    27: 'Tín hiệu giao thông',
    28: 'Khu vực có người đi bộ',
    29: 'Khu vực có trẻ em qua đường',
    30: 'Khu vực có xe đạp băng qua',
    31: 'Cảnh báo băng tuyết',
    32: 'Cảnh báo động vật hoang dã',
    33: 'Hết giới hạn tốc độ và cấm vượt',
    34: 'Rẽ phải phía trước',
    35: 'Rẽ trái phía trước',
    36: 'Chỉ được đi thẳng',
    37: 'Đi thẳng hoặc rẽ phải',
    38: 'Đi thẳng hoặc rẽ trái',
    39: 'Đi bên phải',
    40: 'Đi bên trái',
    41: 'Vòng xuyến bắt buộc',
    42: 'Hết lệnh cấm vượt',
    43: 'Hết lệnh cấm vượt xe trên 3.5 tấn'
}

class TrafficSignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận Diện Biển Báo Giao Thông")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2C3E50')

        # Lấy thông tin process hiện tại
        self.process = psutil.Process()

        # Tải mô hình Keras
        try:
            print("Đang tải mô hình Keras...")
            model_path = os.path.abspath('final_model.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path}")
            self.model = load_model(model_path, compile=False)
            self.model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
            print("Đã tải mô hình Keras thành công!")
        except Exception as e:
            error_msg = f"Lỗi khi tải mô hình: {str(e)}"
            print(error_msg)
            messagebox.showerror("Lỗi", error_msg)
            raise

        # Frame chính
        main_frame = tk.Frame(root, bg='#2C3E50')
        main_frame.pack(expand=True, fill='both', padx=30, pady=30)

        # Tiêu đề
        title_label = tk.Label(main_frame,
                              text="HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG",
                              font=("Arial", 24, "bold"),
                              bg='#2C3E50', fg='#ECF0F1')
        title_label.pack(fill='x', pady=(0, 20))

        # Frame nội dung
        content_frame = tk.Frame(main_frame, bg='#2C3E50')
        content_frame.pack(fill='both', expand=True)

        # Frame camera
        camera_frame = tk.Frame(content_frame, bg='#34495E', relief='solid', borderwidth=2)
        camera_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=(0, 20))

        self.camera_label = tk.Label(camera_frame, bg='#34495E')
        self.camera_label.pack(pady=10, padx=10, expand=True)

        # Nút điều khiển
        control_frame = tk.Frame(camera_frame, bg='#34495E')
        control_frame.pack(pady=10)

        self.is_running = False
        self.start_stop_button = tk.Button(control_frame,
                                          text="BẮT ĐẦU CAMERA",
                                          command=self.toggle_camera,
                                          font=("Arial", 12, "bold"),
                                          width=20, bg='#27AE60', fg='white',
                                          relief='flat', cursor='hand2')
        self.start_stop_button.pack(pady=10)

        # Frame kết quả
        result_frame = tk.Frame(content_frame, bg='#34495E', relief='solid', borderwidth=2)
        result_frame.pack(side=tk.RIGHT, fill='both', expand=True)

        self.result_title = tk.Label(result_frame,
                                    text="KẾT QUẢ NHẬN DIỆN",
                                    font=("Arial", 18, "bold"),
                                    bg='#34495E', fg='#ECF0F1')
        self.result_title.pack(pady=20)

        self.result_label = tk.Label(result_frame,
                                    text="Chưa phát hiện biển báo",
                                    font=("Arial", 14), wraplength=400,
                                    justify=tk.LEFT, bg='#34495E', fg='#ECF0F1')
        self.result_label.pack(pady=20)

        # Trạng thái xử lý
        self.status_label = tk.Label(result_frame,
                                    text="",
                                    font=("Arial", 12), bg='#34495E', fg='#F1C40F')
        self.status_label.pack(pady=10)

        # Frame hiển thị thông số hiệu suất
        performance_frame = tk.Frame(result_frame, bg='#34495E')
        performance_frame.pack(pady=10, fill='x', padx=20)

        tk.Label(performance_frame,
                 text="THÔNG SỐ HIỆU SUẤT",
                 font=("Arial", 14, "bold"),
                 bg='#34495E', fg='#ECF0F1').pack(anchor='w', pady=(10, 5))

        self.fps_label = tk.Label(performance_frame,
                                 text="FPS: 0.0",
                                 font=("Arial", 12),
                                 bg='#34495E', fg='#2ECC71',
                                 justify=tk.LEFT)
        self.fps_label.pack(anchor='w')

        self.cpu_label = tk.Label(performance_frame,
                                 text="CPU: 0.0%",
                                 font=("Arial", 12),
                                 bg='#34495E', fg='#2ECC71',
                                 justify=tk.LEFT)
        self.cpu_label.pack(anchor='w')

        self.ram_label = tk.Label(performance_frame,
                                 text="RAM: 0.0 MB",
                                 font=("Arial", 12),
                                 bg='#34495E', fg='#2ECC71',
                                 justify=tk.LEFT)
        self.ram_label.pack(anchor='w')

        # Lịch sử kết quả
        self.history = []
        self.history_label = tk.Label(result_frame,
                                     text="Lịch sử:\n",
                                     font=("Arial", 10), wraplength=400,
                                     justify=tk.LEFT, bg='#34495E', fg='#BDC3C7')
        self.history_label.pack(pady=10)

        # Hướng dẫn
        instruction_text = """
Hướng dẫn sử dụng:
1. Nhấn "BẮT ĐẦU CAMERA" để khởi động
2. Đặt biển báo trước camera
3. Giữ biển báo ổn định và đủ ánh sáng
4. Kết quả hiển thị tự động
5. Nhấn "DỪNG CAMERA" để tắt
        """
        tk.Label(result_frame, text=instruction_text, font=("Arial", 10),
                 bg='#34495E', fg='#BDC3C7', justify=tk.LEFT).pack(pady=10)

        # Khởi tạo camera
        self.cap = None
        self.after_id = None
        self.frame_buffer = None
        self.last_prediction_time = 0
        self.prediction_interval = 1500 # 1.5 giây/1 dự đoán
        self.confidence_threshold = 0.8

        # Biến cho tính FPS
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        # Template matching
        self.templates_dir = "templates"
        self.templates = {}
        self.load_templates()

    def load_templates(self):
        try:
            if not os.path.exists(self.templates_dir):
                os.makedirs(self.templates_dir)
                print("Thư mục templates đã được tạo")
                return
            for sign_id in classes:
                template_path = os.path.join(self.templates_dir, f"{sign_id}.jpg")
                if os.path.exists(template_path):
                    template = cv2.imread(template_path)
                    if template is not None:
                        template = cv2.resize(template, (30, 30))
                        self.templates[sign_id] = template
                        print(f"Đã tải template cho biển báo {classes[sign_id]}")
        except Exception as e:
            print(f"Lỗi khi tải templates: {str(e)}")

    def speak_vietnamese(self, text):
        try:
            tts = gTTS(text=text, lang='vi')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                temp_filename = tf.name
            tts.save(temp_filename)

            pygame.mixer.init()
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.music.unload()
            os.remove(temp_filename)
        except Exception as e:
            print(f"Lỗi khi phát âm thanh tiếng Việt: {e}")

    def preprocess_image(self, image):
        try:
            # Resize ngay từ đầu để giảm tải
            image = cv2.resize(image, (30, 30))
            # Chuyển đổi màu 1 lần 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray)
            # image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # Fixed syntax error
            # image = cv2.resize(image, (30, 30))

            # Chuẩn hóa
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=0)
            return image
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh: {str(e)}")
            raise

    def template_matching(self, frame, predicted_class):
        try:
            if predicted_class not in self.templates:
                return True
            template = self.templates[predicted_class]
            frame_resized = cv2.resize(frame, (30, 30))
            result = cv2.matchTemplate(frame_resized, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            return max_val > 0.8
        except Exception as e:
            print(f"Lỗi khi so sánh template: {str(e)}")
            return True

    def predict_sign(self, image):
        try:
            processed_image = self.preprocess_image(image)
            prediction = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(prediction[0]) + 1
            confidence = prediction[0][predicted_class - 1]
            if confidence < self.confidence_threshold or not self.template_matching(image, predicted_class):
                return None, 0
            return predicted_class, confidence
        except Exception as e:
            print(f"Lỗi khi dự đoán: {str(e)}")
            return None, 0

    def refresh_camera(self):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Không thể kết nối camera")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Không thể đọc frame")
                return True
            except Exception as e:
                print(f"Lỗi khi làm mới camera (lần {attempt+1}): {str(e)}")
                time.sleep(1)
        messagebox.showerror("Lỗi", "Không thể kết nối camera sau nhiều lần thử")
        return False

    def toggle_camera(self):
        if not self.is_running:
            if self.refresh_camera():
                self.is_running = True
                self.start_stop_button.config(text="DỪNG CAMERA", bg='#f44336')
                self.last_fps_time = time.time()
                self.frame_count = 0
                self.update_camera()

                self.process = psutil.Process()  # Lấy đối tượng tiến trình

                self.root.after(1000, self.update_performance_metrics)
        else:
            self.stop_camera()

    def stop_camera(self):
        self.is_running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame_buffer = None
        self.last_prediction_time = 0
        self.start_stop_button.config(text="BẮT ĐẦU CAMERA", bg='#27AE60')
        self.camera_label.config(image='')
        self.result_label.config(text="Chưa phát hiện biển báo")
        self.status_label.config(text="")
        self.fps_label.config(text="FPS: 0.0")
        self.cpu_label.config(text="CPU: 0.0%")
        self.ram_label.config(text="RAM: 0.0 MB")
        self.history = []
        self.history_label.config(text="Lịch sử:\n")

    def update_camera(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self.frame_buffer = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_display = cv2.resize(frame_rgb, (640, 480))
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_display))
                self.camera_label.config(image=photo)
                self.camera_label.image = photo
                threading.Thread(target=self.process_prediction, args=(frame,), daemon=True).start()
                
                # Cập nhật FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                    self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            
            self.after_id = self.root.after(33, self.update_camera) # tần suất cập nhật giao diện

    def update_performance_metrics(self):
        if self.is_running:
            try:
                # Lấy CPU của chương trình
                cpu_usage = self.process.cpu_percent(interval=None) / psutil.cpu_count()

                # Lấy RAM của tiến trình (bộ nhớ đang sử dụng)
                memory_info = self.process.memory_info()
                ram_usage_mb = memory_info.rss / (1024 * 1024)  # Chuyển từ bytes sang MB

                # Cập nhật giao diện
                self.cpu_label.config(text=f"CPU: {cpu_usage:.1f}%")
                self.ram_label.config(text=f"RAM: {ram_usage_mb:.1f} MB")
            except Exception as e:
                print(f"Lỗi khi lấy thông số hiệu suất: {str(e)}")
                self.cpu_label.config(text="CPU: N/A")
                self.ram_label.config(text="RAM: N/A")
        
        # Lên lịch cập nhật tiếp theo
        self.root.after(1000, self.update_performance_metrics)

    def process_prediction(self, frame):
        current_time = time.time() * 1000
        if current_time - self.last_prediction_time >= self.prediction_interval:
            self.root.after(0, lambda: self.status_label.config(text="Đang xử lý..."))
            try:
                predicted_class, confidence = self.predict_sign(frame)
                if predicted_class is not None:
                    sign_name = classes[predicted_class]
                    result_text = f"Biển báo: {sign_name}\nĐộ tin cậy: {confidence:.2%}"
                    self.history.append(result_text)
                    if len(self.history) > 5:
                        self.history.pop(0)
                    history_text = "Lịch sử:\n" + "\n".join(self.history)
                    self.root.after(0, lambda: self.result_label.config(text=result_text, fg='#2ECC71'))
                    self.root.after(0, lambda: self.history_label.config(text=history_text))
                    threading.Thread(target=self.speak_vietnamese, args=(sign_name,), daemon=True).start()
                else:
                    self.root.after(0, lambda: self.result_label.config(text="Chưa phát hiện biển báo", fg='#ECF0F1'))
                self.last_prediction_time = current_time
            except Exception as e:
                print(f"Lỗi khi dự đoán: {str(e)}")
                self.root.after(0, lambda: self.result_label.config(text="Lỗi khi dự đoán", fg='#E74C3C'))
            self.root.after(0, lambda: self.status_label.config(text=""))
            try:
                cpu_usage = self.process.cpu_percent(interval=0.1)
                memory_info = self.process.memory_info()
                ram_usage_mb = memory_info.rss / (1024 * 1024)
                # print(f"FPS: {self.fps:.1f} | CPU: {cpu_usage:.1f}% | RAM: {ram_usage_mb:.1f} MB")
            except Exception as e:
                print(f"Lỗi khi ghi log hiệu suất: {str(e)}")

    def __del__(self):
        self.stop_camera()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignGUI(root)
    root.mainloop()