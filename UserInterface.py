import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
import tensorflow as tf
from datetime import datetime

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
        self.root.configure(bg='#2C3E50')  # Màu nền tối hiện đại
        
        try:
            print("Đang tải mô hình...")
            model_path = os.path.abspath('final_model.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path}")
            
            # Tải mô hình
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile(optimizer='adam', 
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])
            
            print("Đã tải mô hình thành công!")
            
        except Exception as e:
            error_msg = f"Lỗi khi tải mô hình: {str(e)}"
            print(error_msg)
            messagebox.showerror("Lỗi", error_msg)
            raise
        
        # Frame chính với padding
        main_frame = tk.Frame(root, bg='#2C3E50')
        main_frame.pack(expand=True, fill='both', padx=30, pady=30)
        
        # Tiêu đề chương trình
        title_frame = tk.Frame(main_frame, bg='#2C3E50')
        title_frame.pack(fill='x', pady=(0, 20))
        
        title_label = tk.Label(title_frame,
                             text="HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG",
                             font=("Arial", 24, "bold"),
                             bg='#2C3E50',
                             fg='#ECF0F1')
        title_label.pack()
        
        # Frame cho camera và kết quả
        content_frame = tk.Frame(main_frame, bg='#2C3E50')
        content_frame.pack(fill='both', expand=True)
        
        # Frame cho camera
        camera_frame = tk.Frame(content_frame, bg='#34495E', relief='solid', borderwidth=2)
        camera_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=(0, 20))
        
        # Label hiển thị camera với khung
        camera_container = tk.Frame(camera_frame, bg='#34495E')
        camera_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.camera_label = tk.Label(camera_container, bg='#34495E')
        self.camera_label.pack(pady=10)
        
        # Frame cho điều khiển
        control_frame = tk.Frame(camera_frame, bg='#34495E')
        control_frame.pack(pady=10)
        
        # Nút Start/Stop với style mới
        self.is_running = False
        self.start_stop_button = tk.Button(control_frame, 
                                         text="BẮT ĐẦU CAMERA", 
                                         command=self.toggle_camera,
                                         font=("Arial", 12, "bold"),
                                         width=20,
                                         bg='#27AE60',
                                         fg='white',
                                         relief='flat',
                                         cursor='hand2',
                                         activebackground='#219A52',
                                         activeforeground='white')
        self.start_stop_button.pack(pady=10)
        
        # Frame cho kết quả
        result_frame = tk.Frame(content_frame, bg='#34495E', relief='solid', borderwidth=2)
        result_frame.pack(side=tk.RIGHT, fill='both', expand=True)
        
        # Label hiển thị kết quả với style mới
        self.result_title = tk.Label(result_frame, 
                                   text="KẾT QUẢ NHẬN DIỆN", 
                                   font=("Arial", 18, "bold"),
                                   bg='#34495E',
                                   fg='#ECF0F1')
        self.result_title.pack(pady=20)
        
        # Frame cho kết quả chi tiết
        result_detail_frame = tk.Frame(result_frame, bg='#2C3E50', relief='solid', borderwidth=1)
        result_detail_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        self.result_label = tk.Label(result_detail_frame, 
                                   text="Chưa phát hiện biển báo", 
                                   font=("Arial", 14),
                                   wraplength=400,
                                   justify=tk.LEFT,
                                   bg='#2C3E50',
                                   fg='#ECF0F1')
        self.result_label.pack(pady=20)
        
        # Hướng dẫn sử dụng
        instruction_frame = tk.Frame(result_frame, bg='#34495E')
        instruction_frame.pack(fill='x', padx=20, pady=20)
        
        instruction_text = """
Hướng dẫn sử dụng:
1. Nhấn nút "BẮT ĐẦU CAMERA" để khởi động
2. Đặt biển báo trước camera
3. Giữ biển báo ổn định và đủ ánh sáng
4. Kết quả sẽ hiển thị tự động
5. Nhấn "DỪNG CAMERA" để tắt
        """
        
        instruction_label = tk.Label(instruction_frame,
                                   text=instruction_text,
                                   font=("Arial", 10),
                                   bg='#34495E',
                                   fg='#BDC3C7',
                                   justify=tk.LEFT)
        instruction_label.pack(pady=10)
        
        # Khởi tạo camera
        self.cap = None
        self.after_id = None
        
        # Tối ưu hóa camera
        self.frame_buffer = None
        self.processing_frame = False
        self.last_frame_time = 0
        self.frame_interval = 1/30  # 30 FPS
        self.prediction_interval = 1000  # 1 giây giữa các lần dự đoán
        self.last_prediction_time = 0  # Thêm biến này
        
        # Thêm biến cho template matching
        self.templates_dir = "templates"
        self.templates = {}
        self.load_templates()
        
        # Tăng ngưỡng độ tin cậy
        self.confidence_threshold = 0.6  # Tăng từ 0.3 lên 0.6
    
    def load_templates(self):
        """Tải các ảnh mẫu từ thư mục templates"""
        try:
            if not os.path.exists(self.templates_dir):
                os.makedirs(self.templates_dir)
                print("Thư mục templates đã được tạo")
                return
                
            for sign_id, sign_name in classes.items():
                template_path = os.path.join(self.templates_dir, f"{sign_id}.jpg")
                if os.path.exists(template_path):
                    template = cv2.imread(template_path)
                    if template is not None:
                        self.templates[sign_id] = template
                        print(f"Đã tải template cho biển báo {sign_name}")
        except Exception as e:
            print(f"Lỗi khi tải templates: {str(e)}")
    
    def preprocess_image(self, image):
        try:
            # Giữ nguyên ảnh màu
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Lọc nhiễu
            image = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Tăng độ tương phản cho từng kênh màu
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Cân bằng histogram
            image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Lọc nhiễu muối tiêu
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Resize ảnh về kích thước 30x30
            image = cv2.resize(image, (30, 30))
            
            # Chuẩn hóa
            image = image.astype('float32') / 255.0
            
            # Thêm chiều batch
            image = np.expand_dims(image, axis=0)
            
            return image
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh: {str(e)}")
            raise
    
    def template_matching(self, frame, predicted_class):
        """So sánh ảnh với template mẫu"""
        try:
            if predicted_class not in self.templates:
                return True  # Không có template mẫu, chấp nhận kết quả dự đoán
                
            template = self.templates[predicted_class]
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Nếu độ tương đồng cao (>0.8), xác nhận kết quả
            return max_val > 0.8
        except Exception as e:
            print(f"Lỗi khi so sánh template: {str(e)}")
            return True
    
    def predict_sign(self, image):
        try:
            # Tiền xử lý ảnh
            processed_image = self.preprocess_image(image)
            
            # Dự đoán
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Lấy class có xác suất cao nhất
            predicted_class = np.argmax(prediction[0]) + 1  # +1 vì classes bắt đầu từ 1
            confidence = prediction[0][predicted_class - 1]
            
            # Kiểm tra độ tin cậy
            if confidence < self.confidence_threshold:
                return None, 0
            
            # So sánh với template mẫu
            if not self.template_matching(image, predicted_class):
                return None, 0
            
            return predicted_class, confidence
        except Exception as e:
            print(f"Lỗi khi dự đoán: {str(e)}")
            raise
    
    def refresh_camera(self):
        """Thử kết nối lại camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Không thể kết nối camera")
            
            # Đặt độ phân giải và FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Đặt buffer size
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Đọc frame đầu tiên để kiểm tra
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Không thể đọc frame từ camera")
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi làm mới camera: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể kết nối camera: {str(e)}")
            return False
    
    def toggle_camera(self):
        if not self.is_running:
            if self.refresh_camera():
                self.is_running = True
                self.start_stop_button.config(text="DỪNG CAMERA", bg='#f44336')
                self.update_camera()
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
        self.last_prediction_time = 0  # Reset thời gian dự đoán
        self.start_stop_button.config(text="BẮT ĐẦU CAMERA", bg='#4CAF50')
        self.camera_label.config(image='')
        self.result_label.config(text="Chưa phát hiện biển báo")
    
    def update_camera(self):
        if self.is_running and self.cap:
            try:
                # Đọc frame mới
                ret, frame = self.cap.read()
                if ret:
                    # Cập nhật frame buffer
                    self.frame_buffer = frame.copy()
                    
                    # Hiển thị frame ngay lập tức
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_display = cv2.resize(frame_rgb, (640, 480))
                    
                    try:
                        photo = ImageTk.PhotoImage(image=Image.fromarray(frame_display))
                        self.camera_label.config(image=photo)
                        self.camera_label.image = photo
                    except Exception as e:
                        print(f"Lỗi khi hiển thị frame: {str(e)}")
                    
                    # Xử lý dự đoán nếu đã đủ thời gian
                    current_time = time.time() * 1000  # Chuyển sang milliseconds
                    if current_time - self.last_prediction_time >= self.prediction_interval:
                        try:
                            predicted_class, confidence = self.predict_sign(self.frame_buffer)
                            
                            if predicted_class is not None:
                                sign_name = classes[predicted_class]
                                
                                # Vẽ thông tin lên frame
                                cv2.putText(frame_display, 
                                          sign_name, 
                                          (10, 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.8, 
                                          (0, 255, 0), 
                                          2)
                                
                                # Cập nhật kết quả
                                result_text = f"Biển báo: {sign_name}\nĐộ tin cậy: {confidence:.2%}"
                                self.result_label.config(text=result_text, fg='#2ECC71')
                            else:
                                self.result_label.config(text="Chưa phát hiện biển báo", fg='#ECF0F1')
                            
                            self.last_prediction_time = current_time
                        except Exception as e:
                            print(f"Lỗi khi dự đoán: {str(e)}")
                            self.result_label.config(text="Lỗi khi dự đoán", fg='#E74C3C')
                
                # Lập lịch cập nhật tiếp theo
                self.after_id = self.root.after(10, self.update_camera)
                
            except Exception as e:
                print(f"Lỗi khi cập nhật camera: {str(e)}")
                self.refresh_camera()
    
    def __del__(self):
        self.stop_camera()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignGUI(root)
    root.mainloop()