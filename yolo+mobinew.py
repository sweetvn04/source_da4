from ultralytics import YOLO
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "./Face_Yolo_50ep.pt"
model_mbnet_path = "./mobilenet50ep_ver2.h5"
# Load YOLO model (face detector)
yolo_model = YOLO(model_path)

# Load MobileNetV2 age+gender model
mobilenet_model = load_model(model_mbnet_path, compile=False)

IMG_SIZE = 128

# Bộ nhớ tạm để lưu dự đoán theo ID
last_preds = {}

# Thời gian cập nhật (giây)
update_interval = 1  # Giảm xuống 0.5s để mượt hơn
last_update_time = 0

# ✅ HÀM PREPROCESSING ĐÚNG (QUAN TRỌNG!)
def preprocess_face(face_crop):
    """
    Preprocessing chuẩn MobileNetV2 - PHẢI GIỐNG TRAINING
    """
    # Resize về 128x128
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    
    # Chuyển BGR sang RGB (OpenCV mặc định là BGR)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # ✅ QUAN TRỌNG: Dùng preprocessing của MobileNetV2
    # Hàm này sẽ scale về [-1, 1] thay vì [0, 1]
    face_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(face_rgb)
    
    # Thêm batch dimension
    face_batch = np.expand_dims(face_preprocessed, axis=0)
    
    return face_batch

# ✅ Hàm predict cải tiến
def predict_age_gender(face_crop, model):
    """
    Dự đoán tuổi và giới tính từ khuôn mặt
    """
    try:
        # Preprocessing đúng cách
        face_batch = preprocess_face(face_crop)
        
        # Predict
        pred_age, pred_gender = model.predict(face_batch, verbose=0)
        
        age = int(pred_age[0][0])
        age = max(0, min(120, age))  # Giới hạn trong khoảng hợp lý
        
        # ✅ Threshold tốt hơn cho gender
        gender_prob = pred_gender[0][0]
        
        # Có thể thử điều chỉnh threshold nếu model bị bias
        threshold = 0.5
        gender = "Female" if gender_prob > threshold else "Male"
        confidence = gender_prob if gender_prob > threshold else (1 - gender_prob)
        
        return age, gender, float(confidence)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None, None

# Track với ByteTrack
results = yolo_model.track(
    # source="http://192.168.1.135:4747/video",  # external cam
    source=0,  # webcam
    tracker="bytetrack.yaml",
    stream=True,
    show=False
)

# FPS counter
fps_start_time = time.time()
fps_counter = 0
fps_display = 0

for r in results:
    frame = r.orig_img.copy()
    current_time = time.time()
    
    # Tính FPS
    fps_counter += 1
    if current_time - fps_start_time >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_start_time = current_time
    
    if r.boxes is not None and len(r.boxes) > 0:
        cls = r.boxes.cls.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1]*len(boxes)
        
        # ✅ Chỉ chạy MobileNet khi đủ thời gian
        should_update = (current_time - last_update_time) > update_interval
        
        for box, track_id, c in zip(boxes, ids, cls):
            x1, y1, x2, y2 = box
            
            # ✅ Padding hợp lý hơn (0.2 = 20%)
            pad = 0.2
            w, h = x2 - x1, y2 - y1
            x1_pad = max(0, int(x1 - w * pad / 2))
            y1_pad = max(0, int(y1 - h * pad / 2))
            x2_pad = min(frame.shape[1], int(x2 + w * pad / 2))
            y2_pad = min(frame.shape[0], int(y2 + h * pad / 2))
            
            roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if roi.size == 0:
                continue
            
            # Chỉ predict khi đủ thời gian
            if should_update:
                age, gender, conf = predict_age_gender(roi, mobilenet_model)
                
                if age is not None:
                    last_preds[track_id] = (gender, age, conf)
        
        # Update time sau khi xử lý hết batch
        if should_update:
            last_update_time = current_time
        
        # ✅ Vẽ bbox + label với confidence
        for box, track_id, c in zip(boxes, ids, cls):
            x1, y1, x2, y2 = box
            
            # Label mặc định
            label = f"ID:{track_id}"
            color = (0, 255, 0)  # Xanh lá
            
            if track_id in last_preds:
                gender, age, conf = last_preds[track_id]
                label = f"ID:{track_id} | {gender} {age}y"
                
                # ✅ Màu theo giới tính
                color = (255, 105, 180) if gender == "Female" else (30, 144, 255)  # Hồng / Xanh dương
                
                # Vẽ confidence bar nhỏ
                conf_text = f"{conf:.0%}"
                cv2.putText(frame, conf_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Vẽ bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label với background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Hiển thị FPS
    cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Hiển thị số người đang track
    cv2.putText(frame, f"Tracking: {len(last_preds)} faces", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("YOLO + MobileNetV2 (Age/Gender)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):  # Clear predictions
        last_preds.clear()
        print("Cleared all predictions")

cv2.destroyAllWindows()

# ✅ In thống kê cuối
print("\n=== Statistics ===")
print(f"Total tracked faces: {len(last_preds)}")
if last_preds:
    genders = [pred[0] for pred in last_preds.values()]
    ages = [pred[1] for pred in last_preds.values()]
    print(f"Male: {genders.count('Male')}, Female: {genders.count('Female')}")
    print(f"Average age: {np.mean(ages):.1f} years")