from ultralytics import YOLO
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load YOLO model (face detector)
yolo_model = YOLO("best.pt")

# Load MobileNetV2 age+gender model
mobilenet_model = load_model("best_age_gender_model50ep.h5", compile=False)
IMG_SIZE = 128

# Bộ nhớ tạm để lưu dự đoán theo ID
last_preds = {}

# Thời gian cập nhật (giây)
update_interval = 1.0  
last_update_time = 0  

# Track với ByteTrack
results = yolo_model.track(
    # source="http://192.168.1.5:4747/video",  using external cam
    source=0, # using camera,
    tracker="bytetrack.yaml",
    stream=True,
    show=False
)

for r in results:
    frame = r.orig_img.copy()
    current_time = time.time()

    if r.boxes is not None:
        cls = r.boxes.cls.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        ids   = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1]*len(boxes)

        for box, track_id, c in zip(boxes, ids, cls):
            # YOLO face model: chỉ có class=0 (face) → không cần lọc
            x1, y1, x2, y2 = box

            # --- Thêm padding để crop rộng hơn ---
            pad = 0.9  # 20%
            w, h = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - w * pad / 2))
            y1 = max(0, int(y1 - h * pad / 2))
            x2 = min(frame.shape[1], int(x2 + w * pad / 2))
            y2 = min(frame.shape[0], int(y2 + h * pad / 2))

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Chỉ chạy MobileNet mỗi N giây
            if current_time - last_update_time > update_interval:
                roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                roi_array = img_to_array(roi_resized) / 255.0
                roi_array = np.expand_dims(roi_array, axis=0)

                pred_age, pred_gender = mobilenet_model.predict(roi_array, verbose=0)
                age = int(pred_age[0][0])
                gender = "Male" if pred_gender[0][0] > 0.5 else "Female"

                last_preds[track_id] = (gender, age)

        # update time sau khi batch predict
        if current_time - last_update_time > update_interval:
            last_update_time = current_time

        # Vẽ bbox + label
        for box, track_id, c in zip(boxes, ids, cls):
            x1, y1, x2, y2 = box
            label = f"ID:{track_id}"

            if track_id in last_preds:
                gender, age = last_preds[track_id]
                label = f"ID:{track_id} {gender}, {age}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO + MobileNetV2 (Age/Gender)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
