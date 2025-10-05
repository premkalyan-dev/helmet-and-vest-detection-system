from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import cv2
import sqlite3
import hashlib
import json


app = Flask(__name__, template_folder='templates', static_folder='static')

# ================= File Uploads =================
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ================= Model =================
model_path = r"C:\Users\premk\Downloads\hackton\helmet_vest_model2\weights\best.pt"
model = YOLO(model_path)

# ================= DB Setup =================
def init_db():
    conn = sqlite3.connect("safetysnap.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        detections TEXT,
        detections_hash TEXT,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        labels TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()  # run once at startup


# ================= Routes =================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in the request'

    file = request.files['file']
    if file.filename == '':
        return 'No image selected'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run YOLO
    results_list = model.predict(filepath, imgsz=320, conf=0.2)
    boxes = results_list[0].boxes
    class_names = model.names
    img = cv2.imread(filepath)

    counts = {}
    detections_data = []

    if boxes is not None:
        h, w, _ = img.shape
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            counts[cls_name] = counts.get(cls_name, 0) + 1

            # Bounding box values
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Normalize (0–1)
            x1n, y1n, x2n, y2n = x1 / w, y1 / h, x2 / w, y2 / h
            detections_data.append({
                "class": cls_name,
                "bbox": [x1n, y1n, x2n, y2n]
            })

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            label = f"{cls_name}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    output_filename = 'output_' + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, img)

    # ================= Save to DB =================
    detections_json = json.dumps(detections_data, sort_keys=True)
    detections_hash = hashlib.sha256(detections_json.encode()).hexdigest()
    labels = ",".join(counts.keys()) if counts else ""

    conn = sqlite3.connect("safetysnap.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO images (filename, detections, detections_hash, labels)
        VALUES (?, ?, ?, ?)
    """, (output_filename, detections_json, detections_hash, labels))
    conn.commit()
    conn.close()

    # ================= Message =================
    if counts:
        detection_message = ", ".join([f"{v} {k}" for k, v in counts.items()]) + " detected."
    else:
        detection_message = "No helmets or vests detected."

    original_image_rel = f"uploads/{filename}"
    output_image_rel = f"uploads/{output_filename}"

    return render_template('result.html',
                           original_image=original_image_rel,
                           output_image=output_image_rel,
                           detection_message=detection_message)


if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    app.run(debug=True)
