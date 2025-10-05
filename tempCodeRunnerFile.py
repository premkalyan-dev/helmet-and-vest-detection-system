from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

# ----------------------------
# Model setup
# ----------------------------
model_path = 'yolov8s.pt'  # pretrained YOLOv8 model
model = YOLO(model_path)

# ----------------------------
# File upload setup
# ----------------------------
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No image selected'

    # save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # perform inference
    results = model(filepath)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename)
    results[0].save(filename=output_path)

    return render_template('result.html', 
                           original_image=filepath, 
                           output_image=output_path)

# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    print("Template folder absolute path:", os.path.abspath(app.template_folder))
    app.run(debug=True)
