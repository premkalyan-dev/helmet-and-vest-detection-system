

# 🪖 Helmet & Vest Detection System 👷‍♂️🦺

Detect helmets and safety vests effortlessly using a custom-trained YOLO CNN model served with a Flask backend — ensuring safer workplaces through smart AI! 🚧🤖

---

## ✨ Features

- 🎯 **YOLO Object Detection:** Real-time helmet & vest detection with a custom YOLO CNN model  
- 🧠 **Deep Learning:** CNN-based training for precise safety gear recognition  
- 🔥 **Flask Backend:** Handles image uploads, runs inference, and returns results  
- 🔐 **User Authentication:** Firebase Email & Password login system for secure access  
- 🖼️ **Image Upload:** Drag & drop or click to select images for detection  
- 🎨 **Modern Responsive UI:** Attractive, user-friendly interface with split login/welcome design  

---
/app
├── model/ # YOLO model weights and config files
├── static/ # Frontend CSS, JS, and images
├── templates/ # HTML templates (e.g., index.html)
├── app.py # Flask backend server script
├── requirements.txt # Python dependencies
README.md # This README file


---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+  
- Flask  
- OpenCV  
- PyTorch / TensorFlow (depending on YOLO implementation)  
- Firebase project for authentication

## 📁 Project Structure

helmet-and-vest-detection-system/
├── api/ # Backend (FastAPI)
│ ├── app.py # API endpoint for helmet/vest detection
│ ├── requirements.txt
│ ├── Dockerfile # (Optional) for cloud deployment
│ └── weights/
│ └── best.pt # Model weights (Git LFS)
│
└── web/ # Frontend (Static Website)
├── index.html
├── app.js
└── styles.css

Install Backend Dependencies

cd api
pip install -r requirements.txt






