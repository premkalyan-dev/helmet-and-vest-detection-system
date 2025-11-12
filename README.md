# 🦺 Helmet and Safety Vest Detection System

An **AI-powered PPE (Personal Protective Equipment) detection system** that automatically detects whether a person is wearing a **helmet** and **safety vest** using a custom-trained **YOLO model**.  
This system is designed for **construction sites, factories, and industrial areas** to help ensure worker safety and compliance.

---

## 📘 Overview

The project uses **YOLOv8**, **FastAPI**, and a simple **static web interface** for real-time detection.  
Users can upload an image through the frontend, and the backend processes it using the trained YOLO model to identify **helmets** and **safety vests** in the image.

---

## 🧩 Features

✅ **Helmet & Vest Detection** – Real-time image inference using a custom-trained YOLO model.  
✅ **FastAPI Backend** – Handles image upload and inference requests.  
✅ **Frontend Web App** – Simple UI for image upload and viewing results.  
✅ **Lightweight Deployment** – Works seamlessly on Render, Railway, or Netlify.  
✅ **Cross-Platform** – Accessible via browser; no heavy setup required.  

---

## 🧠 Tech Stack

| **Layer** | **Technology Used** |
|------------|----------------------|
| 🖥️ **Frontend** | HTML, CSS, JavaScript |
| ⚙️ **Backend** | FastAPI |
| 🤖 **AI Model** | YOLOv8 (Custom Trained) |
| 💾 **Database** | None (lightweight setup) |
| ☁️ **Deployment** | Render / Railway (Backend), Netlify (Frontend) |
| 🐍 **Language** | Python 3.11 |

---

## 📂 Project Structure

```
helmet-and-vest-detection-system/
│
├── api/                           # Backend (FastAPI)
│   ├── app.py                     # API endpoint for helmet/vest detection
│   ├── requirements.txt           # Backend dependencies
│   ├── Dockerfile                 # Optional: For Render/Railway deployment
│   └── weights/
│       └── best.pt                # Trained YOLO model weights (via Git LFS)
│
├── web/                           # Frontend (Static Website for Netlify)
│   ├── index.html                 # Web UI page
│   ├── app.js                     # Sends image to backend API and receives prediction
│   └── styles.css                 # Styling for the UI
│
├── .gitignore
├── .gitattributes
└── README.md
```

---

## ⚙️ Setup and Run Locally

### 🪜 1) Clone the Repository
```bash
git clone https://github.com/your-username/helmet-and-vest-detection-system.git
cd helmet-and-vest-detection-system
```

### ⚙️ 2) Install Backend Dependencies
```bash
cd api
pip install -r requirements.txt
```

### 🚀 3) Run the Backend Server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Backend API runs at:
```
http://localhost:7860/predict-image
```

### 💻 4) Open the Frontend UI
Open this file directly in your browser:
```
web/index.html
```

Upload an image → click **Detect** → view results instantly.

---

## 🌍 Deployment Guide

### ☁️ Deploy Backend (Render / Railway)
1. Create a **New Web Service**.  
2. Connect this GitHub repository.  
3. Set the **Start Command**:
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
4. Deploy and copy your backend URL.

### 🖥️ Deploy Frontend (Netlify)
1. Go to [Click here for live Project](https://huggingface.co/spaces/chpremkalyan/helmet_vest_detection).  
2. Connect your GitHub repository.  
3. Set:
   - **Build Command:** none  
   - **Publish Directory:** `web`  
4. Deploy the site.  
5. Update your backend API URL in `web/app.js`.

---

## 🧠 How It Works

1. The user uploads an image through the web interface.  
2. The frontend sends the image to the FastAPI backend.  
3. YOLOv8 processes the image and detects helmets and vests.  
4. The backend sends annotated image results back to the frontend.  
5. The user can view detection results directly in the browser.

---

## 🧾 API Endpoint

| **Method** | **Endpoint** | **Description** |
|-------------|--------------|-----------------|
| POST | `/predict-image` | Upload an image for detection and get YOLO results |

---

## 👥 Team & Contributions

| **Member** | **Role** | **Contributions** |
|-------------|-----------|------------------|
| 🧠 **Nikita Mulakala (Team Leader)** | **AI & Backend Developer** | - Collected and annotated dataset for helmet and safety vest detection.<br>- Handled preprocessing and augmentation for model training.<br>- Fine-tuned YOLO weights (`best.pt`) and improved detection accuracy.<br>- Assisted in evaluation and testing of inference results.<br>- Documented model performance metrics and dataset details. |
| ⚙️ **Prem Kalyan (Team Member)** | **Model & Data Engineer** | - Trained and optimized the YOLO model for helmet and vest detection.<br>- Developed the FastAPI backend for image inference and prediction endpoint (`/predict-image`).<br>- Integrated model weights and managed API deployment (Render/Railway).<br>- Structured overall project architecture and managed backend dependencies.<br>- Prepared README documentation and testing workflow. |
| 🎨 **Swathi (Team Member)** | **Frontend Developer** | - Designed and developed the static web UI using HTML, CSS, and JavaScript.<br>- Implemented image upload and detection display logic in `app.js`.<br>- Integrated frontend with FastAPI backend for live prediction results.<br>- Deployed the frontend using Netlify and ensured responsive design.<br>- Worked on enhancing the overall user experience (UI/UX). |
| 📝 **Aman (Team Member)** | **Documentation & Deployment Support** | - Assisted with backend deployment setup on Render/Railway.<br>- Managed GitHub repository (version control, `.gitignore`, and `.gitattributes`).<br>- Created setup guides and contributed to the project documentation.<br>- Verified end-to-end system integration and conducted testing on deployed app.<br>- Ensured project deliverables were completed and well-presented. |

---

## 📦 Requirements

- Python 3.11+  
- Node.js (optional, for frontend development)  
- YOLOv8 Weights (`best.pt`)  
- Required Python packages listed in `api/requirements.txt`

---

## 👨‍💻 About the Team

A passionate group of developers working together to build **AI-powered safety solutions** for real-world applications in industrial and construction environments.  
Our focus: **Efficiency, Safety, and Real-Time AI Detection.**

---

## 🪪 License

This project is licensed under the **MIT License** – free for educational and personal use.

---

## 🌟 Support

If you like this project, please give it a ⭐ **Star** on GitHub — your support motivates us to keep innovating!
