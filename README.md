# Helmet and Safety Vest Detection System

This project automatically detects whether a person is wearing a **helmet** and **safety vest** using a custom trained **YOLO model**. It can be used in construction sites, industries, and factories to ensure proper PPE compliance.

## 📂 Project Structure

```
helmet-and-vest-detection-system/
│
├── api/                           # Backend (FastAPI)
│   ├── app.py                     # API endpoint for helmet/vest detection
│   ├── requirements.txt           # Backend dependencies
│   ├── Dockerfile                 # Optional: For Render/Railway deployment
│   └── weights/
│       └── best.pt                # Trained model weights (tracked via Git LFS)
│
├── web/                           # Frontend (Static Website for Netlify)
│   ├── index.html                 # Frontend UI page
│   ├── app.js                     # Sends image to backend API and receives prediction
│   └── styles.css                 # Styling for the UI
│
├── .gitignore
├── .gitattributes
└── README.md
```

## ⚙️ Setup and Run Locally

### 1) Clone the Repository
```bash
git clone https://github.com/your-username/helmet-and-vest-detection-system.git
cd helmet-and-vest-detection-system
```

### 2) Install Backend Dependencies
```bash
cd api
pip install -r requirements.txt
```

### 3) Run the Backend Server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Backend API runs at:
```
http://localhost:7860/predict-image
```

### 4) Open the Frontend UI
Open this file directly in your browser:
```
web/index.html
```

Upload an image → click **Detect** → Result will appear below.

## 🌍 Deployment Guide

### Deploy Backend (Render / Railway)

1. Create New Web Service
2. Connect this repository
3. Set **Start Command**:
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
4. Deploy and copy the backend URL

### Deploy Frontend 

1. Go to [https://app.netlify.com/start](https://huggingface.co/spaces/chpremkalyan/helmet_vest_detection)
2. Select this repository
3. Set:
   - Build Command: _none_
   - Publish Directory: `web`
4. Deploy
5. Update API URL in `web/app.js`

## 👨‍💻 Author
**Prem Kalyan****Nikita mulakala****Aman****Swathi**
