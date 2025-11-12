🦺 Helmet and Safety Vest Detection System

An AI-powered PPE (Personal Protective Equipment) detection system that automatically detects whether a person is wearing a helmet and safety vest using a custom-trained YOLO model.
This system is designed for construction sites, factories, and industrial areas to help ensure worker safety and compliance.

📘 Overview

The project uses YOLOv8, FastAPI, and a simple static web interface for real-time detection.
Users can upload an image through the frontend, and the backend processes it using the trained YOLO model to identify helmets and safety vests in the image.

🧩 Features

✅ Helmet & Vest Detection – Real-time image inference using a custom-trained YOLO model.
✅ FastAPI Backend – Handles image upload and inference requests.
✅ Frontend Web App – Simple UI for image upload and viewing results.
✅ Lightweight Deployment – Works seamlessly on Render, Railway, or Netlify.
✅ Cross-Platform – Accessible via browser; no heavy setup required.

🧠 Tech Stack
Layer	Technology Used
🖥️ Frontend	HTML, CSS, JavaScript
⚙️ Backend	FastAPI
🤖 AI Model	YOLOv8 (Custom Trained)
💾 Database	None (lightweight setup)
☁️ Deployment	Render / Railway (Backend), Netlify (Frontend)
🐍 Language	Python 3.11
📂 Project Structure
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

⚙️ Setup and Run Locally
🪜 1) Clone the Repository
git clone https://github.com/your-username/helmet-and-vest-detection-system.git
cd helmet-and-vest-detection-system

⚙️ 2) Install Backend Dependencies
cd api
pip install -r requirements.txt

🚀 3) Run the Backend Server
uvicorn app:app --host 0.0.0.0 --port 7860


Backend API runs at:

http://localhost:7860/predict-image

💻 4) Open the Frontend UI

Open this file directly in your browser:

web/index.html


Upload an image → click Detect → view results instantly.

🌍 Deployment Guide
☁️ Deploy Backend (Render / Railway)

Create a New Web Service.

Connect this GitHub repository.

Set the Start Command:

uvicorn app:app --host 0.0.0.0 --port $PORT


Deploy and copy your backend URL.

🖥️ Deploy Frontend (Netlify)

Go to Netlify Deployment
.

Connect your GitHub repository.

Set:

Build Command: none

Publish Directory: web

Deploy the site.

Update your backend API URL in web/app.js.

🧠 How It Works

The user uploads an image through the web interface.

The frontend sends the image to the FastAPI backend.

YOLOv8 processes the image and detects helmets and vests.

The backend sends annotated image results back to the frontend.

The user can view detection results directly in the browser.

🧾 API Endpoint
Method	Endpoint	Description
POST	/predict-image	Upload an image for detection and get YOLO results
👥 Team & Contributions
Member	Role	Contributions
🧠 Nikita Mulakala (Team Member)	Model & Data Engineer	- Collected and annotated dataset for helmet and safety vest detection.
- Handled preprocessing and augmentation for model training.
- Fine-tuned YOLO weights (best.pt) and improved detection accuracy.
- Assisted in evaluation and testing of inference results.
- Documented model performance metrics and dataset details.
⚙️ Prem Kalyan (Team Leader)	AI & Backend Developer	- Trained and optimized the YOLO model for helmet and vest detection.
- Developed the FastAPI backend for image inference and prediction endpoint (/predict-image).
- Integrated model weights and managed API deployment (Render/Railway).
- Structured overall project architecture and managed backend dependencies.
- Prepared README documentation and testing workflow.
🎨 Swathi (Team Member)	Frontend Developer	- Designed and developed the static web UI using HTML, CSS, and JavaScript.
- Implemented image upload and detection display logic in app.js.
- Integrated frontend with FastAPI backend for live prediction results.
- Deployed the frontend using Netlify and ensured responsive design.
- Worked on enhancing the overall user experience (UI/UX).
📝 Aman (Team Member)	Documentation & Deployment Support	- Assisted with backend deployment setup on Render/Railway.
- Managed GitHub repository (version control, .gitignore, and .gitattributes).
- Created setup guides and contributed to the project documentation.
- Verified end-to-end system integration and conducted testing on deployed app.
- Ensured project deliverables were completed and well-presented.
📦 Requirements

Python 3.11+

Node.js (optional, for frontend development)

YOLOv8 Weights (best.pt)

Required Python packages listed in api/requirements.txt

👨‍💻 About the Team

A passionate group of developers working together to build AI-powered safety solutions for real-world applications in industrial and construction environments.
Our focus: Efficiency, Safety, and Real-Time AI Detection.

🪪 License

This project is licensed under the MIT License – free for educational and personal use.

🌟 Support

If you like this project, please give it a ⭐ Star on GitHub — your support motivates us to keep innovating!
