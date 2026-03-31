# ✍️ Handwritten Digit Recognition (TensorFlow)

A full-stack web application that recognizes handwritten digits using a **TensorFlow/Keras neural network trained on the MNIST dataset**.

Draw a digit in your browser, send it to a Flask backend, and get real-time predictions with confidence scores and probability distribution.

---

## 🚀 Features

### 🧠 Deep Learning Model
- Built with **TensorFlow / Keras**
- Trained on the **MNIST dataset**
- Multi-layer neural network with dropout
- ~98% accuracy on test data
- Automatically loads saved model or trains on first run

### 🌐 Interactive Frontend
- Canvas-based drawing interface
- Works with mouse & touch (mobile-friendly)
- Real-time prediction display
- Visual probability bars for all digits (0–9)

### ⚙️ Backend API (Flask)
- RESTful API endpoints
- Image preprocessing pipeline
- Model inference
- SQLite database for storing predictions
- CORS enabled

### 🗄️ Database
- Stores prediction history
- Tracks confidence scores
- Provides aggregated statistics

---

## 🏗️ Project Structure


project/
│
├── backend/
│ └── app.py # Flask API + ML model
│
├── frontend/
│ ├── index.html # UI layout
│ └── app.js # Canvas + API logic
│
├── predictions.db # SQLite database (auto-created)
├── digit_recognition_model.h5 # Saved trained model
└── README.md


---

## 🧰 Tech Stack

**Backend**
- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- SQLite

**Frontend**
- HTML5
- Vanilla JavaScript
- Canvas API

---

## ⚡ Getting Started

### 1️⃣ Install Dependencies

```
pip install flask flask-cors tensorflow numpy pillow
```
2️⃣ Run Backend
python app.py
First Run Behavior:
Database is created (predictions.db)
Model is trained on MNIST (takes ~1–2 minutes)
Model is saved as digit_recognition_model.h5

Server runs at:

http://localhost:5000
3️⃣ Run Frontend

Open the HTML file:

frontend/index.html

Or serve it locally:

npx serve frontend
🔌 API Endpoints
🏠 Home
GET /
🔍 Predict Digit
POST /api/predict

Request:
```
{
  "image": "base64-encoded-image"
}
```
Response:
```
{
  "digit": 5,
  "confidence": 0.987,
  "probabilities": [0.01, 0.00, ..., 0.98],
  "message": "Prediction successful"
}
```
📜 Prediction History
GET /api/history

Returns last 50 predictions.

📊 Statistics
GET /api/stats

Example:
```
{
  "total_predictions": 120,
  "digit_distribution": [
    { "digit": 0, "count": 15, "avg_confidence": 0.96 }
  ]
}
```
🧠 How It Works
1. Drawing Input
User draws a digit on canvas
Converted to base64 PNG
2. Image Preprocessing
Convert to grayscale
Resize to 28×28 pixels
Invert colors (white digit on black background)
Normalize pixel values (0–1)
Flatten into 784-length vector
3. Neural Network Architecture
Input Layer:    784 neurons (28×28)
Hidden Layer 1: 128 neurons (ReLU)
Dropout:        0.2
Hidden Layer 2: 64 neurons (ReLU)
Dropout:        0.2
Output Layer:   10 neurons (Softmax)
4. Prediction
Model outputs probabilities for digits 0–9
Highest probability = predicted digit
🗄️ Database Schema
predictions
Column	Type
id	INTEGER
predicted_digit	INTEGER
confidence	REAL
timestamp	TIMESTAMP
image_data	TEXT
🎯 Usage Tips
Draw digits large and centered
Avoid overlapping strokes
Use simple, clean shapes
Works best with single digits (0–9)
🔐 Notes
Model is trained on MNIST dataset
Works well for standard handwritten digits
Not designed for multi-digit recognition
🚧 Future Improvements
Support multi-digit recognition
Add CNN for better accuracy
Deploy to cloud (Docker + CI/CD)
Save full images instead of partial base64
Improve UI/UX design
Add authentication for API usage
📄 License

MIT License

🙌 Acknowledgements
MNIST dataset (handwritten digits)
TensorFlow & Keras teams
