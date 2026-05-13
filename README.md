# AI Health Coach 🩺
> **A Mini Doctor in Your Pocket**

AI Health Coach is an intelligent, integrated health monitoring system that combines real-time wearable data, machine learning disease prediction, and AI-driven personalized advice to help users manage their well-being proactively.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🌟 Features

### 1. **Live Health Dashboard** 📊
- Connects to (simulated) wearable devices.
- Visualizes real-time metrics: Heart Rate, HRV, Steps, Sleep Quality, SpO2, and Stress Levels.
- Bento-grid style layout for intuitive data consumption.

### 2. **Weather Impact Analysis** 🌦️
- Real-time weather tracking (Temperature, Humidity, AQI, UV Index).
- Correlates environmental factors with health risks.
- Provides dynamic workout and lifestyle recommendations based on current weather.

### 3. **AI Doctor & RAG Chatbot** 💬
- **Conversational Interface:** Chat with an AI assistant about your health concerns.
- **RAG (Retrieval-Augmented Generation):** Uses a local medical knowledge base to ground answers in verified data.
- **Tools:** The AI has access to your user profile and live health stats to give context-aware advice.

### 4. **Disease Prediction (ML Models)** 🏥
- **Diabetes Prediction:** Uses health metrics to estimate diabetes risk.
- **Heart Disease Risk:** Analyzes cardiovascular factors.
- **Stroke Risk:** Assessment based on lifestyle and medical history.
- **Burnout Score:** Calculates hidden burnout risk from sleep and stress patterns.
- **Sleep Troubleshooter:** Identifies root causes of poor sleep quality.

### 5. **Dynamic Workout Planner** 🏋️
- Generates personalized workout plans.
- Adapts to your energy levels (step count), profile age, and current weather conditions.

---

## 🤖 AI Agent Architecture

This project implements a custom **Agentic Architecture** where the AI actively perceives, reasons, and responds.



### 1. **"Workout Planner" (Task-Specific Agent)**
- **Type:** Planning Agent
- **Role:** Synthesizes data from Weather API, Wearable Sensors, and RAG Knowledge Base to generate structured, environment-aware workout routines.

### 2. **"Silent Observer" (Event-Driven Agent)**
- **Type:** Monitoring Agent
- **Role:** Runs in the background of every health check. Autonomously detects critical risks (e.g., High Burnout, High Disease Risk) and triggers system-wide alerts without user intervention.

---

## 🛠️ Technology Stack

- **Frontend:** [Streamlit](https://streamlit.io/) (for interactive web UI).
- **Backend Logic:** Python (Modular service architecture).
- **Database:** SQLite (for user profiles and auth) + FAISS (Vector DB for RAG).
- **Machine Learning:** Scikit-learn, LightGBM, Mistral AI (LLM).
- **Data Visualization:** Plotly Interactive Charts.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8 or higher
- An API Key for [Mistral AI](https://console.mistral.ai/)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd AI_Health_Coach
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory and add your keys:
   ```env
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

```
AI_Health_Coach/
├── app.py                   # Main Streamlit Application Entry Point (Frontend & Logic)
├── requirements.txt         # Global Project Dependencies
├── users.db                 # SQLite Database (Stores User Credentials & Profiles)
├── bg_img.png               # Asset: Background Image for Login Screen
├── build_rag.py             # Script to build/update the RAG Vector Database
├── .env                     # Environment Config (API Keys - Excluded from Git)
│
├── backend/                 # Core Application Logic
│   ├── database.py          # Database Connection & User Management Functions
│   ├── main.py              # Alternative Backend Entry Point (FastAPI/Testing)
│   ├── requirements.txt     # Backend-specific Dependencies
│   │
│   ├── config/              # Configuration Files
│   │   └── defaults.py      # Default values for Health Metrics (e.g., avg glucose)
│   │
│   ├── services/            # Business Logic & External Services
│   │   ├── wearable_service.py     # Simulates Real-time Galaxy Watch Data
│   │   ├── weather_service.py      # Fetches Live Weather from OpenMeteo API
│   │   ├── weather_health_rules.py # Logic mapping Weather conditions to Health Tips
│   │   ├── lifestyle_rules.py      # Logic for Lifestyle-based scoring
│   │   ├── ml_diabetes.py          # Wrapper for Diabetes Prediction Model
│   │   ├── ml_heart.py             # Wrapper for Heart Disease Prediction Model
│   │   ├── ml_stroke.py            # Wrapper for Stroke Prediction Model
│   │   ├── ml_burnout.py           # Logic for Invisible Burnout Score
│   │   └── ml_sleep.py             # Logic for Sleep Quality Root Cause Analysis
│   │
│   ├── models/              # Pre-trained Machine Learning Models (.pkl)
│   │   ├── diabetes_model.pkl      # Trained Diabetes Model
│   │   ├── diabetes_scaler.pkl     # Scaler for Diabetes Inputs
│   │   ├── heart_model.pkl         # Trained Heart Disease Model
│   │   ├── heart_scaler.pkl        # Scaler for Heart Inputs
│   │   ├── stroke_model.pkl        # Trained Stroke Model
│   │   ├── stroke_scaler.pkl       # Scaler for Stroke Inputs
│   │   ├── invisible_burnout_random_forest.pkl # Random Forest Model for Burnout
│   │   └── sleep_root_cause_lightgbm.pkl       # LightGBM Model for Sleep Analysis
│   │
│   ├── rag/                 # RAG (Retrieval Augmented Generation) System
│   │   ├── rag_service.py          # Logic for Vector Search & Context Retrieval
│   │   ├── health_guidlines.pdf    # Source Knowledge Base Document
│   │   └── Dynamic_Workout_Planner.pdf # Source Workout Document
│   │
│   └── tools/               # Agent Helper Tools
│       └── agent_tools.py          # Functions accessible by the AI Agent
│
└── ML_models/               # Jupyter Notebooks for Model Training
    ├── diabetes.ipynb              # Training script for Diabetes Model
    ├── heart.ipynb                 # Training script for Heart Model
    ├── Stroke.ipynb                # Training script for Stroke Model
    ├── Invisible Burnout Score.ipynb   # Research & Training for Burnout Score
    ├── Sleep Root-Cause Explainer.ipynb # Research & Training for Sleep Analysis
    └── Feature_Engineering_ML.ipynb    # Data Preprocessing Experiments
```

## 🛡️ Usage Notes

- **Login:** Create an account on the "Sign Up" tab first.
- **Wearable Sim:** Use the "Connect Device" button in the sidebar to start the data stream.
- **ML Models:** Navigate to the "Health Check" tab to run specific disease risk assessments.
- **Chat:** Use the "AI Doctor" tab for general health queries.

---

<p align="center">Made with ❤️ for Health & AI</p>
