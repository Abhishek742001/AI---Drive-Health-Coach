# 🩺 AI Health Coach — Complete Project Documentation

> **"A Mini Doctor in Your Pocket"**
> Python · Streamlit · Mistral AI · FAISS · Scikit-learn · LightGBM · Random Forest

---

## 1. 📌 Problem Statement

Healthcare today is largely **reactive rather than proactive** — individuals seek medical attention only after symptoms become critical, often missing early warning signs of dangerous conditions like diabetes, heart disease, stroke, or burnout. Access to personalized, continuous health monitoring is typically limited to those who can afford regular doctor visits or expensive wearable ecosystems.

Existing health applications offer **generic, one-size-fits-all advice** that fails to account for an individual's real-time physiological data, lifestyle patterns, or environmental factors like temperature, AQI, and UV index. Silent conditions such as burnout, pre-diabetes, and sleep disorders go undetected for years simply because there is no accessible, intelligent system that can identify them from everyday behavioral signals.

There is also a critical gap in **AI-driven conversational health support** — most people cannot get immediate, context-aware, and personalized answers to their health concerns without booking a doctor's appointment.

**AI Health Coach** solves this by providing a unified intelligent platform that continuously monitors health, predicts disease risks using machine learning, delivers personalized AI-powered guidance grounded in medical knowledge, and generates environment-aware fitness plans — making proactive healthcare accessible to everyone.

---

## 2. 🎯 Project Objectives

| # | Objective | How Achieved |
|---|---|---|
| 1 | Continuous Health Monitoring | Simulated Galaxy Watch data (sine-wave HR, HRV, SpO2, Steps) |
| 2 | Early Disease Risk Detection | 5 trained ML models running on user inputs |
| 3 | Personalized AI Medical Guidance | RAG + Mistral LLM with real-time context injection |
| 4 | Environment-Aware Advice | OpenMeteo API + rules engine |
| 5 | Automated Risk Alerting | Silent Observer Agent — no user trigger needed |
| 6 | Smart Workout Planning | Planning Agent with multi-source reasoning |

---

## 3. 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           USER (Browser)                             │
│                     Streamlit Web App (app.py)                       │
│                                                                      │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌────────┐  │
│  │Dashboard │ │ Weather  │ │Health Check│ │AI Doctor │ │Workout │  │
│  └────┬─────┘ └────┬─────┘ └─────┬──────┘ └────┬─────┘ └───┬────┘  │
└───────┼────────────┼─────────────┼──────────────┼───────────┼───────┘
        │            │             │              │           │
   Wearable    OpenMeteo      5 ML Models     RAG + LLM   Planning
   Service        API           (.pkl)        (FAISS +    Agent
   (Simulated)  (Free API)                   Mistral AI)
        │            │             │              │           │
      Charts     Weather       Risk Scores    AI Doctor   Workout
    (Plotly)     Health         + Alerts       Chatbot      Plan
                  Rules       Silent Obs.       (JSON)     (JSON)
                             Agent (Auto)
                                  │
                               SQLite DB
                             (User Profiles)
```

---

## 4. 📂 Complete Module Breakdown

### 4.1 `app.py` — Main Application (1426 lines)

The **orchestration layer** of the entire project. Responsibilities:

- **Page Config**: Wide layout, dark theme, custom CSS (glassmorphism, bento-grid, Google Fonts: *Plus Jakarta Sans*)
- **State Management**: `st.session_state` tracks login, device connection, chat history, active alerts
- **5 Feature Tabs**: Dashboard, Weather Impact, Health Check, AI Doctor, Workout Plan
- **Smart Data Fusion**: `get_unified_wellness_inputs()` — merges wearable data with form inputs, wearable data takes priority
- **Agent Logic**: Contains both the Workout Planner Agent and the Silent Observer Agent
- **Helper Functions**: `create_donut_chart()`, `render_bento_metric()`, `generate_workout_plan()`, `_get_fallback_plan()`

---

### 4.2 `backend/database.py` — Authentication & User Storage

**Technology: SQLite + bcrypt**

```python
# Table Schema
CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL,   # bcrypt hashed
    profile  TEXT             # JSON blob: age, weight, conditions
)

# Key Functions
init_db()            → Creates table on first run
register_user()      → Hashes password with bcrypt.hashpw() + gensalt()
login_user()         → Verifies with bcrypt.checkpw()
update_profile()     → Stores profile as JSON string
```

**Why bcrypt?** It is a slow, salted hashing algorithm specifically designed for password storage. Unlike MD5 or SHA, bcrypt is inherently resistant to brute-force and rainbow table attacks.

---

### 4.3 `backend/services/wearable_service.py` — Simulated Wearable Device

Simulates a **Samsung Galaxy Watch 5** with scientifically modeled data:

```python
# Heart Rate — Circadian sine wave + noise
base_hr = 75 + 10 * math.sin((hour - 10) * math.pi / 12)
# Lowest at 4am, highest at 4pm — matches human physiology

# Steps — probabilistic accumulation
if random.random() > 0.7:  # 30% chance of walking per second
    step_count += elapsed * 1.5

# HRV (RMSSD) — Normal range 25–85 ms
hrv_rmssd = round(random.uniform(25, 85), 1)

# Stress Score — inverse to HRV
stress_score = 100 - hrv_rmssd + noise

# Derived Metrics
calories = steps * 0.04      # ~0.04 kcal per step
distance = steps * 0.0008    # ~0.8m per step = 0.0008 km
sleep_score = (duration/8)*50 + (efficiency/100)*50
```

Also generates **24-hour trend history** arrays for Plotly charts (Heart Rate + Hourly Steps).

---

### 4.4 `backend/services/weather_service.py` — Live Weather

- **API**: OpenMeteo (completely free, no API key)
- **Geocoding**: Converts city name → lat/lon coordinates
- **Data Fetched**: Temperature, Feels-like, Humidity, AQI, UV Index, Wind Speed, Sunrise, Sunset
- **Forecast**: 5-day forecast for planning purposes

---

### 4.5 `backend/services/weather_health_rules.py` — Rules Engine

Maps environmental conditions → health warnings:

| Condition | Rule | Alert |
|---|---|---|
| AQI > threshold | Air quality dangerous | Avoid outdoor exercise |
| UV Index high | Skin burn risk | Apply SPF, wear hat |
| Temp > 35°C + Humidity > 80% | Heat stress | Hydrate, rest indoors |
| Temperature very low | Cardiovascular strain | Warm up properly |

---

## 5. 🤖 ML Models — Deep Dive

### Overview of All Models

| # | Disease | Algorithm | Type | Output |
|---|---|---|---|---|
| 1 | Diabetes | Scikit-learn Classifier | Binary Classification | High / Low Risk |
| 2 | Heart Disease | Scikit-learn Classifier | Binary Classification | High / Low Risk |
| 3 | Stroke | Scikit-learn Classifier | Binary Classification | High / Low Risk |
| 4 | Burnout | Random Forest Regressor | Regression (Score 0–100) | Score + Level |
| 5 | Sleep Quality | LightGBM Regressor | Regression (Efficiency %) | Score + Root Causes |

All models use:
- **`joblib`** for loading pre-trained `.pkl` files
- **`StandardScaler`** for input normalization (scaler fitted during training, reused at inference)
- **Strict feature ordering** to prevent input misalignment

---

### 5.1 🩸 Diabetes Prediction Model

**File**: `ml_diabetes.py` + `diabetes_model.pkl` + `diabetes_scaler.pkl`

**Dataset**: Pima Indians Diabetes Dataset (UCI Repository)
- 768 samples, binary labels (0 = No Diabetes, 1 = Diabetes)

**Input Features:**

| Feature | Description | Unit |
|---|---|---|
| `Pregnancies` | Number of pregnancies | Count |
| `Glucose` | Plasma glucose concentration (2-hr oral glucose test) | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Tricep skinfold thickness | mm |
| `Insulin` | 2-hour serum insulin | μU/mL |
| `BMI` | Body Mass Index | kg/m² |
| `DiabetesPedigreeFunction` | Genetic diabetes likelihood score | Float |
| `Age` | Patient age | Years |

**Default Values**: Glucose=120, BMI=25, Age=40, BP=80

**Algorithm**: Scikit-learn Classifier (Logistic Regression / SVM)

**Why this algorithm?**
- The Pima dataset is small (768 rows) — simple linear models generalize better than complex ones
- Logistic Regression is interpretable and performs well with tabular medical data
- Feature scaling (StandardScaler) ensures all features contribute equally despite different units

**Prediction Logic:**
```python
X = scaler.transform([[Pregnancies, Glucose, BP, SkinThickness,
                        Insulin, BMI, DiabetesPedigreeFunction, Age]])
return "High Risk" if model.predict(X)[0] == 1 else "Low Risk"
```

---

### 5.2 ❤️ Heart Disease Prediction Model

**File**: `ml_heart.py` + `heart_model.pkl` + `heart_scaler.pkl`

**Dataset**: Cleveland Heart Disease Dataset (UCI ML Repository)
- 303 samples, binary labels (0 = No Disease, 1 = Disease Present)
- Gold standard dataset used in cardiovascular ML research since the 1980s

**Input Features (14 Clinically Validated):**

| Feature | Description | Values |
|---|---|---|
| `age` | Patient age | Years |
| `sex` | Biological sex | 0=Female, 1=Male |
| `dataset` | Data source origin | Encoded int |
| `cp` | Chest pain type | 0–3 (0=Typical angina, 3=Asymptomatic) |
| `trestbps` | Resting blood pressure | mm Hg |
| `chol` | Serum cholesterol | mg/dL |
| `fbs` | Fasting blood sugar > 120 mg/dL | 0=No, 1=Yes |
| `restecg` | Resting ECG results | 0–2 |
| `thalch` | Maximum heart rate achieved | bpm |
| `exang` | Exercise-induced angina | 0=No, 1=Yes |
| `oldpeak` | ST depression induced by exercise | mm |
| `slope` | Slope of peak exercise ST segment | 0–2 |
| `ca` | Number of major vessels colored by fluoroscopy | 0–3 |
| `thal` | Thalassemia type | 0–3 |

**Default Values**: Age=45, Cholesterol=200, BP=120, MaxHR=150

**Algorithm**: Scikit-learn Classifier (Random Forest / SVM)

**Why this algorithm?**
- Requires handling **both categorical and continuous** features — Random Forest handles mixed types natively
- 14 features with complex non-linear interactions (e.g., ST depression + chest pain type = highly predictive together)
- Random Forest provides feature importance for medical interpretability

---

### 5.3 🧠 Stroke Risk Prediction Model

**File**: `ml_stroke.py` + `stroke_model.pkl` + `stroke_scaler.pkl`

**Dataset**: Stroke Prediction Dataset (Kaggle)
- ~5,110 samples, severe class imbalance (only ~5% positive cases)
- Required SMOTE or class-weight balancing during training

**Input Features (10):**

| Feature | Description | Values |
|---|---|---|
| `gender` | Patient gender | 0=Female, 1=Male |
| `age` | Patient age | Years |
| `hypertension` | Has hypertension | 0=No, 1=Yes |
| `heart_disease` | Has heart disease | 0=No, 1=Yes |
| `ever_married` | Marital status | 0=No, 1=Yes |
| `work_type` | Type of occupation | Encoded int (0–4) |
| `Residence_type` | Urban or Rural | 0=Rural, 1=Urban |
| `avg_glucose_level` | Average blood glucose | mg/dL |
| `bmi` | Body Mass Index | kg/m² |
| `smoking_status` | Smoking history | Encoded int |

**Default Values**: Age=45, Glucose=120, BMI=25

**Algorithm**: Gradient Boosting Classifier

**Why this algorithm?**
- Dataset is highly **imbalanced** (~95% negative). Gradient Boosting with `class_weight='balanced'` or SMOTE handles rare positive (stroke) cases far better than logistic regression
- Iteratively learns from misclassified examples — critical when stroke cases are rare but life-threatening to miss

---

### 5.4 🔥 Invisible Burnout Score Model

**File**: `ml_burnout.py` + `invisible_burnout_random_forest.pkl`

**Dataset**: Custom-engineered dataset based on physiological and behavioral signals
- Burnout is a **latent variable** — not directly measurable, estimated from proxy signals

**Input Features (7 Wearable-Derived):**

| Feature | Description | Source |
|---|---|---|
| `hrv_7d_avg` | 7-day average HRV (RMSSD) | Wearable |
| `sleep_7d_avg` | 7-day average sleep duration | Wearable |
| `sleep_pressure` | Accumulated sleep debt metric | Derived |
| `stress_score` | Daily stress score (0–100) | Wearable |
| `activity_load` | Physical activity level (0–100) | Wearable |
| `baseline_hrv` | Personal HRV baseline | Wearable |
| `hrv_deviation` | Current HRV vs baseline deviation | Derived |

**Output:**

| Score Range | Level | Action |
|---|---|---|
| 0–30 | 🟢 Low | Good — maintain current habits |
| 30–70 | 🟡 Medium | Warning — prioritize recovery |
| 70–100 | 🔴 High | Critical — rest + consult professional |

**Algorithm: Random Forest Regressor**

**Why Random Forest?**
1. **Feature Importance** — Ranks which features (HRV, sleep, stress) contribute most to burnout score
2. **Robustness to Noise** — Wearable data is noisy; averaging across many trees handles outliers well
3. **No Scaling Required** — Features have very different scales (HRV in ms vs stress 0–100); Random Forest is scale-invariant
4. **Continuous Score Output** — Produces a 0–100 burnout score, not just binary High/Low, making it more clinically useful
5. **No Overfitting** — Bagging (bootstrap aggregation) across many trees prevents memorizing training data

**Fallback**: If model fails, returns a simulated score to prevent app crashes during demos.

---

### 5.5 💤 Sleep Root Cause Analysis Model

**File**: `ml_sleep.py` + `sleep_root_cause_lightgbm.pkl`

**Dataset**: Custom physiological dataset combining wearable metrics with sleep study data

**Input Features (18 — most comprehensive model):**

| Feature | Description |
|---|---|
| `avg_hr_day_bpm` | Average daytime heart rate |
| `resting_hr_bpm` | Resting heart rate |
| `hrv_rmssd_ms` | Heart Rate Variability (RMSSD) |
| `stress_score` | Daily stress score (0–100) |
| `spo2_avg_pct` | Average blood oxygen % |
| `sleep_duration_hours` | Total sleep time |
| `sleep_architecture_score` | Deep/REM sleep quality score |
| `activity_load` | Physical activity level |
| `hr_strain` | Cardiovascular training load |
| `sleep_pressure` | Accumulated sleep debt |
| `baseline_hrv` | Personal HRV baseline |
| `baseline_rhr` | Personal resting HR baseline |
| `hrv_deviation` | HRV deviation from personal baseline |
| `rhr_deviation` | Resting HR deviation from baseline |
| `hrv_7d_avg` | 7-day HRV rolling average |
| `sleep_7d_avg` | 7-day sleep duration rolling average |
| `day_of_week` | Day of the week (0=Mon, 6=Sun) |
| `is_weekend` | Weekend flag (0/1) |

**Output:** Sleep efficiency score (0–100%) + Root cause explanations

**Explainability Layer (Rule-Based on top of ML):**
```python
if stress_score > 65:
    → "High daily stress may be delaying sleep onset"
if hrv_rmssd_ms < 30 or hrv_deviation < -10:
    → "Low HRV: Body needs more recovery time"
if sleep_duration_hours < 6:
    → "Short sleep duration is the primary issue"
if activity_load < 30:
    → "Low physical activity reduces sleep drive"
```

**Algorithm: LightGBM (Light Gradient Boosting Machine)**

**Why LightGBM?**
1. **Speed** — Trains up to 20x faster than XGBoost using histogram-based leaf-wise splitting; critical for 18 input features
2. **Memory Efficiency** — Uses GOSS (Gradient-based One-Side Sampling), retaining only high-gradient data points
3. **Tabular Data Champion** — Consistently outperforms other algorithms on structured health data benchmarks
4. **Multi-way Feature Interactions** — Sleep quality depends on complex combinations (high stress + low HRV + low activity = very poor sleep); LightGBM captures these via leaf-wise growth
5. **No Feature Scaling Needed** — Tree-based models are scale-invariant
6. **SHAP Compatibility** — Built-in feature importance + SHAP values to explain predictions

**Comparison Table — Why NOT other algorithms?**

| Algorithm | Why Not Chosen |
|---|---|
| Logistic Regression | Only binary output; sleep needs continuous efficiency score |
| Decision Tree | Prone to overfitting with 18 features |
| Neural Network | Overkill for 18 features; interpretability lost |
| XGBoost | Slower, more memory-intensive — LightGBM is strictly superior variant |
| **LightGBM ✅** | Fast, accurate, handles interactions, continuous output, interpretable |

---

## 6. 🤖 AI Agent Architecture

### Agent 1: Workout Planner (Planning Agent)

```
TRIGGER: User opens Workout Plan tab

PERCEPTION:
    ├── Weather API → Current conditions + AQI
    ├── Wearable Service → Steps, HRV, Stress, Heart Rate
    ├── User Profile → Age, Weight, Goals
    └── ML Results → Diabetes/Heart/Stroke/Burnout/Sleep risks

REASONING (RAG):
    └── FAISS Search: "workout for [weather], [steps] steps, [age] years old"
        → Returns top-3 relevant workout guideline chunks

PLANNING (LLM — Mistral AI):
    └── Rich prompt with all context → JSON response:
        {workout_type, duration, intensity, exercises[], safety_notes, reason}

ADAPTATION RULES:
    - Burnout > 70     → Yoga/Stretching instead of HIIT
    - Heart Risk HIGH  → Low intensity only
    - Low HRV < 30ms   → Recovery workout
    - Bad AQI          → Indoor alternatives
    - High Stress      → Stress-relief exercises included

FALLBACK: If Mistral API fails → _get_fallback_plan() heuristics
```

### Agent 2: Silent Observer (Event-Driven Agent)

```
TRIGGER: Every Health Check run (automatic, no user action needed)

MONITORING — After all 5 ML models run:
    ├── Burnout score > 70?       → Alert: "High Burnout Risk Detected"
    ├── Diabetes = "High Risk"?   → Alert: "High Diabetes Risk"
    ├── Heart = "High Risk"?      → Alert: "Cardiovascular Risk Alert"
    └── Stroke = "High Risk"?     → Alert: "Stroke Risk Detected"

ACTION:
    → Alerts appended to st.session_state.active_alerts[]
    → Persist in SIDEBAR across ALL tabs
    → Until user manually clears them

WHY "SILENT"?
    → No user trigger required
    → Always watching in the background
    → Automatic escalation without human intervention
```

---

## 7. 💬 RAG System — End-to-End

### Phase 1: Indexing (One-time via `build_rag.py`)

```
Medical PDFs (health_guidelines.pdf + Dynamic_Workout_Planner.pdf)
        ↓ pdfplumber
Extract text page by page
        ↓ Chunk into 500-char segments with source tag
"[Source: health_guidelines.pdf | Page 3] ...text..."
        ↓ SentenceTransformer (all-MiniLM-L6-v2)
Convert each chunk → 384-dimensional embedding vector
        ↓ FAISS IndexFlatL2
Build vector index (exact L2 nearest neighbor search)
        ↓ Save
data/health_rag_db/index.faiss + index.pkl
```

### Phase 2: Retrieval (At query time via `rag_service.py`)

```
User Query → embed → FAISS search (top_k=3)
        ↓
3 most semantically similar chunks retrieved
        ↓
Injected into Mistral AI prompt with user profile + live vitals
        ↓
Grounded, personalized medical response
```

**Why FAISS?**
- Free and runs **100% locally** — no paid vector DB
- **Millisecond** search over thousands of chunks
- `IndexFlatL2` = exact search, no approximation errors
- `@st.cache_resource` = loaded once, not on every request

---

## 8. 🛠️ Technology Stack

| Layer | Technology | Reason Chosen |
|---|---|---|
| **Frontend** | Streamlit | Python-native UI, no JS needed |
| **Styling** | Custom CSS | Glassmorphism, bento-grid, dark theme |
| **Charts** | Plotly | Interactive, animated, responsive |
| **Backend** | Python 3.8+ | Universal ML/AI ecosystem |
| **Auth DB** | SQLite + bcrypt | Lightweight, serverless, secure |
| **Vector DB** | FAISS | Fast, local, free similarity search |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Lightweight, accurate, local |
| **PDF Parser** | pdfplumber | Reliable medical PDF extraction |
| **LLM** | Mistral AI (mistral-tiny) | Cost-effective, supports JSON mode |
| **ML** | Scikit-learn | Industry standard, simple API |
| **Gradient Boost** | LightGBM | Fastest, most memory-efficient GBM |
| **Ensemble** | Random Forest | Robust, no scaling, feature importance |
| **Weather** | OpenMeteo API | Free, real-time, no API key needed |
| **Env Config** | python-dotenv | Secure API key management |

---

## 9. 🔄 Complete Data Flow

```
① App starts → init_db() creates SQLite users.db

② User logs in → bcrypt verify → profile JSON loaded
   → st.session_state populated

③ Device Connected → wearable_service.get_wearable_data()
   → HR, HRV, SpO2, Steps, Sleep, Stress, 24hr History
   → Dashboard: Bento cards + Plotly charts rendered

④ Weather Tab → OpenMeteo API → forecast data
   → weather_health_rules → health tips generated

⑤ Health Check → Smart Data Fusion (wearable > form inputs)
   → 5 ML models run:
      predict_diabetes()      → High / Low Risk
      predict_heart()         → High / Low Risk
      predict_stroke()        → High / Low Risk
      predict_burnout()       → Score 0–100 + Level
      predict_sleep_quality() → Efficiency % + Root Causes
   → Silent Observer Agent checks all → appends alerts to sidebar

⑥ AI Doctor Chat
   → Query embedded → FAISS top-3 chunks retrieved
   → Profile + Vitals + RAG context → Mistral prompt
   → Response rendered in chat UI

⑦ Workout Plan
   → Weather + Wearable + Profile + ML risks loaded
   → RAG search for guidelines
   → Planning Agent → Mistral AI → JSON plan
   → Workout card rendered (or fallback if API fails)
```

---

## 10. 🚀 Setup & Run Guide

### Step 1 — Clone & Navigate
```bash
git clone <repo-url>
cd AI_Health_Coach
```

### Step 2 — Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Create `.env` File
```env
MISTRAL_API_KEY=your_key_from_console.mistral.ai
```

### Step 5 — Build RAG Vector Database (ONE-TIME)
```bash
python build_rag.py
# ✅ RAG Index Rebuilt Successfully!
```

### Step 6 — Run the App
```bash
streamlit run app.py
# Opens at: http://localhost:8501
```

### Step 7 — Use the App
1. **Sign Up** first on the Sign Up tab
2. Click **"🔗 Connect Device"** in sidebar to start wearable simulation
3. Go to **Health Check** → run disease predictions
4. Go to **AI Doctor** → ask health questions
5. Go to **Workout Plan** → get personalized fitness plan

---

## 11. 📁 Project Folder Structure

```
AI_Health_Coach/
├── app.py                        ← Main entry point (Frontend + Agents)
├── build_rag.py                  ← One-time RAG index builder
├── requirements.txt              ← Python dependencies
├── users.db                      ← SQLite DB (auto-created)
├── bg_img.png                    ← Login background image
├── .env                          ← API keys (not in git)
├── PROJECT_DOCUMENTATION.md      ← This file ✅
│
├── backend/
│   ├── database.py               ← Auth: SQLite + bcrypt
│   ├── main.py                   ← FastAPI entry (testing)
│   │
│   ├── config/
│   │   └── defaults.py           ← Default values for ML forms
│   │
│   ├── services/
│   │   ├── wearable_service.py   ← Galaxy Watch simulator
│   │   ├── weather_service.py    ← OpenMeteo API
│   │   ├── weather_health_rules.py ← Weather → health tips
│   │   ├── ml_diabetes.py        ← Diabetes model inference
│   │   ├── ml_heart.py           ← Heart disease inference
│   │   ├── ml_stroke.py          ← Stroke risk inference
│   │   ├── ml_burnout.py         ← Burnout score (Random Forest)
│   │   └── ml_sleep.py           ← Sleep analysis (LightGBM)
│   │
│   ├── models/                   ← Pre-trained .pkl files
│   │   ├── diabetes_model.pkl + diabetes_scaler.pkl
│   │   ├── heart_model.pkl + heart_scaler.pkl
│   │   ├── stroke_model.pkl + stroke_scaler.pkl
│   │   ├── invisible_burnout_random_forest.pkl
│   │   └── sleep_root_cause_lightgbm.pkl
│   │
│   ├── rag/
│   │   ├── rag_service.py        ← FAISS search + embeddings
│   │   ├── health_guidlines.pdf  ← Medical knowledge base
│   │   └── Dynamic_Workout_Planner.pdf
│   │
│   ├── data/health_rag_db/
│   │   ├── index.faiss           ← Vector index (auto-built)
│   │   └── index.pkl             ← Document chunks (auto-built)
│   │
│   └── tools/
│       └── agent_tools.py        ← AI Agent helper functions
│
└── ML_models/                    ← Jupyter Notebooks (Training)
    ├── diabetes.ipynb
    ├── heart.ipynb
    ├── Stroke.ipynb
    ├── Invisible Burnout Score.ipynb
    ├── Sleep Root-Cause Explainer.ipynb
    └── Feature_Engineering_ML.ipynb
```

---

*Made with ❤️ for Health & AI*
