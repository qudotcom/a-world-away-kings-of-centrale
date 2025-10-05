```md
# 🚀 ExoVision AI — README

**ExoVision AI** is a hackathon-ready MVP that classifies **Kepler Objects of Interest (KOI)** using a pre-trained **LightGBM** model. It also gives users the option to **fine-tune the existing model** in-memory with their own labeled dataset — no retraining from scratch, just incremental learning.

This project combines:
- a **modern front-end** (HTML, CSS, JS)
- a **robust Flask backend**
- and a **pre-trained LightGBM classifier**

Designed and built by **Kings of Centrale** 🏆 — where vision meets innovation.

---

## 🧭 Table of Contents

- [✨ Features](#-features)
- [📂 Project Structure](#-project-structure)
- [⚙️ Requirements](#️-requirements)
- [🚀 Quickstart](#-quickstart)
- [🧠 How It Works](#-how-it-works)
- [📊 Data Format](#-data-format)
- [📡 API Endpoint](#-api-endpoint)
- [🧩 Example Requests](#-example-requests)
- [🔁 Fine-Tuning Logic](#-fine-tuning-logic)
- [💻 Frontend Integration](#-frontend-integration)
- [🔐 Security Considerations](#-security-considerations)
- [⚡ Performance Tips](#-performance-tips)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [🌱 Future Improvements](#-future-improvements)
- [👑 Credits](#-credits)
- [📜 License](#-license)

---

## ✨ Features

- 🧬 **Pre-trained LightGBM classifier** (`model.pkl`)  
- 🧠 **Fine-tuning mode** to improve predictions with labeled data  
- 📈 **Automated classification** of KOIs: *Confirmed*, *Candidate*, *False Positive*  
- ⚡ **In-memory retraining** for fast hackathon demos  
- 🌌 **Space-themed interface** with modern UI/UX  
- 📁 **Excel (.xlsx)** file upload with drag-and-drop support  
- 🔁 **JSON output** for easy front-end integration  

---

## 📂 Project Structure

```

exo_vision_ai/
├─ templates/
│  └─ index.html        # Frontend UI
├─ app.py               # Flask backend API
├─ model.pkl            # Pre-trained LightGBM model
├─ requirements.txt     # Python dependencies
└─ README.md

````

---

## ⚙️ Requirements

**Python ≥ 3.9**

```txt
flask>=3.0.0
werkzeug>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
````

---

## 🚀 Quickstart

1. **Clone the repository**

```bash
git clone https://github.com/your-username/exovision-ai.git
cd exovision-ai
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Ensure your model file exists**
   Place your pre-trained LightGBM model at the project root:

```
model.pkl
```

> 💡 To create one:

```python
from lightgbm import LGBMClassifier
import joblib

model = LGBMClassifier()
# model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
```

5. **Run the Flask app**

```bash
python app.py
```

6. **Open the web interface**

```
http://localhost:5000/
```

---

## 🧠 How It Works

The app supports two modes:

| Mode                   | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| 🛰️ **Classification** | Uses pre-trained `model.pkl` to classify KOIs.                 |
| 🪐 **Fine-tuning**     | Retrains (in-memory) the LightGBM model with labeled KOI data. |

Each upload triggers backend logic:

1. Parse `.xlsx` file via `pandas` + `openpyxl`
2. Extract numeric features
3. Use LightGBM classifier to predict or fine-tune
4. Return JSON with predictions + confidence scores

---

## 📊 Data Format

### 🛰️ **Classification Mode**

**Sheet name:** `Data`
**Required columns:**

* `orbital_period`
* `transit_depth`
* `transit_duration`
* `planet_radius`
* `stellar_temperature`

Optional columns are allowed (ignored if not numeric).

---

### 🪐 **Fine-Tuning Mode**

**Label column:** `Disposition_Using_Kepler_Data`
**Accepted values:**

* `CONFIRMED`
* `CANDIDATE`
* `FALSE POSITIVE`

> Case-insensitive — they’re normalized automatically.

---

### 🧾 Example Excel (CSV view)

```
orbital_period,transit_depth,transit_duration,planet_radius,stellar_temperature,Disposition_Using_Kepler_Data
365.24,0.12,2.3,1.1,5720,CONFIRMED
10.5,0.02,1.1,0.8,4500,CANDIDATE
```

---

## 📡 API Endpoint

**POST** `/api/process`
Accepts `multipart/form-data`

**Form Fields:**

| Field  | Type                  | Description    |
| ------ | --------------------- | -------------- |
| `file` | `.xlsx`               | KOI dataset    |
| `mode` | `predict` / `retrain` | Operation mode |

**Response (200 OK):**

```json
{
  "mode": "predict",
  "used_features": ["orbital_period", "transit_depth"],
  "results": [
    {"index": 0, "predicted_label": "CANDIDATE", "confidence": 0.78}
  ]
}
```

**Error:**

```json
{ "error": "Missing required columns" }
```

---

## 🧩 Example Requests

### 🔹 Curl (Predict)

```bash
curl -X POST http://localhost:5000/api/process \
  -F "file=@koi.xlsx" \
  -F "mode=predict"
```

### 🔹 Curl (Retrain)

```bash
curl -X POST http://localhost:5000/api/process \
  -F "file=@koi_labeled.xlsx" \
  -F "mode=retrain"
```

### 🔹 JavaScript

```js
const fd = new FormData();
fd.append('file', fileInput.files[0]);
fd.append('mode', 'predict');

const res = await fetch('/api/process', { method: 'POST', body: fd });
const data = await res.json();
console.log(data);
```

---

## 🔁 Fine-Tuning Logic

* Backend loads `model.pkl` at startup
* When `mode=retrain`:

  1. Extract labeled rows
  2. Match features to existing model
  3. Use LightGBM’s `train(init_model=...)` to continue boosting
  4. Keep fine-tuned model in memory
* Not persisted (use `joblib.dump()` manually if needed)

---

## 💻 Frontend Integration

🧩 **HTML UI** (`templates/index.html`):

* Drag-and-drop upload
* Mode selector (predict / retrain)
* Auto-calls `/api/process`
* Displays results in table format
* Styled with:

  * glowing accents
  * centered layout
  * animated starfield

---

## 🔐 Security Considerations

Before exposing publicly:

* ✅ Limit file size: `MAX_CONTENT_LENGTH = 10 * 1024 * 1024` (10MB)
* ✅ Validate `.xlsx` MIME type
* ✅ Sanitize filenames & data
* ✅ Escape outputs (already handled)
* ✅ Add authentication (if needed)
* ✅ Deploy with gunicorn + HTTPS

---

## ⚡ Performance Tips

* Increase LightGBM `num_boost_round` for better fine-tuning
* Use **combined datasets** (original + new) for meaningful retraining
* Offload retraining to background jobs (Celery/RQ)
* Cache predictions if repeatedly classifying similar datasets

---

## 🛠️ Troubleshooting

| Issue                      | Cause                           | Fix                               |
| -------------------------- | ------------------------------- | --------------------------------- |
| `Missing required columns` | Excel file doesn’t match schema | Rename columns properly           |
| `No model loaded`          | Missing `model.pkl`             | Place model file in root          |
| `Low accuracy`             | Unscaled / mismatched features  | Match feature names & scaling     |
| `Fine-tuning no change`    | Too few rows or epochs          | Increase labeled samples / rounds |

---

## 🌱 Future Improvements

* 🧾 Preview uploaded Excel file (first 5 rows)
* 🔄 Model versioning system
* 🧪 Unit tests for parsing & inference
* 📬 Async training jobs
* 📊 Add metrics visualization
* 🧠 Integrate SHAP for explainability

---

## 👑 Credits

Developed with passion by **Kings of Centrale**

> Where AI meets Astronomy — and Vision meets Innovation 🌌

---

## 📜 License

**MIT License**
Use freely for learning, hackathons, and research. Attribution appreciated.

```
```
