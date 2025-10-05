# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import traceback
import os
import warnings

# Suppress sklearn warnings
try:
    from sklearn.base import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# Try importing LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    lgb_import_error = None
except Exception as e:
    lgb = None
    LGB_AVAILABLE = False
    lgb_import_error = str(e)

app = Flask(__name__, template_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

MODEL_PATH = "model.pkl"
preloaded_model = None
model_load_error = None
model_loaded_once = False

# Required features for prediction
REQUIRED_FEATURES = [
    'orbital_period', 
    'transit_depth', 
    'transit_duration', 
    'planet_radius', 
    'stellar_temperature'
]

# Label mapping
LABEL_MAP = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}


def try_load_model():
    """Attempt to load the model from disk."""
    global preloaded_model, model_load_error, model_loaded_once

    if preloaded_model is not None:
        return

    if model_loaded_once:
        return

    model_loaded_once = True

    if not os.path.exists(MODEL_PATH):
        model_load_error = f"No model file found at {MODEL_PATH}."
        print("⚠️", model_load_error)
        return

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preloaded_model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully from", MODEL_PATH)
    except Exception as e:
        preloaded_model = None
        tb = traceback.format_exc()
        model_load_error = str(e)
        print("⚠️ Failed to load model:", model_load_error)
        if "lib_lightgbm" in tb or "libomp" in tb:
            print("ℹ️ LightGBM / libomp missing. On macOS: brew install libomp && pip install --force-reinstall lightgbm")
        print(tb)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process():
    """
    Process uploaded Excel file for prediction or retraining.
    Expects multipart/form-data with:
      - file: Excel file (.xlsx)
      - mode: 'predict' or 'retrain'
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        mode = request.form.get('mode', 'predict')

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith('.xlsx'):
            return jsonify({"error": "Only .xlsx files are supported"}), 400

        # Read Excel file
        try:
            df = pd.read_excel(file, sheet_name='Data')
        except Exception as e:
            return jsonify({"error": f"Failed to read Excel file: {str(e)}"}), 400

        if df.empty:
            return jsonify({"error": "Excel file is empty"}), 400

        # Handle different modes
        if mode == "predict":
            return handle_prediction(df)
        elif mode == "retrain":
            return handle_retraining(df)
        else:
            return jsonify({"error": "Invalid mode. Use 'predict' or 'retrain'"}), 400

    except Exception as e:
        tb = traceback.format_exc()
        print("❌ Exception in /api/process:", tb)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


def handle_prediction(df):
    """Handle prediction mode."""
    try_load_model()

    if preloaded_model is None:
        details = {
            "model_path_exists": os.path.exists(MODEL_PATH),
            "model_load_error": model_load_error,
            "lightgbm_available": LGB_AVAILABLE,
        }
        return jsonify({
            "error": "Model is not available for prediction.",
            "details": details,
            "advice": ["Place trained model at model.pkl", "Ensure LightGBM is installed: pip install lightgbm"]
        }), 500

    # Check for required features
    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        return jsonify({
            "error": f"Missing required columns: {', '.join(missing)}",
            "required": REQUIRED_FEATURES,
            "found": list(df.columns)
        }), 400

    # Extract features
    X = df[REQUIRED_FEATURES].values

    # Handle missing values
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)

    try:
        # Get predictions
        predictions = preloaded_model.predict(X)
        
        # Get probabilities if available
        try:
            probabilities = preloaded_model.predict_proba(X)
            confidences = np.max(probabilities, axis=1)
        except:
            confidences = np.ones(len(predictions))

        # Format results
        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            pred_label = LABEL_MAP.get(int(pred), f"Class {int(pred)}")
            results.append({
                "predicted_label": pred_label,
                "confidence": float(conf)
            })

        return jsonify({
            "mode": "predict",
            "results": results,
            "used_features": REQUIRED_FEATURES
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("❌ Prediction failed:", tb)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


def handle_retraining(df):
    """Handle fine-tuning mode - loads existing model and continues training."""
    # Load existing model first
    try_load_model()
    
    if preloaded_model is None:
        return jsonify({
            "error": "Cannot fine-tune: base model not loaded",
            "details": model_load_error,
            "advice": ["Ensure model.pkl exists and is loadable"]
        }), 500
    
    if not LGB_AVAILABLE:
        return jsonify({
            "error": "LightGBM not available. Fine-tuning requires LightGBM.",
            "how_to_fix": [
                "On macOS: brew install libomp",
                "Then: pip3 install --force-reinstall lightgbm"
            ],
            "lightgbm_import_error": lgb_import_error
        }), 500

    # Check for required columns
    if 'Disposition_Using_Kepler_Data' not in df.columns:
        return jsonify({
            "error": "Missing 'Disposition_Using_Kepler_Data' column required for fine-tuning"
        }), 400

    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        return jsonify({
            "error": f"Missing required feature columns: {', '.join(missing)}"
        }), 400

    try:
        # Prepare data
        X = df[REQUIRED_FEATURES].values
        y_raw = df['Disposition_Using_Kepler_Data'].str.upper()
        
        # Map labels to integers
        label_to_int = {'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2}
        y = y_raw.map(label_to_int)
        
        if y.isna().any():
            return jsonify({"error": "Invalid labels found. Expected: CONFIRMED, CANDIDATE, FALSE POSITIVE"}), 400

        y = y.values.astype(int)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Continue training from existing model
        dtrain = lgb.Dataset(X, label=y)
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbosity": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1
        }
        
        # Train additional rounds on top of existing model
        booster = lgb.train(
            params, 
            dtrain, 
            num_boost_round=100,
            init_model=preloaded_model  # Continue from existing model
        )
        booster.save_model(MODEL_PATH)
        
        # Clear cached model so next prediction loads the updated one
        global preloaded_model, model_load_error, model_loaded_once
        preloaded_model = None
        model_load_error = None
        model_loaded_once = False

        return jsonify({
            "mode": "retrain",
            "message": "✅ Model fine-tuned and saved successfully",
            "samples_trained": len(X),
            "results": []
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("❌ Fine-tuning failed:", tb)
        return jsonify({"error": "Fine-tuning failed", "details": str(e)}), 500


if __name__ == "__main__":
    print("\n🚀 Starting Flask server...")
    if not LGB_AVAILABLE:
        print("⚠️ LightGBM import failed:", lgb_import_error)
        print("\n💡 To enable fine-tuning on macOS:")
        print("    brew install libomp")
        print("    pip3 install --force-reinstall lightgbm")
    else:
        print("✅ LightGBM available and ready.")

    if os.path.exists(MODEL_PATH):
        print(f"ℹ️ Model file detected at {MODEL_PATH}.")
    else:
        print(f"ℹ️ No model file at {MODEL_PATH}. Place your trained model there.")

    app.run(host="0.0.0.0", port=5000, debug=True)
