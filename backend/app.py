"""
Flask REST API Backend for Lung Cancer Detection System
Endpoints:
  - POST /predict
  - POST /preprocess
  - GET  /model-info
  - GET  /health
  - GET  /history
  - DELETE /history/<record_id>
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from predict import encode_array_to_png_data_url, predict_ct_scan
from preprocessing import preprocess_ct_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_EVAL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "results", "evaluation_metrics.json"))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm", "nii", "gz"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB limit
CORS(app)


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_used TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                thumbnail_png TEXT NOT NULL,
                gradcam_png TEXT NOT NULL
            )
            """
        )
        conn.commit()


def is_allowed_file(filename: str) -> bool:
    if filename.lower().endswith(".nii.gz"):
        return True
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health():
    available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith((".keras", ".h5"))]
    return jsonify({
        "status": "ok",
        "service": "PulmoVision AI Flask Backend",
        "available_models": available_models,
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    if not os.path.exists(RESULTS_EVAL_PATH):
        return jsonify({
            "error": "Evaluation metrics not found. Execute training/evaluate.py first."
        }), 404
    with open(RESULTS_EVAL_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return jsonify(data)


@app.route("/preprocess", methods=["POST"])
def preprocess_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No CT image uploaded under 'image' form field."}), 400
    file = request.files["image"]
    if not file.filename or not is_allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: PNG, JPG, JPEG, DCM, NII."}), 400

    filter_mode = request.form.get("filterMode", "gaussian")
    safe_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    temp_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(temp_path)

    try:
        prep = preprocess_ct_pipeline(temp_path, filter_mode=filter_mode)
        return jsonify({
            "metadata": prep["metadata"],
            "filterMode": filter_mode,
            "outputs": {
                "originalPngDataUrl": encode_array_to_png_data_url(prep["original_resized"]),
                "denoisedPngDataUrl": encode_array_to_png_data_url(prep["denoised"]),
                "clahePngDataUrl": encode_array_to_png_data_url(prep["clahe"]),
                "edgesPngDataUrl": encode_array_to_png_data_url(prep["edges"]),
                "roiMaskPngDataUrl": encode_array_to_png_data_url(prep["lung_roi_mask"]),
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No CT image uploaded under 'image' form field."}), 400
    file = request.files["image"]
    if not file.filename or not is_allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: PNG, JPG, JPEG, DCM, NII."}), 400

    model_id = request.form.get("modelId", "resnet50")
    filter_mode = request.form.get("filterMode", "gaussian")
    safe_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    temp_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(temp_path)

    try:
        result = predict_ct_scan(temp_path, MODEL_DIR, model_id=model_id, filter_mode=filter_mode)
        record_id = f"scan_{uuid.uuid4().hex[:10]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO prediction_history (id, timestamp, model_used, predicted_class, confidence, probabilities_json, thumbnail_png, gradcam_png)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    timestamp,
                    model_id,
                    result["predicted_class"],
                    result["confidence"],
                    json.dumps(result["probabilities"]),
                    result["visualizations"]["original_png"],
                    result["visualizations"]["gradcam_overlay_png"],
                ),
            )
            conn.commit()

        result["id"] = record_id
        result["timestamp"] = timestamp
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/history", methods=["GET"])
def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, model_used, predicted_class, confidence, probabilities_json, thumbnail_png, gradcam_png
            FROM prediction_history
            ORDER BY timestamp DESC
            LIMIT 50
            """
        ).fetchall()

    records = [
        {
            "id": r[0],
            "timestamp": r[1],
            "modelUsed": r[2],
            "predictedClass": r[3],
            "confidence": r[4],
            "probabilities": json.loads(r[5]),
            "thumbnailDataUrl": r[6],
            "gradcamOverlayDataUrl": r[7],
        }
        for r in rows
    ]
    return jsonify({"count": len(records), "history": records})


@app.route("/history/<record_id>", methods=["DELETE"])
def delete_history(record_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
        conn.commit()
        if cur.rowcount == 0:
            return jsonify({"error": "Record not found."}), 404
    return jsonify({"deletedId": record_id})


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("FLASK_PORT", 5000)), debug=False)
