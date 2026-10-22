"""
Network Intrusion Detection web app (Flask backend).

Run:   python app.py        then open http://127.0.0.1:5000
First run trains the models automatically (about 1-3 minutes).
"""
import io
import json
import os
import random

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

import train as trainer

ART = trainer.ART_DIR
MAX_ROWS = 20000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB uploads

STATE = {}


# ----------------------------------------------------------------- model loading
def load_state():
    meta_path = os.path.join(ART, "meta.json")
    if not os.path.exists(meta_path):
        print("No trained models found - training now (first run only)...")
        trainer.main()
    with open(meta_path) as fh:
        meta = json.load(fh)
    STATE["meta"] = meta
    STATE["rf"] = joblib.load(os.path.join(ART, "rf.joblib"))
    STATE["scaler"] = joblib.load(os.path.join(ART, "scaler.joblib"))
    if meta["dnn_backend"] == "keras":
        from tensorflow import keras
        STATE["dnn"] = keras.models.load_model(os.path.join(ART, "dnn.keras"))
    else:
        STATE["dnn"] = joblib.load(os.path.join(ART, "dnn_mlp.joblib"))
    with open(os.path.join(ART, "samples.json")) as fh:
        STATE["samples"] = json.load(fh)
    print(f"Models loaded (DNN backend: {meta['dnn_backend']}).")


def dnn_proba(X):
    if STATE["meta"]["dnn_backend"] == "keras":
        return STATE["dnn"].predict(X, verbose=0).ravel()
    return STATE["dnn"].predict_proba(X)[:, 1]


def prepare(df):
    """Raw feature table -> scaled numpy array ready for both models."""
    meta = STATE["meta"]
    names = meta["feature_names"]
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing[:8])
                         + (" ..." if len(missing) > 8 else ""))
    X = df[names].copy()
    for col, cats in meta["categories"].items():
        mapped = X[col].astype(str).str.strip().map({v: i for i, v in enumerate(cats)})
        if mapped.isnull().any():
            bad = X[col][mapped.isnull()].iloc[0]
            raise ValueError(f"Unknown value '{bad}' in column '{col}'.")
        X[col] = mapped
    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Non-numeric value found in a numeric column ({exc}).")
    if X.isnull().any().any():
        raise ValueError("Empty values found - every feature needs a value.")
    return STATE["scaler"].transform(X.values.astype(float))


def label_of(p):
    return "anomaly" if p >= 0.5 else "normal"


def run_models(df):
    X = prepare(df)
    return STATE["rf"].predict_proba(X)[:, 1], dnn_proba(X)


# ----------------------------------------------------------------- pages / API
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summary")
def summary():
    return jsonify(STATE["meta"])


@app.route("/api/sample")
def sample():
    want = request.args.get("label", "any")
    pool = [s for s in STATE["samples"] if want == "any" or s["class"].lower() == want]
    row = dict(random.choice(pool))
    actual = str(row.pop("class")).lower()
    return jsonify({"features": row, "actual": actual})


@app.route("/api/sample_csv")
def sample_csv():
    df = pd.DataFrame(STATE["samples"]).sample(frac=1, random_state=7)
    return Response(df.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sample_connections.csv"})


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    if not isinstance(features, dict):
        return jsonify(error="Send JSON like {\"features\": {...}}"), 400
    try:
        rf_p, dnn_p = run_models(pd.DataFrame([features]))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    rf_p, dnn_p = float(rf_p[0]), float(dnn_p[0])
    return jsonify({
        "rf": {"probability": rf_p, "label": label_of(rf_p)},
        "dnn": {"probability": dnn_p, "label": label_of(dnn_p)},
        "agreement": label_of(rf_p) == label_of(dnn_p),
    })


@app.route("/api/predict_csv", methods=["POST"])
def predict_csv():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify(error="Please choose a CSV file."), 400
    try:
        df = pd.read_csv(file)
    except Exception:
        return jsonify(error="Could not read that file as CSV."), 400
    truncated = len(df) > MAX_ROWS
    df = df.head(MAX_ROWS).reset_index(drop=True)
    try:
        rf_p, dnn_p = run_models(df)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    rf_l = ["anomaly" if p >= 0.5 else "normal" for p in rf_p]
    dnn_l = ["anomaly" if p >= 0.5 else "normal" for p in dnn_p]

    actual = None
    if "class" in df.columns:
        actual = df["class"].astype(str).str.strip().str.lower()
        if not actual.isin(["normal", "anomaly"]).all():
            actual = None

    if request.args.get("download") == "1":
        out = df.copy()
        out["rf_probability"] = rf_p.round(4)
        out["rf_prediction"] = rf_l
        out["dnn_probability"] = dnn_p.round(4)
        out["dnn_prediction"] = dnn_l
        return Response(out.to_csv(index=False), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=nids_predictions.csv"})

    summ = {
        "rows": int(len(df)),
        "truncated": truncated,
        "rf_anomalies": rf_l.count("anomaly"),
        "dnn_anomalies": dnn_l.count("anomaly"),
        "models_agree": sum(a == b for a, b in zip(rf_l, dnn_l)),
    }
    if actual is not None:
        summ["actual_anomalies"] = int((actual == "anomaly").sum())
        summ["rf_accuracy"] = float((actual.values == pd.Series(rf_l).values).mean())
        summ["dnn_accuracy"] = float((actual.values == pd.Series(dnn_l).values).mean())

    rows = []
    for i in range(min(200, len(df))):
        rows.append({"row": i + 1,
                     "rf_probability": float(rf_p[i]), "rf_label": rf_l[i],
                     "dnn_probability": float(dnn_p[i]), "dnn_label": dnn_l[i],
                     "actual": None if actual is None else actual.iloc[i]})
    return jsonify({"summary": summ, "rows": rows})


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large (limit 25 MB)."), 413


if __name__ == "__main__":
    load_state()
    print("\n  Open http://127.0.0.1:5000 in your browser\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
