#!/usr/bin/env python
"""
 @file   gui.py
 @brief  Web GUI for MIMII baseline — train, view stats, test.
"""
import os
import sys
import glob
import json
import threading
import time
import logging as py_logging

import numpy as np
import librosa
import yaml
from flask import Flask, render_template, request, jsonify
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from baseline import (
    dataset_generator, list_to_vector_array, file_to_vector_array,
    load_pickle, save_pickle, ConditionalVAE,
    cond_key_to_idx, make_onehot,
)

# Suppress Flask's default logging
py_logging.getLogger("werkzeug").setLevel(py_logging.WARNING)

app = Flask(__name__)

state = {
    "training": False,
    "current": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "eta_seconds": None,
    "progress": [],
    "logs": [],
    "log_seq": 0,
    "results": {},
}

param = None


def load_config():
    global param
    with open("baseline.yaml") as f:
        param = yaml.safe_load(f)


def find_target_dirs(base):
    """Find all directories containing normal/ and abnormal/ subdirs (any depth)."""
    base = os.path.abspath(base)
    targets = []
    for root, dirs, files in os.walk(base):
        if 'normal' in dirs and 'abnormal' in dirs:
            targets.append(root)
            dirs[:] = [d for d in dirs if d not in ('normal', 'abnormal')]
    return sorted(targets)


def get_available_datasets():
    load_config()
    base = os.path.abspath(param["base_directory"])
    if not os.path.exists(base):
        return []
    dirs = find_target_dirs(base)
    datasets = []
    for d in dirs:
        rel = os.path.relpath(d, base).replace(os.sep, '/')
        parts = rel.split('/')
        if len(parts) >= 3:
            # MIMII standard: {db}/{machine_type}/{machine_id}
            db = parts[-3]
            machine_type = parts[-2]
            machine_id = parts[-1]
        elif len(parts) == 2:
            # combined: {type}_{db}/{machine_id}
            machine_id = parts[-1]
            combined = parts[-2]
            machine_type = combined.split('_')[-1]
            db = '_'.join(combined.split('_')[:-1]) if '_' in combined else combined
        else:
            machine_id = parts[-1]
            machine_type = machine_id
            db = "unknown"
        datasets.append({
            "path": d,
            "db": db,
            "machine_type": machine_type,
            "machine_id": machine_id,
            "key": f"{machine_type}_{machine_id}_{db}",
        })
    return datasets


def get_trained_models():
    load_config()
    model_dir = param["model_directory"]
    if not os.path.exists(model_dir):
        return []
    files = sorted(glob.glob(f"{model_dir}/model_*.pth"))
    models = []
    for f in files:
        name = os.path.basename(f).replace("model_", "").replace(".pth", "")
        mtime = os.path.getmtime(f)
        models.append({"key": name, "file": f, "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))})
    return models





def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    state["logs"].append(f"[{timestamp}] {msg}")
    if len(state["logs"]) > 2000:
        state["logs"] = state["logs"][-1000:]
    state["log_seq"] += 1


def background_train(targets=None):
    """Train a single ConditionalVAE on all datasets combined."""
    base = os.path.abspath(param["base_directory"])
    all_ds = get_available_datasets()
    if targets:
        all_ds = [d for d in all_ds if d["key"] in targets]
    if not all_ds:
        log("No datasets for conditional training")
        return

    keys = [d["key"] for d in all_ds]
    log(f"Conditional VAE on {len(keys)} datasets: {', '.join(keys)}")

    # Load / generate data for all datasets
    all_data = []
    all_labels = []  # cond indices
    for d in all_ds:
        key = d["key"]
        log(f"[{key}] Loading...")
        train_pickle = f"{param['pickle_directory']}/train_{key}.pickle"
        if os.path.exists(train_pickle):
            data = load_pickle(train_pickle)
        else:
            train_files, _, _, _ = dataset_generator(d["path"])
            if len(train_files) == 0:
                log(f"[{key}] SKIP: empty")
                continue
            data = list_to_vector_array(
                train_files, msg=f"[{key}] feat",
                n_mels=param["feature"]["n_mels"], frames=param["feature"]["frames"],
                n_fft=param["feature"]["n_fft"], hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"],
            )
            save_pickle(train_pickle, data)
        if len(data) == 0:
            log(f"[{key}] SKIP: no data")
            continue
        idx = keys.index(key)
        all_data.append(data)
        all_labels.extend([idx] * len(data))

    if len(all_data) == 0:
        log("No data for conditional training")
        return

    combined = np.concatenate(all_data, axis=0)
    cond_indices = np.array(all_labels)
    log(f"Combined data: {combined.shape}, {len(keys)} conditions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
    bottleneck_size = param["fit"].get("bottleneck_size", 16)
    cond_dim = len(keys)
    model = ConditionalVAE(input_dim, cond_dim=cond_dim, bottleneck_size=bottleneck_size).to(device)

    use_norm = param["fit"].get("normalize", False)
    scaler = StandardScaler().fit(combined) if use_norm else None
    if scaler:
        combined = scaler.transform(combined)

    epochs = param["fit"]["epochs"]
    beta = param["fit"].get("beta", 0.1)
    alpha = param["fit"].get("alpha", 0.1)
    denoising_std = param["fit"].get("denoising_std", 0.0)

    train_t = torch.from_numpy(combined).float()
    cond_t = torch.from_numpy(cond_indices).long()
    dataset = TensorDataset(train_t, cond_t)
    val_len = int(len(dataset) * param["fit"].get("validation_split", 0.1))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=param["fit"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=param["fit"]["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=param["fit"].get("learning_rate", 0.001))
    recon_criterion = nn.MSELoss()

    log(f"Training ConditionalVAE ({epochs} epochs, α={alpha}, β={beta})...")
    state["total_epochs"] = epochs
    train_start = time.time()
    epoch_times = []

    for epoch in range(epochs):
        state["current_epoch"] = epoch + 1
        if not state["training"]:
            log("Training stopped")
            return
        model.train()
        total_loss = 0
        n_batches = 0
        for bx, bc in train_loader:
            bx, bc = bx.to(device), bc.to(device)
            cond = torch.zeros(bx.size(0), cond_dim, device=device).scatter_(1, bc.unsqueeze(1), 1)
            if denoising_std > 0:
                noisy_x = bx + torch.randn_like(bx) * denoising_std
            else:
                noisy_x = bx
            recon, mu, log_var, logits = model(noisy_x, cond)
            recon_loss = recon_criterion(recon, bx)
            kl = model.kl_loss(mu, log_var).mean()
            cls_loss = model.classifier_loss(logits, bc)
            loss = recon_loss + beta * kl + alpha * cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            n_batches += 1
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        total_val = 0
        with torch.no_grad():
            for bx, bc in val_loader:
                bx, bc = bx.to(device), bc.to(device)
                cond = torch.zeros(bx.size(0), cond_dim, device=device).scatter_(1, bc.unsqueeze(1), 1)
                recon, mu, log_var, _ = model(bx, cond)
                recon_loss = recon_criterion(recon, bx)
                kl = model.kl_loss(mu, log_var).mean()
                loss = recon_loss + beta * kl
                total_val += loss.item() * bx.size(0)
        val_loss = total_val / len(val_loader.dataset)

        elapsed = time.time() - train_start
        epoch_times.append(elapsed / (epoch + 1))
        remaining = epochs - (epoch + 1)
        state["eta_seconds"] = int(remaining * epoch_times[-1])

        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            log(f"[CondVAE] {epoch+1:5d}/{epochs}  loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # Save model
    model_file = f"{param['model_directory']}/model_conditional.pth"
    torch.save(model.state_dict(), model_file)
    if scaler:
        save_pickle(f"{param['pickle_directory']}/scaler_conditional.pickle", scaler)
    log(f"ConditionalVAE saved to {model_file}")

    # Evaluate each dataset
    model.eval()
    score_mode = param["fit"].get("score_mode", "mean")
    result_file = f"{param['result_directory']}/{param['result_file']}"
    os.makedirs(param["result_directory"], exist_ok=True)
    saved = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            saved = yaml.safe_load(f) or {}

    log("Evaluating all datasets...")
    for d in all_ds:
        key = d["key"]
        if key not in keys:
            continue
        idx = keys.index(key)
        _, _, eval_files, eval_labels = dataset_generator(d["path"])
        if len(eval_files) == 0:
            log(f"[{key}] SKIP eval: empty")
            continue
        state["current"] = key
        y_pred = np.zeros(len(eval_labels))
        for i, fn in enumerate(eval_files):
            if not state["training"]:
                return
            data = file_to_vector_array(
                fn, n_mels=param["feature"]["n_mels"], frames=param["feature"]["frames"],
                n_fft=param["feature"]["n_fft"], hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"],
            )
            if data.shape[0] == 0:
                y_pred[i] = 0
                continue
            if scaler:
                data = scaler.transform(data)
            dt = torch.from_numpy(data).float().to(device)
            cond_vec = make_onehot(idx, cond_dim).to(device).unsqueeze(0).expand(dt.size(0), -1)
            with torch.no_grad():
                recon, _, _, logits = model(dt, cond_vec)
                frame_err = torch.mean((dt - recon) ** 2, dim=1).cpu().numpy()
            if score_mode == "max":
                y_pred[i] = float(np.max(frame_err))
            elif score_mode == "p95":
                y_pred[i] = float(np.percentile(frame_err, 95))
            else:
                y_pred[i] = float(np.mean(frame_err))

        auc = metrics.roc_auc_score(eval_labels, y_pred)
        pauc = metrics.roc_auc_score(eval_labels, y_pred, max_fpr=0.1)
        log(f"[{key}] AUC = {auc:.6f}  pAUC = {pauc:.6f}")
        stats = {"AUC": float(auc), "pAUC": float(pauc)}
        saved[key] = stats
        state["results"][key] = stats

        fpr, tpr, ths = metrics.roc_curve(eval_labels, y_pred)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        stats["EER"] = float(fpr[eer_idx])
        youden = tpr - fpr
        best_i = int(np.argmax(youden))
        stats["best_threshold"] = float(ths[best_i])
        pred_best = (np.array(y_pred) > ths[best_i]).astype(int)
        stats["best_F1"] = float(metrics.f1_score(eval_labels, pred_best, zero_division=0))

        # Config snapshot
        stats["config"] = {
            "model": "conditional_vae",
            "bottleneck": bottleneck_size,
            "beta": beta, "alpha": alpha,
            "denoising_std": denoising_std,
            "epochs": epochs, "score_mode": score_mode,
            "cond_dim": cond_dim,
        }

    with open(result_file, "w") as f:
        yaml.dump(saved, f)
    log("ConditionalVAE training done")


# ── Routes ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    load_config()
    if request.method == "POST":
        updates = request.get_json(silent=True) or {}
        for section, keys in updates.items():
            if section in param and isinstance(param[section], dict):
                for k, v in keys.items():
                    if k in param[section]:
                        param[section][k] = v
        with open("baseline.yaml", "w") as f:
            yaml.dump(param, f, default_flow_style=False)
        return jsonify({"status": "saved"})
    return jsonify(param)


@app.route("/api/datasets")
def api_datasets():
    try:
        return jsonify(get_available_datasets())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models")
def api_models():
    try:
        return jsonify(get_trained_models())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    if state["training"]:
        return jsonify({"error": "Training already in progress"}), 400
    data = request.get_json(silent=True) or {}
    targets = data.get("targets")
    thread = threading.Thread(target=background_train, args=(targets,), daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    state["training"] = False
    return jsonify({"status": "stopping"})


@app.route("/api/status")
def api_status():
    since = request.args.get("since", 0, type=int)
    log_start_seq = state["log_seq"] - len(state["logs"])
    if since < log_start_seq:
        since = log_start_seq
    new_logs = state["logs"][since - log_start_seq:] if since < state["log_seq"] else []
    return jsonify({
        "training": state["training"],
        "current": state["current"],
        "current_epoch": state["current_epoch"],
        "total_epochs": state["total_epochs"],
        "eta_seconds": state["eta_seconds"],
        "progress": state["progress"],
        "logs": new_logs,
        "log_seq": state["log_seq"],
        "results": state["results"],
    })


@app.route("/api/results")
def api_results():
    load_config()
    result_file = f"{param['result_directory']}/{param['result_file']}"
    saved = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            saved = yaml.safe_load(f) or {}
    merged = {**saved, **state["results"]}
    return jsonify(merged)


@app.route("/api/admin/check", methods=["POST"])
def api_admin_check():
    load_config()
    data = request.get_json(silent=True) or {}
    pw = data.get("password", "")
    correct = param.get("admin_password", "admin123")
    return jsonify({"ok": pw == correct})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get audio
    if "file" in request.files:
        f = request.files["file"]
        data_bytes = f.read()
        import io
        audio_data, sr = librosa.load(io.BytesIO(data_bytes), sr=None, mono=False)
        model_key = request.form.get("model_key", "")
    else:
        body = request.get_json(silent=True) or {}
        filepath = body.get("path", "")
        if not filepath or not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        audio_data, sr = librosa.load(filepath, sr=None, mono=False)
        model_key = body.get("model_key", "")

    if audio_data.ndim > 1:
        audio_data = audio_data[0, :]

    # Load model
    if not model_key:
        models = get_trained_models()
        if not models:
            return jsonify({"error": "No trained models found. Train a model first."}), 404
        model_key = models[0]["key"]

    input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
    bottleneck_size = param["fit"].get("bottleneck_size", 16)

    # Load ConditionalVAE model
    cond_model_path = f"{param['model_directory']}/model_conditional.pth"
    if not os.path.exists(cond_model_path):
        return jsonify({"error": "No trained model found. Train a ConditionalVAE first."}), 404
    # Get cond_dim from saved results
    cond_dim = 48
    result_file = f"{param['result_directory']}/{param['result_file']}"
    if os.path.exists(result_file):
        with open(result_file) as f:
            saved = yaml.safe_load(f) or {}
        for v in saved.values():
            if isinstance(v, dict) and v.get("config", {}).get("cond_dim"):
                cond_dim = v["config"]["cond_dim"]
                break
    model = ConditionalVAE(input_dim, cond_dim=cond_dim, bottleneck_size=bottleneck_size).to(device)
    model.load_state_dict(torch.load(cond_model_path, map_location=device, weights_only=True))
    # Get the condition index for this model_key
    all_keys = [d["key"] for d in get_available_datasets()]
    cond_idx = cond_key_to_idx(model_key, all_keys) if all_keys else 0
    if cond_idx >= cond_dim:
        cond_idx = cond_idx % cond_dim
    model.eval()

    mel = librosa.feature.melspectrogram(
        y=audio_data, sr=sr,
        n_fft=param["feature"]["n_fft"],
        hop_length=param["feature"]["hop_length"],
        n_mels=param["feature"]["n_mels"],
        power=param["feature"]["power"],
    )
    log_mel = 20.0 / param["feature"]["power"] * np.log10(mel + sys.float_info.epsilon)
    vec_frames = param["feature"]["frames"]
    vec_size = log_mel.shape[1] - vec_frames + 1
    if vec_size < 1:
        return jsonify({"error": f"Audio too short ({log_mel.shape[1]} frames, need >= {vec_frames})"}), 400
    vec = np.zeros((vec_size, input_dim), float)
    for t in range(vec_frames):
        vec[:, param["feature"]["n_mels"] * t: param["feature"]["n_mels"] * (t + 1)] = log_mel[:, t: t + vec_size].T

    dt = torch.from_numpy(vec).float().to(device)
    with torch.no_grad():
        cond_vec = make_onehot(cond_idx, cond_dim).to(device).unsqueeze(0).expand(dt.size(0), -1)
        out = model(dt, cond_vec)
        recon = out[0]
        frame_errors = torch.mean((dt - recon) ** 2, dim=1).cpu().numpy()

    score = float(np.max(frame_errors))

    # Load best_threshold from evaluation results
    best_threshold = None
    result_file = f"{param['result_directory']}/{param['result_file']}"
    if os.path.exists(result_file):
        with open(result_file) as f:
            saved = yaml.safe_load(f) or {}
        if model_key in saved and "best_threshold" in saved[model_key]:
            best_threshold = saved[model_key]["best_threshold"]

    return jsonify({
        "score": score,
        "score_str": f"{score:.6f}",
        "best_threshold": best_threshold,
        "model_key": model_key,
        "num_frames": len(frame_errors),
    })


if __name__ == "__main__":
    load_config()
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting MIMII GUI at http://127.0.0.1:{port}")
    app.template_folder = "templates"
    app.jinja_env.auto_reload = True
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(host="127.0.0.1", port=port, debug=True, threaded=True)
