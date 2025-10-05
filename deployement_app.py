# deployement_app.py - multi-satellite model server (XGB + stacked TESS)
from flask import Flask, request, jsonify, Response
import os, json, joblib, traceback
from pathlib import Path
import numpy as np

try:
    from flask_cors import CORS
    HAS_FLASK_CORS = True
except Exception:
    HAS_FLASK_CORS = False

import lightgbm as lgb

# Use relative paths for model directories - CORRECTED PATH
MODEL_DIR = Path(__file__).parent / "All_Satellite_Models_xgb_stack_final"
TESS_MODEL_DIR = MODEL_DIR / "tess"
API_KEY = None

SATELLITES = {
    "kepler": {"basename": "kepler_koi", "dir": MODEL_DIR / "kepler"},
    "k2": {"basename": "k2_pandc", "dir": MODEL_DIR / "k2"},
    "tess": {"basename": "tess_toi", "dir": TESS_MODEL_DIR}
}

def load_obj(p):
    try:
        return joblib.load(p)
    except Exception:
        try:
            import pickle
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            raise

def load_store():
    print(f"=== DEBUG: load_store() ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent}")
    print(f"MODEL_DIR path: {MODEL_DIR}")
    print(f"MODEL_DIR exists: {MODEL_DIR.exists()}")

    # Check script directory contents
    script_dir = Path(__file__).parent
    if script_dir.exists():
        print(f"Script directory contents: {[str(p) for p in script_dir.iterdir()]}")

    store = {}
    for sat, info in SATELLITES.items():
        d = info["dir"]
        print(f"\n=== LOADING {sat.upper()} ===")
        print(f"[{sat}] Directory path: {d}")
        print(f"[{sat}] Directory exists: {d.exists()}")

        if d.exists():
            print(f"[{sat}] Directory contents: {list(d.iterdir())}")
        else:
            print(f"[{sat}] Directory does not exist!")

        store[sat] = {"dir": str(d), "files": {}}
        basename = info["basename"]

        if sat in ("kepler", "k2"):
            model_p = d / f"{basename}_xgb.pkl"
            imp_p = d / f"{basename}_imputer.joblib"
            sc_p = d / f"{basename}_scaler.joblib"
            feats_p = d / f"{basename}_features_list.json"
            meta_p = d / f"{basename}_metadata.json"
            calib_p = d / f"{basename}_calibrators.joblib"

            print(f"[{sat}] Looking for files:")
            for name, path in [("xgb", model_p), ("imputer", imp_p), ("scaler", sc_p), ("features", feats_p), ("metadata", meta_p), ("calibrators", calib_p)]:
                exists = path.exists()
                print(f"  {name}: {path} -> {exists}")

            for k, p in [("model", model_p), ("imputer", imp_p), ("scaler", sc_p), ("features", feats_p), ("metadata", meta_p), ("calibrators", calib_p)]:
                store[sat]["files"][k] = str(p) if p.exists() else None
            try: store[sat]["model"] = load_obj(store[sat]["files"]["model"]) if store[sat]["files"]["model"] else None
            except Exception as e: print(f"Load model error {sat}: {e}"); store[sat]["model"] = None
            try: store[sat]["imputer"] = load_obj(store[sat]["files"]["imputer"]) if store[sat]["files"]["imputer"] else None
            except Exception as e: print(f"Load imputer error {sat}: {e}"); store[sat]["imputer"] = None
            try: store[sat]["scaler"] = load_obj(store[sat]["files"]["scaler"]) if store[sat]["files"]["scaler"] else None
            except Exception as e: print(f"Load scaler error {sat}: {e}"); store[sat]["scaler"] = None
            try: store[sat]["features"] = json.load(open(store[sat]["files"]["features"])) if store[sat]["files"]["features"] else None
            except Exception as e: print(f"Load features error {sat}: {e}"); store[sat]["features"] = None
            try: store[sat]["metadata"] = json.load(open(store[sat]["files"]["metadata"])) if store[sat]["files"]["metadata"] else {}
            except Exception as e: print(f"Load metadata error {sat}: {e}"); store[sat]["metadata"] = {}
            try: store[sat]["calibrators"] = load_obj(store[sat]["files"]["calibrators"]) if store[sat]["files"]["calibrators"] else {}
            except Exception as e: print(f"Load calibrators error {sat}: {e}"); store[sat]["calibrators"] = {}
        else:  # tess stacking layout
            model_xgb = d / "tess_toi_xgb.pkl"
            model_lgb = d / "tess_toi_lgbm.txt"
            meta_m = d / "tess_toi_stack_meta.pkl"
            imp_p = d / "tess_toi_imputer.joblib"
            sc_p = d / "tess_toi_scaler.joblib"
            feats_p = d / "tess_toi_features_list.json"
            meta_p = d / "tess_toi_metadata.json"
            calib_p = d / "tess_toi_calibrators.joblib"

            print(f"[{sat}] Looking for files:")
            for name, path in [("xgb", model_xgb), ("lgb", model_lgb), ("meta", meta_m), ("imputer", imp_p), ("scaler", sc_p), ("features", feats_p), ("metadata", meta_p), ("calibrators", calib_p)]:
                exists = path.exists()
                print(f"  {name}: {path} -> {exists}")

            store[sat]["files"] = {
                "xgb": str(model_xgb) if model_xgb.exists() else None,
                "lgb": str(model_lgb) if model_lgb.exists() else None,
                "meta": str(meta_m) if meta_m.exists() else None,
                "imputer": str(imp_p) if imp_p.exists() else None,
                "scaler": str(sc_p) if sc_p.exists() else None,
                "features": str(feats_p) if feats_p.exists() else None,
                "metadata": str(meta_p) if meta_p.exists() else None,
                "calibrators": str(calib_p) if calib_p.exists() else None
            }
            try: store[sat]["xgb"] = load_obj(store[sat]["files"]["xgb"]) if store[sat]["files"]["xgb"] else None
            except Exception as e: print(f"Load xgb error {sat}: {e}"); store[sat]["xgb"] = None
            try: store[sat]["lgb"] = lgb.Booster(model_file=store[sat]["files"]["lgb"]) if store[sat]["files"]["lgb"] else None
            except Exception as e: print(f"Load lgb error {sat}: {e}"); store[sat]["lgb"] = None
            try: store[sat]["meta"] = load_obj(store[sat]["files"]["meta"]) if store[sat]["files"]["meta"] else None
            except Exception as e: print(f"Load meta error {sat}: {e}"); store[sat]["meta"] = None
            try: store[sat]["imputer"] = load_obj(store[sat]["files"]["imputer"]) if store[sat]["files"]["imputer"] else None
            except Exception as e: print(f"Load imputer error {sat}: {e}"); store[sat]["imputer"] = None
            try: store[sat]["scaler"] = load_obj(store[sat]["files"]["scaler"]) if store[sat]["files"]["scaler"] else None
            except Exception as e: print(f"Load scaler error {sat}: {e}"); store[sat]["scaler"] = None
            try: store[sat]["features"] = json.load(open(store[sat]["files"]["features"])) if store[sat]["files"]["features"] else None
            except Exception as e: print(f"Load features error {sat}: {e}"); store[sat]["features"] = None
            try: store[sat]["metadata"] = json.load(open(store[sat]["files"]["metadata"])) if store[sat]["files"]["metadata"] else {}
            except Exception as e: print(f"Load metadata error {sat}: {e}"); store[sat]["metadata"] = {}
            try: store[sat]["calibrators"] = load_obj(store[sat]["files"]["calibrators"]) if store[sat]["files"]["calibrators"] else {}
            except Exception as e: print(f"Load calibrators error {sat}: {e}"); store[sat]["calibrators"] = {}

    print(f"\n=== LOAD SUMMARY ===")
    for sat in store:
        print(f"{sat}: files found: {sum(1 for v in store[sat]['files'].values() if v is not None)}")
        print(f"  features loaded: {store[sat].get('features') is not None}")

    return store

STORE = load_store()

app = Flask(__name__)
if HAS_FLASK_CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors(resp):
    if not HAS_FLASK_CORS:
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-KEY"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

def check_key(req):
    if API_KEY is None: return True
    key = req.headers.get("X-API-KEY") or req.args.get("api_key")
    return str(API_KEY).strip().strip('"').strip("'") == str(key)

@app.route("/")
def index():
    return Response("<h3>Multi-satellite model API</h3>", mimetype="text/html")

@app.route("/debug")
def debug_paths():
    return jsonify({
        "cwd": os.getcwd(),
        "script_dir": str(Path(__file__).parent),
        "model_dir": str(MODEL_DIR),
        "model_dir_exists": MODEL_DIR.exists(),
        "script_dir_contents": [str(p) for p in Path(__file__).parent.iterdir()] if Path(__file__).parent.exists() else [],
        "satellites": {
            sat: {
                "dir": str(info["dir"]),
                "dir_exists": info["dir"].exists(),
                "features_file_exists": (info["dir"] / f"{info['basename']}_features_list.json").exists() if info["dir"].exists() else False
            }
            for sat, info in SATELLITES.items()
        }
    })

@app.route("/status")
def status():
    if not check_key(request): return jsonify({"error": "invalid_api_key"}), 401
    out = {}
    for sat in STORE:
        out[sat] = {
            "model_loaded": bool(STORE[sat].get("model") or STORE[sat].get("xgb")),
            "dir": STORE[sat]["dir"],
            "files": STORE[sat]["files"],
            "metadata": STORE[sat].get("metadata", {}),
            "features_loaded": STORE[sat].get("features") is not None
        }
    return jsonify(out)

def build_vector(features_list, features_dict):
    X = []
    for f in features_list:
        v = features_dict.get(f, None)
        if v is None:
            for k in features_dict.keys():
                if k.lower() == f.lower():
                    v = features_dict[k]
                    break
        try:
            if v is None or str(v).strip().lower() in ("", "nan", "none"):
                X.append(np.nan)
            else:
                X.append(float(v))
        except:
            X.append(np.nan)
    return np.array([X], dtype=float)

def apply_calibrators(calibs, classes, probs):
    out_probs = probs.copy()
    for i, c in enumerate(classes):
        iso = calibs.get(c)
        if iso is None: continue
        try:
            out_probs[0, i] = float(iso.transform([probs[0, i]])[0])
        except Exception:
            pass
    s = out_probs.sum()
    if s > 0:
        out_probs = out_probs / s
    return out_probs

@app.route("/features/<sat>")
def features(sat):
    if not check_key(request): return jsonify({"error": "invalid_api_key"}), 401
    sat = sat.lower()
    if sat not in STORE: return jsonify({"error": "unknown_satellite"}), 400
    feats = STORE[sat].get("features")
    if feats is None:
        print(f"ERROR: Features missing for {sat}. Store content: {STORE[sat]}")
        return jsonify({"error": "features_list_missing"}), 500
    return jsonify({"features": feats, "metadata": STORE[sat].get("metadata", {})})

@app.route("/predict/<sat>", methods=["POST"])
def predict(sat):
    if not check_key(request): return jsonify({"error": "invalid_api_key"}), 401
    sat = sat.lower()
    if sat not in STORE: return jsonify({"error": "unknown_satellite"}), 400
    body = request.get_json(force=True, silent=True)
    if not body or "features" not in body: return jsonify({"error": "missing_features_field"}), 400
    features_dict = body["features"]
    store = STORE[sat]
    feats = store.get("features")
    if feats is None: return jsonify({"error": "features_list_missing"}), 500

    Xraw = build_vector(feats, features_dict)

    try:
        imputer = store.get("imputer")
        scaler = store.get("scaler")
        if imputer is not None: X_proc = imputer.transform(Xraw)
        else: X_proc = Xraw
        if scaler is not None: X_proc = scaler.transform(X_proc)
    except Exception as e:
        return jsonify({"error": "transform_error", "detail": str(e), "trace": traceback.format_exc()}), 500

    try:
        if sat in ("kepler", "k2"):
            model = store.get("model")
            if model is None: return jsonify({"error": "model_missing"}), 500
            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_proc)[0]
            else:
                pred = model.predict(X_proc)[0]
                probs = np.array([1.0])
            classes = store.get("metadata", {}).get("classes") or (getattr(model, "classes_", None).tolist() if hasattr(model, "classes_") else None)
            if classes is None:
                classes = [str(i) for i in range(len(probs))]
            calibs = store.get("calibrators", {}) or {}
            probs = apply_calibrators(calibs, classes, np.array([probs]))[0]
            meta = store.get("metadata", {})
            thresholds = meta.get("class_thresholds", {})
            pred_class = None
            pred_prob = None
            candidates = []
            for i, c in enumerate(classes):
                thr = thresholds.get(c)
                if thr is not None and probs[i] >= thr:
                    candidates.append((c, float(probs[i])))
            if candidates:
                pred_class, pred_prob = max(candidates, key=lambda x: x[1])
            else:
                idx = int(np.argmax(probs))
                pred_class = classes[idx]
                pred_prob = float(probs[idx])

            if sat == "kepler":
                koi_label = pred_class
                is_exo = koi_label.strip().upper() != "FALSE POSITIVE"
                return jsonify({"probability": float(pred_prob), "koi_pdisposition": koi_label, "is_exoplanet": bool(is_exo)})
            else:  # k2
                archive_disp = pred_class
                planet_type = "exo" if archive_disp.strip().upper() != "FALSE POSITIVE" else "not_exo"
                return jsonify({"probability": float(pred_prob), "archive_disposition": archive_disp, "planet_type": planet_type})

        elif sat == "tess":
            xgbm = store.get("xgb")
            lgbm = store.get("lgb")
            meta = store.get("meta")
            classes = store.get("metadata", {}).get("classes") or []
            if xgbm is None or lgbm is None or meta is None:
                return jsonify({"error": "tess_models_missing"}), 500
            p1 = xgbm.predict_proba(X_proc)
            p2 = lgbm.predict(X_proc)
            if isinstance(p2, np.ndarray) and p2.ndim == 1:
                p2 = np.vstack([1 - p2, p2]).T
            if p1.shape[1] != len(classes):
                tmp = np.zeros((p1.shape[0], len(classes)))
                tmp[:, :p1.shape[1]] = p1
                p1 = tmp
            if p2.shape[1] != len(classes):
                tmp = np.zeros((p2.shape[0], len(classes)))
                tmp[:, :p2.shape[1]] = p2
                p2 = tmp
            stacked = np.hstack([p1, p2])
            meta_probs = meta.predict_proba(stacked)[0]
            calibs = store.get("calibrators", {}) or {}
            meta_probs = apply_calibrators(calibs, classes, np.array([meta_probs]))[0]
            thresholds = store.get("metadata", {}).get("class_thresholds", {})
            candidates = []
            for i, c in enumerate(classes):
                thr = thresholds.get(c)
                if thr is not None and meta_probs[i] >= thr:
                    candidates.append((c, float(meta_probs[i])))
            if candidates:
                pred_class, pred_prob = max(candidates, key=lambda x: x[1])
            else:
                idx = int(np.argmax(meta_probs))
                pred_class = classes[idx]
                pred_prob = float(meta_probs[idx])
            planet_type = "exo" if pred_class.strip().upper() not in ("FP", "FA") else "not_exo"
            explanation_map = {
                "APC": "Ambiguous planetary candidate",
                "FA": "False alarm",
                "FP": "False positive",
                "KP": "Known planet",
                "PC": "Planetary candidate",
                "CP": "Confirmed planet"
            }
            return jsonify({
                "probability": float(pred_prob),
                "tfopwg_disp": pred_class,
                "tfopwg_disp_explanation": explanation_map.get(pred_class, ""),
                "planet_type": planet_type
            })
        else:
            return jsonify({"error": "unknown_satellite"}), 400
    except Exception as e:
        return jsonify({"error": "inference_error", "detail": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    print("Starting app; MODEL_DIR:", MODEL_DIR)
    print("TESS_MODEL_DIR:", TESS_MODEL_DIR)
    for s in STORE:
        print(s, "loaded files:", STORE[s]["files"])
        print(s, "features loaded:", STORE[s].get("features") is not None)
    # FIXED: Removed "git " prefix from host
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))