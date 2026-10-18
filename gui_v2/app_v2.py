"""
app_v2.py
---------
DentalAI Web Application (v2) - Full Interactive Clinical Decision Suite.
Runs on Port 7861 alongside the untouched legacy GUI on 7860.
Replicates App1's dynamic model discovery, lazy-cached inference, and exact GT/IoU matching,
while delivering the advanced Clinical GUI v2 (FDI Odontogram, 160x200 contextual tooth crop,
zoom/pan viewport, high-DPI canvas overlay, interactive box editing, and zero-copy filesystem path browsing).
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parents[1]                     # .../yolov11_training
YOLOV4 = BASE_DIR.parent / "yolov4"                               # .../yolov4
YOLO11 = YOLOV4 / "yolo11"
DARKNET = YOLOV4 / "darknet"

VAL_IMG_DIR = BASE_DIR / "datasets" / "aneesa" / "images" / "val"
DENTEX_IMG_DIR = YOLO11 / "external" / "dentex_yolo" / "images"
DENTEX_LBL_DIR = YOLO11 / "external" / "dentex_yolo" / "labels"
CORRECTIONS_DIR = BASE_DIR / "datasets" / "corrections_v2"
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = BASE_DIR / "datasets" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GUI_DIR = Path(__file__).resolve().parent

MAX_UPLOAD = 50 * 1024 * 1024
MAX_JSON = 10 * 1024 * 1024
CONF_FLOOR = 0.05

LOCK = threading.Lock()
_CACHE = {}
_EVAL_CACHE = {}
_CACHE_VERSION = {}
CUSTOM_DIR = None


def read_names():
    names_file = DARKNET / "yolo.names"
    if names_file.exists():
        return [n.strip() for n in names_file.read_text().splitlines() if n.strip()]
    # Default 32 FDI tooth classes
    return [
        "18","17","16","15","14","13","12","11","21","22","23","24","25","26","27","28",
        "48","47","46","45","44","43","42","41","31","32","33","34","35","36","37","38"
    ]


NAMES = read_names()


def discover_models():
    models = {}
    label_map = {
        "bench_rtdetr_l": "RT-DETR-L (1280px)",
        "bench_rtdetr_x": "RT-DETR-X (1280px)",
        "bench_yolo11l": "YOLO11l (1280px)",
        "bench_yolo11x": "YOLO11x (1280px)",
        "bench_yolo12m": "YOLO12m (1280px)",
        "bench_yolo26m": "YOLO26m (1280px)",
        "bench_yolov10m": "YOLOv10m (1280px)",
        "bench_yolov3u": "YOLOv3u (1280px)",
        "bench_yolov8m": "YOLOv8m (1280px)",
        "bench_yolov9c": "YOLOv9c (1280px)",
        "bench_yolov9e": "YOLOv9e (1280px)",
        "yolo11m_teeth": "YOLO11m (640px)",
        "yolo11m_teeth_1280": "YOLO11m (1280px)",
    }
    # Check yolo11/runs
    if (YOLO11 / "runs").exists():
        for w in sorted((YOLO11 / "runs").glob("*/weights/best.pt")):
            folder = w.parent.parent.name
            name = label_map.get(folder, folder)
            kind = "rtdetr" if "rtdetr" in folder.lower() else "ultra"
            models[name] = (kind, w)

    # Check yolov11_training/runs/train
    y11_train_runs = BASE_DIR / "runs" / "train"
    if y11_train_runs.exists():
        for w in sorted(y11_train_runs.glob("*/weights/best.pt")):
            folder = w.parent.parent.name
            name = label_map.get(folder, folder)
            if name not in models:
                kind = "rtdetr" if "rtdetr" in folder.lower() else "ultra"
                models[name] = (kind, w)

    v4 = DARKNET / "backup" / "yolov4-custom_last.weights"
    cfg = DARKNET / "cfg" / "yolov4-custom.cfg"
    if v4.exists() and cfg.exists():
        models["YOLOv4 (original, Darknet)"] = ("yolov4", v4)

    return models


def val_images():
    val_txt = DARKNET / "val.txt"
    if val_txt.exists():
        lines = [l.strip() for l in val_txt.read_text().splitlines() if l.strip()]
        return [str((DARKNET / l).resolve()) for l in lines]
    elif VAL_IMG_DIR.exists():
        exts = {".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"}
        return [str(f.resolve()) for f in sorted(VAL_IMG_DIR.iterdir()) if f.suffix.lower() in exts]
    return []


MODELS = discover_models()
VAL = val_images()


def default_imgsz(model_name):
    return 640 if "640" in model_name else 1280


def parse_args_yaml(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"^(imgsz|epochs|batch|model):\s*(.+)$", line)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def model_info(name):
    if name not in MODELS:
        return {"name": name, "kind": "unknown", "weights": "", "size_mb": 0, "modified": "", "imgsz": 1280}
    kind, path = MODELS[name]
    st = path.stat()
    info = {
        "name": name,
        "kind": kind,
        "weights": str(path.relative_to(BASE_DIR)) if BASE_DIR in path.parents else str(path),
        "size_mb": round(st.st_size / 1e6, 1),
        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
        "imgsz": default_imgsz(name) if kind in ("ultra", "rtdetr") else 416,
        "metrics": None,
        "in_training": False,
        "epochs_done": None,
        "epochs_total": None,
    }
    if kind in ("ultra", "rtdetr"):
        run = path.parent.parent
        args = parse_args_yaml(run / "args.yaml")
        if "imgsz" in args:
            try: info["imgsz"] = int(args["imgsz"])
            except: pass
        if "epochs" in args:
            try: info["epochs_total"] = int(args["epochs"])
            except: pass
        csv_path = run / "results.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
            if rows:
                key = "metrics/mAP50-95(B)"
                best = max(rows, key=lambda r: float(r[key])) if key in rows[0] else rows[-1]
                info["epochs_done"] = len(rows)
                try:
                    info["metrics"] = {
                        "best_epoch": int(float(best.get("epoch", len(rows)))),
                        "precision": round(float(best.get("metrics/precision(B)", 0)), 3),
                        "recall": round(float(best.get("metrics/recall(B)", 0)), 3),
                        "mAP50": round(float(best.get("metrics/mAP50(B)", 0)), 3),
                        "mAP50-95": round(float(best.get(key, 0)), 3),
                        "note": "validation set",
                    }
                except Exception:
                    pass
            info["in_training"] = (time.time() - csv_path.stat().st_mtime) < 180
    return info


def load(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name}")
    kind, path = MODELS[model_name]
    mtime = path.stat().st_mtime
    cached = _CACHE.get(model_name)
    if cached and cached[0] == mtime:
        return cached[1]
    if kind == "rtdetr":
        from ultralytics import RTDETR
        obj = RTDETR(str(path))
    elif kind == "ultra":
        from ultralytics import YOLO
        obj = YOLO(str(path))
    else:
        net = cv2.dnn.readNet(str(path), str(DARKNET / "cfg" / "yolov4-custom.cfg"))
        obj = cv2.dnn_DetectionModel(net)
        obj.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True, crop=False)
    _CACHE[model_name] = (mtime, obj)
    return obj


def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def predict(img, model_name, conf=0.25, iou=0.60, imgsz=1280, clahe=False):
    kind, _ = MODELS[model_name]
    if clahe:
        img = apply_clahe(img)
    dets = []
    t0 = time.time()
    with LOCK:
        model = load(model_name)
        if kind in ("ultra", "rtdetr"):
            res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                dets.append({
                    "cls": res.names[int(c)],
                    "conf": round(float(s), 4),
                    "box": [round(float(v), 1) for v in b]
                })
        else:
            classes, scores, boxes = model.detect(img, conf, iou)
            for c, s, (x, y, w, h) in zip(np.array(classes).reshape(-1), np.array(scores).reshape(-1), boxes):
                if int(c) < len(NAMES):
                    dets.append({
                        "cls": NAMES[int(c)],
                        "conf": round(float(s), 4),
                        "box": [float(x), float(y), float(x + w), float(y + h)]
                    })
    ms = (time.time() - t0) * 1000
    return dets, ms


def load_gt(path_str, w, h):
    p = Path(path_str)
    # Search candidates for corresponding label text file
    candidates = [
        p.with_suffix(".txt"),
        p.parent.parent / "labels" / p.parent.name / f"{p.stem}.txt",
        p.parent.parent / "labels" / f"{p.stem}.txt",
        DARKNET / "data" / f"{p.stem}.txt",
        DENTEX_LBL_DIR / f"{p.stem}.txt",
    ]
    lab = next((c for c in candidates if c.exists()), None)
    out = []
    if lab and lab.exists():
        for line in lab.read_text().splitlines():
            t = line.split()
            if len(t) != 5:
                continue
            c = int(float(t[0]))
            x, y, bw, bh = (float(v) for v in t[1:])
            cls_name = NAMES[c] if c < len(NAMES) else str(c)
            out.append({
                "cls": cls_name,
                "box": [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
            })
    return out


def iou_xyxy(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def match_eval(dets, gt, thr=0.5):
    used = set()
    tp = 0
    for d in sorted(dets, key=lambda d: -d["conf"]):
        best, bj = thr, -1
        for j, g in enumerate(gt):
            if j in used or g["cls"] != d["cls"]:
                continue
            v = iou_xyxy(d["box"], g["box"])
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
            tp += 1
    fp = len(dets) - tp
    fn = len(gt) - tp
    prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def val_path(self, name):
        # Look up in VAL list first
        for p in VAL:
            if Path(p).name == name or Path(p).stem == Path(name).stem:
                return p
        # Check standard dirs
        for d in [VAL_IMG_DIR, DENTEX_IMG_DIR, UPLOAD_DIR, CUSTOM_DIR]:
            if d and (d / name).exists():
                return str((d / name).resolve())
        return None

    def resolve_image_path(self, q):
        path_param = q.get("path", [""])[0]
        name_param = q.get("name", [""])[0] or q.get("img", [""])[0]
        src_param = q.get("source", ["val"])[0]

        if path_param:
            p = Path(path_param).resolve()
            if p.exists() and p.is_file():
                return str(p)

        if name_param:
            p_name = Path(name_param).resolve()
            if p_name.exists() and p_name.is_file():
                return str(p_name)

        if src_param == "dentex" and DENTEX_IMG_DIR.exists():
            cand = DENTEX_IMG_DIR / name_param
            if cand.exists(): return str(cand.resolve())

        if src_param == "custom" and CUSTOM_DIR and CUSTOM_DIR.exists():
            cand = CUSTOM_DIR / name_param
            if cand.exists(): return str(cand.resolve())

        if src_param == "upload" and UPLOAD_DIR.exists():
            cand = UPLOAD_DIR / name_param
            if cand.exists(): return str(cand.resolve())

        vp = self.val_path(name_param)
        if vp: return vp

        return None

    def parse_det_params(self, q):
        model_name = q.get("model", ["RT-DETR-L (1280px)"])[0]
        if model_name not in MODELS:
            # Pick first available or best match
            for m in MODELS:
                if model_name.split()[0].lower() in m.lower():
                    model_name = m
                    break
            else:
                model_name = list(MODELS.keys())[0]

        conf = min(max(float(q.get("conf", [0.25])[0]), 0.01), 0.99)
        iou = min(max(float(q.get("iou", [0.60])[0]), 0.05), 0.95)
        imgsz = int(q.get("imgsz", ["0"])[0])
        if imgsz <= 0:
            imgsz = default_imgsz(model_name)
        imgsz = min(max(imgsz, 320), 2560)
        clahe = q.get("clahe", ["0"])[0] in ("1", "true")
        return model_name, conf, iou, imgsz, clahe

    def do_GET(self):
        global CUSTOM_DIR
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                html_path = GUI_DIR / "index.html"
                self.send_bytes(html_path.read_bytes(), "text/html; charset=utf-8")

            elif u.path in ("/app_icon.png", "/favicon.ico"):
                icon_path = GUI_DIR / "app_icon.png"
                if icon_path.exists():
                    self.send_bytes(icon_path.read_bytes(), "image/png")
                else:
                    self.send_json({"error": "icon not found"}, 404)

            elif u.path == "/api/info":
                models_info = [model_info(m) for m in MODELS]
                val_names = [Path(p).name for p in VAL]
                self.send_json({
                    "models": models_info,
                    "val": val_names,
                    "names": NAMES,
                    "auth": False
                })

            elif u.path in ("/api/original", "/image"):
                img_path = self.resolve_image_path(q)
                if not img_path or not Path(img_path).exists():
                    return self.send_json({"error": "image not found"}, 404)
                p = Path(img_path)
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                self.send_bytes(p.read_bytes(), mime)

            elif u.path == "/api/gt":
                img_path = self.resolve_image_path(q)
                if not img_path or not Path(img_path).exists():
                    return self.send_json({"error": "image not found"}, 404)
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                gt = load_gt(img_path, w, h)
                self.send_json({"gt": gt, "width": w, "height": h})

            elif u.path == "/images_list":
                src = q.get("source", ["val"])[0]
                if src == "dentex":
                    folder = DENTEX_IMG_DIR
                elif src == "custom" and CUSTOM_DIR and CUSTOM_DIR.exists():
                    folder = CUSTOM_DIR
                elif src == "upload":
                    folder = UPLOAD_DIR
                else:
                    folder = VAL_IMG_DIR if VAL_IMG_DIR.exists() else (DARKNET / "data")

                exts = {".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                files = []
                if folder and folder.exists():
                    for f in sorted(folder.iterdir()):
                        if f.is_file() and f.suffix.lower() in exts:
                            files.append(f.name)
                elif src == "val":
                    files = [Path(p).name for p in VAL]

                self.send_json({"images": files[:1000], "source": src, "folder": str(folder) if folder else ""})

            elif u.path == "/browse_fs":
                path_str = q.get("path", [""])[0]
                p = Path(path_str).resolve() if path_str else BASE_DIR
                if not p.exists() or not p.is_dir():
                    p = BASE_DIR

                subdirs = []
                img_files = []
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                try:
                    for entry in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                        if entry.name.startswith("."):
                            continue
                        if entry.is_dir():
                            try:
                                n_imgs = sum(1 for f in entry.glob("*") if f.suffix.lower() in exts)
                            except Exception:
                                n_imgs = 0
                            subdirs.append({"name": entry.name, "path": str(entry.resolve()), "num_images": n_imgs})
                        elif entry.is_file() and entry.suffix.lower() in exts:
                            img_files.append({"name": entry.name, "path": str(entry.resolve()), "size": entry.stat().st_size})
                except Exception as e:
                    return self.send_json({"error": str(e)}, 500)

                self.send_json({
                    "status": "success",
                    "current_path": str(p),
                    "parent_path": str(p.parent) if p.parent != p else None,
                    "directories": subdirs,
                    "images": img_files,
                    "total_images": len(img_files)
                })

            elif u.path == "/load_folder_path":
                path_str = q.get("path", [""])[0]
                p = Path(path_str).resolve()
                if p.exists() and p.is_dir():
                    CUSTOM_DIR = p
                    exts = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
                    files = [f.name for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts]
                    self.send_json({"status": "success", "images": files, "count": len(files), "path": str(p)})
                else:
                    self.send_json({"error": f"Directory not found: {path_str}"}, 400)

            elif u.path == "/predict":
                # GET-based prediction for web compatibility
                img_path = self.resolve_image_path(q)
                if not img_path or not Path(img_path).exists():
                    return self.send_json({"error": "image not found"}, 404)

                model_name, conf, iou, imgsz, clahe = self.parse_det_params(q)
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                dets, ms = predict(img, model_name, conf, iou, imgsz, clahe)

                gt = load_gt(img_path, w, h)
                metrics = match_eval(dets, gt, thr=0.5)

                # Format detections for both App1 (box: [x1,y1,x2,y2]) and App2 (x1,y1,x2,y2 keys)
                boxes_out = []
                for d in dets:
                    boxes_out.append({
                        "cls": d["cls"],
                        "conf": d["conf"],
                        "box": d["box"],
                        "x1": d["box"][0],
                        "y1": d["box"][1],
                        "x2": d["box"][2],
                        "y2": d["box"][3],
                    })

                gt_boxes_out = []
                for g in gt:
                    gt_boxes_out.append({
                        "cls": g["cls"],
                        "box": g["box"],
                        "x1": g["box"][0],
                        "y1": g["box"][1],
                        "x2": g["box"][2],
                        "y2": g["box"][3],
                    })

                self.send_json({
                    "detections": dets,
                    "boxes": boxes_out,
                    "gt_boxes": gt_boxes_out,
                    "metrics": metrics,
                    "latency": round(ms, 1),
                    "ms": round(ms, 1),
                    "image": Path(img_path).name,
                    "width": w,
                    "height": h,
                    "count": len(boxes_out)
                })

            else:
                self.send_json({"error": f"Path not found: {u.path}"}, 404)

        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/api/predict_upload", "/api/predict_val", "/api/predict_path", "/predict"):
                model_name, conf, iou, imgsz, clahe = self.parse_det_params(q)

                if u.path == "/api/predict_upload":
                    n = int(self.headers.get("Content-Length", 0))
                    if n <= 0 or n > MAX_UPLOAD:
                        return self.send_json({"error": "bad upload size"}, 400)
                    raw_bytes = self.rfile.read(n)
                    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        return self.send_json({"error": "not a readable image"}, 400)
                    img_path = None
                else:
                    img_path = self.resolve_image_path(q)
                    if not img_path or not Path(img_path).exists():
                        return self.send_json({"error": "image not found"}, 404)
                    img = cv2.imread(img_path)

                h, w = img.shape[:2]
                dets, ms = predict(img, model_name, conf, iou, imgsz, clahe)

                gt = load_gt(img_path, w, h) if img_path else []
                metrics = match_eval(dets, gt, thr=0.5)

                boxes_out = []
                for d in dets:
                    boxes_out.append({
                        "cls": d["cls"],
                        "conf": d["conf"],
                        "box": d["box"],
                        "x1": d["box"][0],
                        "y1": d["box"][1],
                        "x2": d["box"][2],
                        "y2": d["box"][3],
                    })

                gt_boxes_out = []
                for g in gt:
                    gt_boxes_out.append({
                        "cls": g["cls"],
                        "box": g["box"],
                        "x1": g["box"][0],
                        "y1": g["box"][1],
                        "x2": g["box"][2],
                        "y2": g["box"][3],
                    })

                self.send_json({
                    "detections": dets,
                    "boxes": boxes_out,
                    "gt_boxes": gt_boxes_out,
                    "metrics": metrics,
                    "latency": round(ms, 1),
                    "ms": round(ms, 1),
                    "image": Path(img_path).name if img_path else "upload.jpg",
                    "width": w,
                    "height": h,
                    "count": len(boxes_out)
                })

            elif u.path in ("/api/save_labels", "/save_corrections"):
                n = int(self.headers.get("Content-Length", 0))
                if n <= 0 or n > MAX_JSON:
                    return self.send_json({"error": "bad body size"}, 400)
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                w = float(body.get("width", 1.0))
                h = float(body.get("height", 1.0))
                img_name = body.get("image", q.get("name", ["unknown.jpg"])[0])
                boxes = body.get("boxes", [])

                lines = []
                for b in boxes:
                    cls_val = b.get("cls", "")
                    if cls_val in NAMES:
                        cls_idx = NAMES.index(cls_val)
                    else:
                        try: cls_idx = int(cls_val)
                        except: cls_idx = 0

                    if "box" in b:
                        x1, y1, x2, y2 = (float(v) for v in b["box"])
                    else:
                        x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])

                    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
                    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                out_file = CORRECTIONS_DIR / f"{Path(img_name).stem}.txt"
                out_file.write_text("\n".join(lines) + "\n")
                self.send_json({"saved": str(out_file), "boxes": len(lines), "count": len(lines)})

            elif u.path == "/upload_opg":
                n = int(self.headers.get("Content-Length", 0))
                if n <= 0 or n > MAX_UPLOAD:
                    return self.send_json({"error": "invalid upload size"}, 400)
                data = self.rfile.read(n)
                filename = f"upload_{int(time.time())}.jpg"
                out_p = UPLOAD_DIR / filename
                out_p.write_bytes(data)
                self.send_json({"status": "success", "filename": filename, "source": "upload"})

            else:
                self.send_json({"error": "not found"}, 404)

        except Exception as e:
            self.send_json({"error": str(e)}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    print("=" * 65)
    print("  🚀 DentalAI Suite v2 (Fully Functional Clinical GUI)")
    print(f"  Discovered {len(MODELS)} models: {list(MODELS.keys())[:5]} ...")
    print(f"  Validation images: {len(VAL)}")
    print(f"  Listening on: http://{args.host}:{args.port}")
    print("=" * 65)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
