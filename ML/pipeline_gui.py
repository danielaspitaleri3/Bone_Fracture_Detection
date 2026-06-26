#!/usr/bin/env python3
# Build marker: PIPELINE_ROI_CLASSIFICATE_PROTECTED_BUILD_2026_06_23
# Build marker: PIPELINE_FPOVERLAPIGNORE_BUILD_2026_06_19
# Build marker: GUI_PR_AP_RESULT_FOOTER_BUILD_2026_06_20
# Build marker: FN_TOOLTIP_SYNCFIX_BUILD_2026_06_19
# Build marker: FIXED_MLGE50_IOUGE50_IGNOREGE20_BUILD_2026_06_19
# Build marker: IOU_DISPLAY_FOR_LOW_IOU_ROIS_BUILD_2026_06_19
"""
Minimal clinical diagnostic GUI.
Build marker: CONFIG_SELECTOR_INLINE_TPGE_METRICSFIX_PER_TECH_BUILD_2026_06_19
GT_SYNCFIX_BUILD_2026_06_19
"""

from __future__ import annotations

# Build marker: CONFIG_SELECTOR_INLINE_TPGE_METRICSFIX_PER_TECH_BUILD_2026_06_19
# GT_SYNCFIX_BUILD_2026_06_19
import sys, subprocess, pickle, threading, time, os, math, hashlib, queue, re, json
import importlib.util
from datetime import datetime
from pathlib import Path
from io import BytesIO

# ── Radice del progetto (rilevata automaticamente) ──────────────────────────
# Questo file si trova in <progetto>/ML, quindi la radice del progetto e' due
# livelli sopra. Cosi' funziona su qualsiasi computer e sistema operativo,
# anche spostando la cartella, senza dover modificare i percorsi a mano.
WINDOWS_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_IMAGE_DIR = WINDOWS_PROJECT_ROOT / "img_fracture"
PROJECT_ML_DIR = WINDOWS_PROJECT_ROOT / "ML"
TRAINING_RESULTS_DIR = PROJECT_ML_DIR / "risultati_addestramento"
TRAINING_MODEL_PATH = TRAINING_RESULTS_DIR / "modello" / "fracture_fp_model.pkl"
TRAINING_SPLIT_IMAGES_DIR = TRAINING_RESULTS_DIR / "split_images"
EVALUATION_RESULTS_DIR = PROJECT_ML_DIR / "risultati_valutazione_postprocessing"
EVALUATION_CSV_DIR = EVALUATION_RESULTS_DIR / "csv"
EVALUATION_CURVES_DIR = EVALUATION_RESULTS_DIR / "curve"
PIPELINE_RESULTS_DIR = PROJECT_ML_DIR / "risultati_pipeline"
PIPELINE_RUNTIME_DIR = PIPELINE_RESULTS_DIR / "runtime"
PIPELINE_ANNOTATED_DIR = PIPELINE_RESULTS_DIR / "immagini_annotate"
PIPELINE_LOG_DIR = PIPELINE_RESULTS_DIR / "log"
# Da questa versione la GUI NON esegue piu' il programma C++ e NON usa cartelle IPA.
# Tutti i file letti/scritti durante l'esecuzione devono stare sotto ML/risultati_pipeline.
CSV_FEATURE_DIR = PIPELINE_RESULTS_DIR / "CSV_feature"
ROI_CLASSIFICATE_DIR = PIPELINE_RESULTS_DIR / "ROI_classificate"
PIPELINE_TEMP_DIR = PIPELINE_RESULTS_DIR / "temp"
# Lettura consentita dai CSV ufficiali IPA; nessuna scrittura/modifica in IPA.
IPA_CSV_FEATURE_DIR = WINDOWS_PROJECT_ROOT / "IPA" / "risultati_rilevamento_fratture" / "CSV_feature"

def _ensure_pipeline_output_dirs() -> None:
    for folder in [PIPELINE_RESULTS_DIR, PIPELINE_RUNTIME_DIR, PIPELINE_ANNOTATED_DIR, PIPELINE_LOG_DIR, CSV_FEATURE_DIR, ROI_CLASSIFICATE_DIR, PIPELINE_TEMP_DIR]:
        folder.mkdir(parents=True, exist_ok=True)



def _timestamped_sibling(path: Path, reason: str = "locked") -> Path:
    """Crea un nome alternativo quando Windows blocca un file aperto."""
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_reason = re.sub(r"[^A-Za-z0-9_]+", "_", str(reason or "locked")).strip("_") or "locked"
    return path.with_name(f"{path.stem}_{safe_reason}_{stamp}{path.suffix}")


def _safe_write_text_file(path, text, encoding="utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding=encoding)
        return path
    except PermissionError:
        alt = _timestamped_sibling(path, "locked")
        print(f"[WARN] File bloccato o aperto: {path}")
        print(f"[WARN] Salvo una nuova copia in: {alt}")
        alt.write_text(text, encoding=encoding)
        return alt


def _safe_save_image(image, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.save(path)
        return path
    except PermissionError:
        alt = _timestamped_sibling(path, "locked")
        print(f"[WARN] Immagine output bloccata o aperta: {path}")
        print(f"[WARN] Salvo una nuova copia in: {alt}")
        image.save(alt)
        return alt


def _filter_feature_df_for_image(df: pd.DataFrame, img_path: Path) -> pd.DataFrame:
    """Quando si usa il CSV combinato, tiene solo le ROI dell'immagine corrente."""
    if df is None or len(df) == 0 or "image" not in df.columns or img_path is None:
        return df
    image_name = Path(img_path).name.lower()
    image_stem = Path(img_path).stem.lower()
    col = df["image"].astype(str).str.replace("\\\\", "/", regex=False)
    names = col.map(lambda v: Path(v).name.lower())
    stems = col.map(lambda v: Path(v).stem.lower())
    mask = (names == image_name) | (stems == image_stem)
    if mask.any():
        filtered = df.loc[mask].copy()
        print(f"[INFO] CSV combinato filtrato per {Path(img_path).name}: {len(filtered)} ROI su {len(df)}.")
        return filtered
    print(f"[WARN] Nel CSV non trovo righe per {Path(img_path).name}; uso CSV completo come fallback.")
    return df

# ── Auto-install dependencies ──────────────────────────────────────────────────
for _pkg, _imp in [("pillow","PIL"), ("numpy","numpy"), ("pandas","pandas"), ("customtkinter","customtkinter")]:
    try: __import__(_imp)
    except ImportError:
        print(f"[*] Installazione {_pkg}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageTk
import customtkinter as ctk

_MPL_TK = None


def _ensure_matplotlib_tk():
    """Carica Matplotlib su backend Tk solo quando serve la finestra metriche."""
    global _MPL_TK
    if _MPL_TK is not None:
        return _MPL_TK
    try:
        import matplotlib
    except ImportError:
        print("[*] Installazione matplotlib...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib
    matplotlib.use("TkAgg", force=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _MPL_TK = (Figure, FigureCanvasTkAgg)
    return _MPL_TK

# ── Theme configuration ────────────────────────────────────────────────────────
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── Design Tokens: clinical review workstation ───────────────────────────────
# Neutrals and restrained teal/blue accents keep radiographs visually dominant.
BG_MAIN       = "#F5F7FA"
BG_SIDEBAR    = "#0A1729"
BG_CARD       = "#FFFFFF"
BG_VIEWER     = "#081321"
BG_VIEWER_2   = "#0C1728"
GLASS_PANEL   = "#FFFFFF"
GLASS_SOFT    = "#EEF3F7"
GLASS_BORDER  = "#D9E2EA"
GLASS_HILITE  = "#FFFFFF"

BORDER_CLR    = "#1E3049"
BORDER_FOCUS  = "#2B9AA6"

TEXT_MAIN     = "#102235"
TEXT_SUB      = "#425466"
TEXT_MUTED    = "#6D7E90"
TEXT_SIDEBAR  = "#E4EDF6"

# Clinical accent colours: teal = action; outcomes detection-level:
#   verde = TP (vera frattura trovata) · arancione = FP (falso positivo)
#   rosso = FN (frattura mancata) · viola = ground truth (verita')
ACCENT_BLUE   = "#0C6B83"
ACCENT_BLUE_L = "#1697A6"
ACCENT_CYAN   = "#0891B2"
ACCENT_GREEN  = "#059669"   # TP
ACCENT_ORANGE = "#D97706"   # FP
ACCENT_RED    = "#DC2626"   # FN
ACCENT_INDIGO = "#6D28D9"
ACCENT_PURPLE = "#A855F7"   # ground truth (viola acceso, ben visibile sulle radiografie)

VIEWER_ACCENT = "#20AFC3"
VIEWER_GRID   = "#14243B"
VIEWER_PANEL  = "#102036"
CANDIDATE_CLR = "#22C3D6"   # ROI grezze prodotte dall'image processing

def interpolate_color(color_hex1, color_hex2, factor):
    c1 = [int(color_hex1[i:i+2], 16) for i in (1, 3, 5)]
    c2 = [int(color_hex2[i:i+2], 16) for i in (1, 3, 5)]
    interp = [int(c1[j] + (c2[j] - c1[j]) * factor) for j in range(3)]
    return f"#{interp[0]:02X}{interp[1]:02X}{interp[2]:02X}"

LABEL_DIR_ENV = "IPA_GT_LABELS_DIR"

# ── Series browser & saved performance plots ──────────────────────────────────
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_CURVE_RELATIVE_DIR = EVALUATION_CURVES_DIR
GLOBAL_METRICS_FILE = _CURVE_RELATIVE_DIR / "global_metrics_nms_only.txt"
SUMMARY_COMPARISON_FILE = _CURVE_RELATIVE_DIR / "summary_nms_only.txt"
# Dati salvati da evaluate_test_postprocess.py. La GUI non mostra piu' PNG statici:
# ridisegna curve Matplotlib reali dai CSV, cosi' il grafico resta interattivo.
CURVE_DATA_CANDIDATES = {
    "FROC": [
        _CURVE_RELATIVE_DIR / "curve_points_nms_only.csv",
    ],
    # Confronto delle 3 FROC ottenute sulla validation (una per soglia NMS).
    "FROC-validation": [
        _CURVE_RELATIVE_DIR / "froc_validation_sweep_points.csv",
    ],
    "Precision-Recall": [
        _CURVE_RELATIVE_DIR / "ap_precision_recall_points_nms_only.csv",
        _CURVE_RELATIVE_DIR / "curve_points_nms_only.csv",
    ],
}
FROC_DB_FLOOR = -40.0

def _build_label_search_paths(img_path: Path):
    candidates = []
    env_path = os.getenv(LABEL_DIR_ENV)
    if env_path:
        candidates.append(Path(env_path))

    # Percorsi reali del progetto.
    candidates.extend([
        PROJECT_IMAGE_DIR,
        PROJECT_IMAGE_DIR / "labels",
        PROJECT_IMAGE_DIR / "label",
    ])

    base = Path(__file__).parent
    candidates.extend([
        base / "images" / "label",
        base / "images" / "labels",
        base / "label",
        base / "labels",
        img_path.parent / "labels",
        img_path.parent / "label",
        img_path.parent.parent / "labels",
        img_path.parent.parent / "label",
    ])

    return [p for p in candidates if p.exists() and p.is_dir()]


def _load_yolo_label_file(label_file: Path, image_size):
    """Carica solo le ground truth YOLO della classe frattura.

    Formato atteso riga YOLO:
        class_id center_x center_y width height

    Le coordinate devono essere normalizzate tra 0 e 1.
    Vengono ignorate tutte le classi diverse da GT_FRACTURE_CLASS_ID.
    """
    boxes = []
    img_w, img_h = image_size

    try:
        with label_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                except Exception:
                    continue

                # IMPORTANTISSIMO: usa solo la classe 3, cioè frattura.
                if cls_id != GT_FRACTURE_CLASS_ID:
                    continue

                abs_w = max(1.0, w * img_w)
                abs_h = max(1.0, h * img_h)
                abs_x = max(0.0, (cx - w / 2.0) * img_w)
                abs_y = max(0.0, (cy - h / 2.0) * img_h)

                boxes.append({
                    "roi_id": f"GT{len(boxes) + 1}",
                    "method": "GT",
                    "x": abs_x,
                    "y": abs_y,
                    "width": abs_w,
                    "height": abs_h,
                    "score": 0.0,
                    "ml_score": 0.0,
                    "ml_pred": 0,
                    "label": 1,
                    "class_id": cls_id,
                    "_source": str(label_file),
                })
    except Exception:
        pass

    return boxes


def _load_label_boxes_for_image(img_path: Path):
    if not img_path or not img_path.exists():
        return []

    search_dirs = _build_label_search_paths(img_path)
    if not search_dirs:
        return []

    label_file = None
    for root in search_dirs:
        candidate = root / f"{img_path.stem}.txt"
        if candidate.exists():
            label_file = candidate
            break

    if not label_file:
        return []

    image_size = load_16bit_image_as_rgb(img_path).size
    return _load_yolo_label_file(label_file, image_size)


# ── Ground Truth YOLO ───────────────────────────────────────────────
GT_FRACTURE_CLASS_ID = 3      # usa solo le label YOLO classe 3 = frattura

# ── Post-processing ML: solo NMS ────────────────────────────────────────────
NMS_TECHNIQUE = "nms"
POSTPROCESS_TECHNIQUES = [NMS_TECHNIQUE]
TECHNIQUE_LABELS = {
    NMS_TECHNIQUE: "NMS",
}
DEFAULT_POSTPROCESS_TECHNIQUE = NMS_TECHNIQUE

NMS_IOU_THRESHOLD = 0.40        # NMS: scarta come duplicato solo box con IoU >= 0.40
TP_IOU_THRESHOLD = 0.50         # REGOLA FISSA: una detection e' TP se best IoU >= 0.50
TP_IOU_RULE_TEXT = f"IoU >= {TP_IOU_THRESHOLD:.2f}"
IGNORE_IOU_THRESHOLD = 0.20     # IGNORE: duplicato solo se IoU GT >= 0.20 su una GT che ha GIA' un TP
ML_MIN_SCORE = 0.50             # soglia ML operativa: ROI considerata se ML >= 0.50


def _canonical_technique_name(name):
    """La GUI lavora solo in modalita' NMS."""
    return NMS_TECHNIQUE

def _load_best_postprocess_params():
    """Carica, se presente, la migliore combinazione trovata dallo script evaluate."""
    candidates = [
        EVALUATION_CURVES_DIR / "best_postprocess_params.json",
        Path.cwd() / "ML" / "risultati_valutazione_postprocessing" / "curve" / "best_postprocess_params.json",
        Path(__file__).parent / "ml" / "postprocess_curves" / "best_postprocess_params.json",  # fallback vecchio
        Path.cwd() / "ml" / "postprocess_curves" / "best_postprocess_params.json",              # fallback vecchio
    ]
    for path in candidates:
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
        except Exception:
            pass
    return None


_best_postprocess_params = _load_best_postprocess_params()
if _best_postprocess_params:
    try:
        ML_MIN_SCORE = 0.50
        json_nms_iou = _best_postprocess_params.get("nms_iou")
        if json_nms_iou is not None:
            try:
                json_nms_iou = float(json_nms_iou)
                if abs(json_nms_iou - NMS_IOU_THRESHOLD) > 1e-12:
                    print(
                        f"[INFO] nms_iou={json_nms_iou:.3f} nel JSON ignorato: "
                        f"la GUI usa sempre NMS con IoU >= {NMS_IOU_THRESHOLD:.2f}."
                    )
            except Exception:
                pass
        json_tp_iou = _best_postprocess_params.get("tp_iou")
        if json_tp_iou is not None:
            try:
                json_tp_iou = float(json_tp_iou)
                if abs(json_tp_iou - TP_IOU_THRESHOLD) > 1e-12:
                    print(
                        f"[INFO] tp_iou={json_tp_iou:.3f} nel JSON ignorato: "
                        f"la GUI usa sempre TP con {TP_IOU_RULE_TEXT}."
                    )
            except Exception:
                pass
        IGNORE_IOU_THRESHOLD = 0.20
        DEFAULT_POSTPROCESS_TECHNIQUE = NMS_TECHNIQUE
        print(
            "[INFO] Parametri NMS caricati: "
            f"ML>={ML_MIN_SCORE:.3f}, NMS IoU>={NMS_IOU_THRESHOLD:.3f}, "
            f"TP>={TP_IOU_THRESHOLD:.2f}"
        )
    except Exception as exc:
        print(f"[WARN] Parametri NMS non validi: {exc}")



def _refresh_best_postprocess_params(verbose=False):
    """Rilegge best_postprocess_params.json e aggiorna i parametri NMS globali."""
    global ML_MIN_SCORE, NMS_IOU_THRESHOLD, IGNORE_IOU_THRESHOLD
    global DEFAULT_POSTPROCESS_TECHNIQUE, _best_postprocess_params

    data = _load_best_postprocess_params()
    if not data:
        return None
    try:
        ML_MIN_SCORE = 0.50
        json_nms_iou = data.get("nms_iou")
        if json_nms_iou is not None:
            try:
                json_nms_iou = float(json_nms_iou)
                if abs(json_nms_iou - NMS_IOU_THRESHOLD) > 1e-12 and verbose:
                    print(
                        f"[INFO] nms_iou={json_nms_iou:.3f} nel JSON ignorato: "
                        f"la GUI usa sempre NMS con IoU >= {NMS_IOU_THRESHOLD:.2f}."
                    )
            except Exception:
                pass
        IGNORE_IOU_THRESHOLD = 0.20
        DEFAULT_POSTPROCESS_TECHNIQUE = NMS_TECHNIQUE
        _best_postprocess_params = data
        if verbose:
            print(
                "[INFO] Best NMS test set riletto: "
                f"ML>={ML_MIN_SCORE:.3f}, NMS IoU>={NMS_IOU_THRESHOLD:.3f}, "
                f"TP>={TP_IOU_THRESHOLD:.2f}"
            )
        return data
    except Exception as exc:
        if verbose:
            print(f"[WARN] Parametri best_postprocess_params.json non validi: {exc}")
        return None


def _best_postprocess_params_summary():
    """Testo compatto mostrato nella finestra Metriche."""
    data = _load_best_postprocess_params() or _best_postprocess_params or {}
    if not data:
        return "Best test set NMS non ancora calcolato: esegui evaluate_test_postprocess_class3_nms_only.py."
    try:
        precision = float(data.get("precision", 0.0)) * 100.0
        recall = float(data.get("recall", 0.0)) * 100.0
        f1 = float(data.get("F1", 0.0))
        f2 = float(data.get("F2", 0.0))
        ap = float(data.get("average_precision_detection_file", data.get("area_pr_threshold_sweep", 0.0)))
    except Exception:
        precision = recall = f1 = f2 = ap = 0.0
    metric = str(data.get("selection_metric", "F1"))
    win = data.get("nms_selected")
    win_txt = f"NMS operativa={float(NMS_IOU_THRESHOLD):.2f} · "
    return (
        f"{win_txt}TEST SET NMS: ML>={ML_MIN_SCORE:.3f} · "
        f"NMS IoU>={float(NMS_IOU_THRESHOLD):.2f} · "
        f"TP IoU>={TP_IOU_THRESHOLD:.2f} · P={precision:.2f}% · R={recall:.2f}% · "
        f"F1={f1:.4f} · F2={f2:.4f} · AP={ap:.4f} · criterio={metric}"
    )


def _format_percent(value, decimals=2):
    try:
        return f"{float(value) * 100.0:.{decimals}f}%"
    except Exception:
        return "—"


def _format_float(value, decimals=4):
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


def _format_int(value):
    try:
        return str(int(float(value)))
    except Exception:
        return "—"


def _load_key_value_file(path):
    values = {}
    try:
        path = Path(path)
        if not path.exists():
            return values
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
    except Exception:
        return {}
    return values


def _ap_values_by_technique_from_files():
    values = {}
    for source in [GLOBAL_METRICS_FILE, SUMMARY_COMPARISON_FILE]:
        values.update(_load_key_value_file(source))

    data = _load_best_postprocess_params() or _best_postprocess_params or {}
    if isinstance(data, dict):
        ap_map = data.get("ap_by_technique")
        if isinstance(ap_map, dict):
            for tech, val in ap_map.items():
                if _canonical_technique_name(tech) == NMS_TECHNIQUE:
                    values[f"AVERAGE_PRECISION_DETECTION_FILE_{NMS_TECHNIQUE}"] = str(val)
        elif data.get("average_precision_detection_file") is not None:
            values[f"AVERAGE_PRECISION_DETECTION_FILE_{NMS_TECHNIQUE}"] = str(data.get("average_precision_detection_file"))

    out = {}
    key = f"AVERAGE_PRECISION_DETECTION_FILE_{NMS_TECHNIQUE}"
    if key in values:
        try:
            out[NMS_TECHNIQUE] = float(values[key])
        except Exception:
            pass
    return out


def _curve_threshold_note():
    """Nota mostrata sotto le curve: chiarisce che la soglia ML varia lungo tutta
    la curva (sweep), mentre ML>=0.50 e' solo il punto operativo fisso della GUI."""
    return (f"Soglia ML variabile lungo la curva · ML>={ML_MIN_SCORE:.2f} "
            f"e' solo il punto operativo della GUI")


def _precision_recall_ap_footer_text():
    ap_values = _ap_values_by_technique_from_files()
    if not ap_values:
        return _best_postprocess_params_summary()
    ap = ap_values.get(NMS_TECHNIQUE, 0.0)
    return f"Precision–Recall NMS: AP={ap:.4f} · AP = area sotto la curva PR · {_curve_threshold_note()}"


def _resolve_curve_data_path(curve_name):
    for candidate in CURVE_DATA_CANDIDATES.get(curve_name, []):
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except Exception:
            continue
    return None


def _curve_float(value, default=None):
    try:
        val = float(value)
        if np.isfinite(val):
            return val
    except Exception:
        pass
    return default


def _clamp01(value):
    value = _curve_float(value, 0.0)
    return max(0.0, min(1.0, value))


def _ratio_to_db(value, floor_db=FROC_DB_FLOOR):
    value = _clamp01(value)
    if value <= 0.0:
        return float(floor_db)
    return max(float(floor_db), float(10.0 * np.log10(value)))


def _load_curve_dataframe(path):
    try:
        df = pd.read_csv(path)
        if "technique" in df.columns:
            mask = df["technique"].astype(str).map(_canonical_technique_name) == NMS_TECHNIQUE
            df = df.loc[mask].copy()
        return df
    except Exception as exc:
        print(f"[WARN] Impossibile leggere dati curva {path}: {exc}")
        return pd.DataFrame()


def _aggregate_curve_max_y_by_x(x_values, y_values, clamp_x_01=False):
    buckets = {}
    for xv, yv in zip(x_values, y_values):
        x = _curve_float(xv)
        y = _curve_float(yv)
        if x is None or y is None:
            continue
        if clamp_x_01:
            x = max(0.0, min(1.0, x))
        elif x < 0:
            continue
        y = _clamp01(y)
        key = round(float(x), 12)
        buckets[key] = max(y, buckets.get(key, 0.0))
    pairs = sorted(buckets.items())
    return [float(x) for x, _y in pairs], [float(y) for _x, y in pairs]


def _monotone_non_decreasing(values):
    out = []
    best = 0.0
    for value in values:
        best = max(best, _clamp01(value))
        out.append(best)
    return out


def _precision_envelope(values):
    out = [_clamp01(v) for v in values]
    best = 0.0
    for idx in range(len(out) - 1, -1, -1):
        best = max(best, out[idx])
        out[idx] = best
    return out


def _limit_curve_x(xs, ys, x_max):
    if not xs or not ys:
        return [], []
    if xs[0] > x_max:
        return [], []
    if xs[-1] <= x_max:
        return xs, ys
    y_at_limit = float(np.interp(float(x_max), np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
    kept = [(x, y) for x, y in zip(xs, ys) if x <= x_max]
    if not kept or kept[-1][0] < x_max:
        kept.append((float(x_max), y_at_limit))
    return [float(x) for x, _y in kept], [float(y) for _x, y in kept]


def _curve_area(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    area = 0.0
    for idx in range(1, len(xs)):
        width = max(0.0, float(xs[idx]) - float(xs[idx - 1]))
        area += width * (float(ys[idx]) + float(ys[idx - 1])) / 2.0
    return float(area)


def _continuous_curve_model(curve_name):
    path = _resolve_curve_data_path(curve_name)
    if not path:
        return None
    df = _load_curve_dataframe(path)
    if df.empty:
        return None

    color = ACCENT_BLUE
    if curve_name == "FROC":
        if "fp_per_image" not in df.columns:
            return None
        y_col = "sensitivity" if "sensitivity" in df.columns else "recall"
        if y_col not in df.columns:
            return None
        xs, ys = _aggregate_curve_max_y_by_x(df["fp_per_image"], df[y_col])
        ys = _monotone_non_decreasing(ys)
        xs, ys = _limit_curve_x(xs, ys, 5.0)
        # La scala log non ammette sensibilita' = 0: tengo i punti > 0, cosi' la curva
        # parte dal primo valore reale.
        pts = [(xi, yi) for xi, yi in zip(xs, ys) if yi > 0]
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Limite inferiore Y adattato ai dati: la curva riempie il grafico.
        y_lower = max(0.01, min(ys) * 0.8)
        # CPM = media delle sensibilita' interpolate ai FP/img fissi {0.125…4}:
        # sintesi standard della FROC, mostrata in legenda.
        cpm_points = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
        cpm_sens = list(np.interp(cpm_points, xs, ys, left=0.0, right=ys[-1]))
        cpm = float(np.mean(cpm_sens))
        # Pallini sui punti (FP/img, sensibilità) usati per il CPM (solo y > 0, log).
        mk = [(p, s) for p, s in zip(cpm_points, cpm_sens) if s > 0]
        markers = {"x": [p for p, _ in mk], "y": [s for _, s in mk],
                   "color": color, "label": "punti CPM (0.125…4 FP/img)"}
        # Se è stato fatto uno sweep NMS su validation, mostro qual è il vincitore.
        _params = _load_best_postprocess_params() or _best_postprocess_params or {}
        _win = _params.get("nms_selected") if isinstance(_params, dict) else None
        win_txt = f"NMS operativa: {float(NMS_IOU_THRESHOLD):.2f} · "
        footer = (f"{win_txt}FROC NMS · asse Y (sensibilità) in scala logaritmica · FP/img 0-5 · "
                  f"CPM={cpm:.3f} (media dei pallini) · " + _curve_threshold_note())
        return {
            "title": "FROC · sensibilità in scala logaritmica",
            "xlabel": "Falsi positivi per immagine (FP/img)",
            "ylabel": "Sensibilità / Recall (scala log)",
            "xlim": (0.0, 5.0),
            "ylim": (y_lower, 1.05),
            "footer": footer,
            "source": str(path),
            "series": [{"label": f"NMS · CPM={cpm:.3f}", "x": xs, "y": ys, "color": color}],
            "markers": markers,
            "reference": None,
            "y_scale": "log",
            "y_kind": "frac",
        }

    if curve_name == "FROC-validation":
        # Le 3 FROC ottenute sulla validation (una per soglia NMS), in un'unica vista.
        if "fp_per_image" not in df.columns or "nms" not in df.columns:
            return None
        y_col = "sensitivity" if "sensitivity" in df.columns else "recall"
        if y_col not in df.columns:
            return None
        palette = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_INDIGO, ACCENT_ORANGE, ACCENT_RED]
        nms_values = sorted({_curve_float(v, 0.0) for v in df["nms"]})
        series = []
        y_min = 1.0
        for i, nv in enumerate(nms_values):
            sub = df[(df["nms"].astype(float) - nv).abs() < 1e-6]
            xs, ys = _aggregate_curve_max_y_by_x(sub["fp_per_image"], sub[y_col])
            ys = _monotone_non_decreasing(ys)
            xs, ys = _limit_curve_x(xs, ys, 5.0)
            pts = [(xi, yi) for xi, yi in zip(xs, ys) if yi > 0]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            y_min = min(y_min, min(ys))
            f1 = _curve_float(sub["nms_f1"].iloc[0], 0.0) if "nms_f1" in sub.columns and len(sub) else 0.0
            is_win = bool(int(_curve_float(sub["nms_selected"].iloc[0], 0))) if "nms_selected" in sub.columns and len(sub) else False
            label = f"NMS {nv:.2f} · F1={f1:.3f}" + ("  ★ scelto" if is_win else "")
            series.append({"label": label, "x": xs, "y": ys, "color": palette[i % len(palette)]})
        if not series:
            return None
        y_lower = max(0.01, y_min * 0.8)
        return {
            "title": "FROC validation · confronto soglie NMS",
            "xlabel": "Falsi positivi per immagine (FP/img)",
            "ylabel": "Sensibilità / Recall (scala log)",
            "xlim": (0.0, 5.0),
            "ylim": (y_lower, 1.05),
            "footer": "Le 3 FROC ottenute sulla validation, una per soglia NMS · ★ = NMS vincente (scelto per F1).",
            "source": str(path),
            "series": series,
            "reference": None,
            "y_scale": "log",
            "y_kind": "frac",
            "fill": False,
        }

    if curve_name == "Precision-Recall":
        if "recall" not in df.columns or "precision" not in df.columns:
            return None
        xs, ys = _aggregate_curve_max_y_by_x(df["recall"], df["precision"], clamp_x_01=True)
        ys = _precision_envelope(ys)
        if xs and xs[0] > 0:
            xs = [0.0] + xs
            ys = [ys[0]] + ys
        xs, ys = _limit_curve_x(xs, ys, 1.0)
        ap_values = _ap_values_by_technique_from_files()
        ap = ap_values.get(NMS_TECHNIQUE, _curve_area(xs, ys))
        label = f"NMS · AP={float(ap):.4f}"
        return {
            "title": "Precision-Recall",
            "xlabel": "Recall / Sensibilità",
            "ylabel": "Precision",
            "xlim": (0.0, 1.0),
            "ylim": (0.0, 1.02),
            "footer": _precision_recall_ap_footer_text(),
            "source": str(path),
            "series": [{"label": label, "x": xs, "y": ys, "color": color}],
            "reference": None,
        }

    # La curva ROC e' stata rimossa dalla GUI: restano solo FROC e Precision-Recall.
    return None


def _evaluate_curve_points_csv_path():
    data = _load_best_postprocess_params() or _best_postprocess_params or {}
    path_value = data.get("curve_points_csv") if isinstance(data, dict) else None
    candidates = []
    if path_value:
        candidates.append(Path(path_value))
    candidates.extend([
        EVALUATION_CURVES_DIR / "curve_points_nms_only.csv",
        Path.cwd() / "ML" / "risultati_valutazione_postprocessing" / "curve" / "curve_points_nms_only.csv",
    ])
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                return path
        except Exception:
            pass
    return candidates[-1]


def _load_evaluate_best_rows_by_technique():
    """Legge curve_points_nms_only.csv e restituisce la riga NMS a soglia ML 0.50."""
    csv_path = _evaluate_curve_points_csv_path()
    if not csv_path or not csv_path.exists():
        return {}
    rows_by_tech = {}
    try:
        import csv as _csv
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                tech = _canonical_technique_name(row.get("technique", NMS_TECHNIQUE))
                if tech != NMS_TECHNIQUE:
                    continue
                try:
                    thr = float(row.get("threshold", 0.0))
                    distance = abs(thr - 0.50)
                except Exception:
                    distance = 999.0
                old = rows_by_tech.get(tech)
                old_dist = old.get("_distance_from_fixed_ml", 999.0) if old else 999.0
                if old is None or distance < old_dist:
                    clean = dict(row)
                    clean["_distance_from_fixed_ml"] = distance
                    clean["_metric_key"] = "FIXED_ML_0_50"
                    clean["_csv_path"] = str(csv_path)
                    rows_by_tech[tech] = clean
        return rows_by_tech
    except Exception as exc:
        print(f"[WARN] Impossibile leggere curve_points_nms_only.csv: {exc}")
        return {}

def _evaluate_best_row_for_technique(technique):
    """Restituisce la riga evaluate NMS, se disponibile."""
    technique = _canonical_technique_name(technique)
    try:
        rows = _load_evaluate_best_rows_by_technique()
        row = rows.get(technique)
        if row:
            return row
    except Exception:
        pass
    data = _load_best_postprocess_params() or _best_postprocess_params or {}
    if isinstance(data, dict) and _canonical_technique_name(data.get("technique")) == technique:
        return data
    return {}


def _threshold_for_technique(technique, default=None):
    """Soglia ML operativa fissa per NMS.

    Regola richiesta: considerare SEMPRE le ROI con probabilita' ML >= 50%.
    Quindi la soglia trovata da evaluate NON viene usata per filtrare in GUI.
    Evaluate puo' ancora salvare curve/parametri, ma l'operativita' della GUI
    resta fissa a 0.50 inclusiva per NMS.
    """
    return 0.50

def _with_fresh_detection_labels(detections, gt_boxes):
    """Ricalcola TP/FP/FN/IGNORE su copie pulite delle detection.

    Evita metriche stale quando il viewer NMS viene aggiornato.
    """
    dets = [dict(d) for d in (detections or [])]
    gts = [dict(g) for g in (gt_boxes or [])]
    metrics = evaluate_detections(dets, gts)
    return dets, gts, metrics


def _technique_index_label(technique):
    technique = _canonical_technique_name(technique)
    try:
        return str(POSTPROCESS_TECHNIQUES.index(technique) + 1)
    except ValueError:
        return "?"


def _metric_line_from_row(prefix, row):
    if not row:
        return f"{prefix}: valori evaluate non disponibili. Esegui evaluate_test_postprocess_class3_nms_only.py."
    return (
        f"{prefix}: DET={_format_int(row.get('detections'))} · "
        f"TP={_format_int(row.get('TP'))} · FP={_format_int(row.get('FP'))} · "
        f"FN={_format_int(row.get('FN'))} · "
        f"P={_format_percent(row.get('precision'))} · R={_format_percent(row.get('recall'))} · "
        f"F1={_format_float(row.get('F1'))} · F2={_format_float(row.get('F2'))} · "
        f"ML>={_format_float(row.get('threshold'), 3)}"
    )


def _box_xyxy(box):
    x1 = float(box.get("x", 0.0))
    y1 = float(box.get("y", 0.0))
    x2 = x1 + float(box.get("width", 0.0))
    y2 = y1 + float(box.get("height", 0.0))
    return x1, y1, x2, y2


def _box_area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = _box_xyxy(a)
    bx1, by1, bx2, by2 = _box_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = _box_area_xyxy((ix1, iy1, ix2, iy2))
    union = _box_area_xyxy((ax1, ay1, ax2, ay2)) + _box_area_xyxy((bx1, by1, bx2, by2)) - inter
    return inter / union if union > 0 else 0.0


def _safe_best_gt_iou(det):
    """Best IoU normalizzata. Ritorna None se la detection non e' stata valutata."""
    try:
        value = det.get("best_gt_iou", None)
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return max(0.0, min(1.0, value))
    except Exception:
        return None


def _best_gt_iou_for_display(det, gt_boxes=None):
    """Restituisce sempre la migliore IoU visualizzabile.

    Prima usa best_gt_iou salvata dalla valutazione. Se manca, la ricalcola
    al volo contro le ground truth disponibili, così anche una ROI con
    IoU < 0.50 non appare come "non valutata" nel tooltip/pannello.
    """
    best_iou = _safe_best_gt_iou(det)
    if best_iou is not None:
        return best_iou
    if not gt_boxes:
        return None
    try:
        values = [_box_iou(det, gt) for gt in gt_boxes]
        if not values:
            return None
        best_iou = max(values)
        if not np.isfinite(best_iou):
            return None
        return max(0.0, min(1.0, float(best_iou)))
    except Exception:
        return None


def _is_tp_iou_match(best_iou, tp_iou=TP_IOU_THRESHOLD):
    """Criterio TP detection-level: confronto inclusivo, >=."""
    try:
        return float(best_iou) >= float(tp_iou)
    except Exception:
        return False


def _format_iou_for_ui(best_iou):
    return "non valutata" if best_iou is None else f"{float(best_iou):.3f}"


def _box_distance_px(a, b):
    """Distanza minima tra due rettangoli. Vale 0 se si intersecano."""
    ax1, ay1, ax2, ay2 = _box_xyxy(a)
    bx1, by1, bx2, by2 = _box_xyxy(b)
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def _safe_ml_score(roi):
    """Restituisce una probabilita/score ML valida in [0, 1]."""
    try:
        score = float(roi.get("ml_score", 0.0))
        if not np.isfinite(score):
            return 0.0
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def _filtered_ml_candidates(rois_list, min_ml_score=ML_MIN_SCORE):
    try:
        min_ml_score = float(min_ml_score)
    except Exception:
        min_ml_score = ML_MIN_SCORE
    min_ml_score = max(0.0, min(1.0, min_ml_score))
    candidates = []
    for roi in rois_list or []:
        try:
            score = _safe_ml_score(roi)
            if score >= min_ml_score:
                candidate = dict(roi)
                candidate["ml_score"] = score
                candidate["score"] = score
                candidate["ml_pred"] = 1
                candidates.append(candidate)
        except Exception:
            continue
    return candidates


def _group_label(group):
    labels = []
    for r in group:
        if "label" in r and not pd.isna(r.get("label")):
            try:
                labels.append(int(r.get("label")))
            except Exception:
                pass
    return 1 if any(v == 1 for v in labels) else (0 if labels else np.nan)


def _make_output_detection(seed, group, idx, technique, min_ml_score, extra=None, union_box=False):
    extra = extra or {}
    if union_box:
        x1 = min(float(r.get("x", 0.0)) for r in group)
        y1 = min(float(r.get("y", 0.0)) for r in group)
        x2 = max(float(r.get("x", 0.0)) + float(r.get("width", 1.0)) for r in group)
        y2 = max(float(r.get("y", 0.0)) + float(r.get("height", 1.0)) for r in group)
        x, y = x1, y1
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    else:
        x = float(seed.get("x", 0.0))
        y = float(seed.get("y", 0.0))
        w = float(max(1.0, float(seed.get("width", 1.0))))
        h = float(max(1.0, float(seed.get("height", 1.0))))
    score = max(_safe_ml_score(r) for r in group) if group else _safe_ml_score(seed)
    out = {
        "roi_id": f"{technique.upper().replace('_', '')[:3]}{idx}",
        "method": TECHNIQUE_LABELS.get(technique, technique),
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "score": float(score),
        "ml_score": float(score),
        "ml_pred": 1,
        "label": _group_label(group),
        "nms_count": int(len(group)),
        "seed_roi_id": str(seed.get("roi_id", "")),
        "seed_ml_score": float(_safe_ml_score(seed)),
        "ml_min_score": float(min_ml_score),
        "technique": technique,
        "technique_label": TECHNIQUE_LABELS.get(technique, technique),
        "source_roi_ids": ",".join(str(r.get("roi_id", "")) for r in group if str(r.get("roi_id", ""))),
        "source_methods": ",".join(sorted(set(str(r.get("method", "")) for r in group if str(r.get("method", ""))))),
    }
    out.update(extra)
    return out


def _nms_keep(candidates, nms_iou_threshold):
    """Non-Maximum Suppression: mantiene la ROI con score maggiore e sopprime duplicati."""
    remaining = sorted(candidates, key=_safe_ml_score, reverse=True)
    winners = []
    while remaining:
        best = remaining.pop(0)
        suppressed, survivors = [], []
        for cand in remaining:
            if _box_iou(best, cand) >= nms_iou_threshold:
                suppressed.append(cand)
            else:
                survivors.append(cand)
        remaining = survivors
        winners.append((best, suppressed))
    return winners


def _nms_rois(candidates, min_ml_score=ML_MIN_SCORE, nms_iou_threshold=NMS_IOU_THRESHOLD):
    try:
        nms_iou_threshold = max(0.0, min(1.0, float(nms_iou_threshold)))
    except Exception:
        nms_iou_threshold = NMS_IOU_THRESHOLD
    out = []
    for seed, suppressed in _nms_keep(candidates, nms_iou_threshold):
        group = [seed] + suppressed
        out.append(_make_output_detection(
            seed, group, len(out) + 1, NMS_TECHNIQUE, min_ml_score,
            extra={"nms_iou_threshold": float(nms_iou_threshold)},
            union_box=False,
        ))
    return out


def postprocess_rois_by_technique(
    rois_list,
    technique=NMS_TECHNIQUE,
    min_ml_score=ML_MIN_SCORE,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
):
    """Applica il post-processing NMS come unica modalita'."""
    candidates = _filtered_ml_candidates(rois_list, min_ml_score=min_ml_score)
    return _nms_rois(candidates, min_ml_score=min_ml_score, nms_iou_threshold=nms_iou_threshold)


def postprocess_ml_rois(
    rois_list,
    min_ml_score=ML_MIN_SCORE,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
    technique=None,
):
    """Wrapper storico pulito: restituisce sempre le ROI finali NMS."""
    return postprocess_rois_by_technique(
        rois_list,
        technique=NMS_TECHNIQUE,
        min_ml_score=min_ml_score,
        nms_iou_threshold=nms_iou_threshold,
    )


# NB: _suppress_overlapping_non_tp_duplicates rimossa con l'ignore: ogni detection
# non-TP conta come FP, senza piu' la categoria IGNORE.

def evaluate_detections(nms_rois, gt_boxes,
                               tp_iou=TP_IOU_THRESHOLD, ignore_iou=IGNORE_IOU_THRESHOLD):
    """Calcola metriche detection-level: TP se best IoU e' >= 0.50."""
    try:
        tp_iou = float(tp_iou)
    except Exception:
        tp_iou = TP_IOU_THRESHOLD
    try:
        ignore_iou = float(ignore_iou)
    except Exception:
        ignore_iou = IGNORE_IOU_THRESHOLD
    gt_boxes = gt_boxes or []
    nms_rois = nms_rois or []
    for gt in gt_boxes:
        gt["matched"] = False
    for det in nms_rois:
        det.pop("post_label", None)
        best_iou = 0.0
        best_gi = None
        for gi, gt in enumerate(gt_boxes):
            iou = _box_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        det["best_gt_iou"] = float(best_iou)
        det["_best_gt"] = best_gi
        # Criterio TP coerente con le curve/metriche: IoU maggiore o uguale a 0.50.
        det["_tp_gt"] = best_gi if _is_tp_iou_match(best_iou, tp_iou) else None
    tp = 0
    gt_has_tp = [False] * len(gt_boxes)
    for gi, gt in enumerate(gt_boxes):
        contenders = [d for d in nms_rois if d.get("_tp_gt") == gi]
        if not contenders:
            continue
        # Se piu' ROI coprono la stessa GT, la TP e' quella con ML score piu' alto
        # (l'IoU resta solo come spareggio), non quella con IoU migliore.
        winner = max(contenders, key=lambda d: (_safe_ml_score(d), float(d.get("best_gt_iou", 0.0) or 0.0)))
        winner["post_label"] = "TP"
        gt["matched"] = True
        gt_has_tp[gi] = True
        tp += 1
    # Ignore RIMOSSO: ogni detection che non e' la TP della sua GT conta come FP
    # (valutazione standard, coerente con il testing).
    for det in nms_rois:
        if det.get("post_label") != "TP":
            det["post_label"] = "FP"
            det["ignore_reason"] = ""
            det["duplicate_kept_roi_id"] = ""
    fp = sum(1 for d in nms_rois if d.get("post_label") == "FP")
    ignored = 0
    fn = sum(1 for gt in gt_boxes if not gt.get("matched", False))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    beta2 = 4.0
    f2 = ((1.0 + beta2) * precision * recall / (beta2 * precision + recall)) if (beta2 * precision + recall) else 0.0
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "IGNORED": int(ignored),
        "GT": int(len(gt_boxes)), "detections": int(len(nms_rois)),
        "precision": float(precision), "recall": float(recall), "sensitivity": float(recall),
        "F1": float(f1), "F2": float(f2),
        "tp_iou": float(tp_iou), "tp_rule": f"IoU >= {float(tp_iou):.2f}",
        "ignore_iou": float(ignore_iou),
        "nms_iou_threshold": float(NMS_IOU_THRESHOLD),
    }

# Palette of hues for the continuously-cycling dynamic lights
# Each entry is an RGB triplet that gets cycled through over time
_DYN_PALETTE = [
    (37,  99, 235),   # electric blue
    (14, 165, 233),   # sky cyan
    (16, 185, 129),   # emerald
    (245,158, 11),    # amber
    (239, 68, 68),    # coral red
    (168, 85, 247),   # violet
    (20, 184,166),    # teal
]

def _dyn_color(phase_offset: float) -> str:
    """Return an RGB hex color that cycles smoothly through _DYN_PALETTE."""
    n = len(_DYN_PALETTE)
    t = (phase_offset % (2 * math.pi)) / (2 * math.pi) * n
    i = int(t) % n
    j = (i + 1) % n
    f = t - int(t)
    r = int(_DYN_PALETTE[i][0] + (_DYN_PALETTE[j][0] - _DYN_PALETTE[i][0]) * f)
    g = int(_DYN_PALETTE[i][1] + (_DYN_PALETTE[j][1] - _DYN_PALETTE[i][1]) * f)
    b = int(_DYN_PALETTE[i][2] + (_DYN_PALETTE[j][2] - _DYN_PALETTE[i][2]) * f)
    return f"#{r:02X}{g:02X}{b:02X}"

# ── ML backend ─────────────────────────────────────────────────────────────────
def _load_training_helpers_module():
    """Carica le funzioni dal file di addestramento/fracture senza obbligare un nome unico."""
    candidates = [
        Path(__file__).with_name("addestramento.py"),
        Path(__file__).with_name("addestramento_ML_risultati_v2.py"),
        Path(__file__).with_name("fracture_fp_filter.py"),
        WINDOWS_PROJECT_ROOT / "addestramento.py",
        WINDOWS_PROJECT_ROOT / "addestramento_ML_risultati_v2.py",
        WINDOWS_PROJECT_ROOT / "fracture_fp_filter.py",
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                spec = importlib.util.spec_from_file_location("training_helpers_module", str(candidate))
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules["training_helpers_module"] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "_extract_features") and hasattr(mod, "_get_scores"):
                    return mod
        except Exception:
            continue
    return None

_helpers_mod = _load_training_helpers_module()
if _helpers_mod is not None:
    _extract_features = _helpers_mod._extract_features
    _get_scores = _helpers_mod._get_scores
else:
    def _extract_features(df, feature_cols, derived_features, method_levels):
        arrays = []
        for col in feature_cols:
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            else:
                v = np.zeros(len(df), dtype=np.float64)
            arrays.append(v[:, None])
        def _n(c): return pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "score" in derived_features: arrays.append(_n("score")[:, None])
        if "width" in derived_features: arrays.append(_n("width")[:, None])
        if "height" in derived_features: arrays.append(_n("height")[:, None])
        if "area" in derived_features: arrays.append((_n("width") * _n("height"))[:, None])
        if "aspect" in derived_features: arrays.append((_n("width") / np.maximum(_n("height"), 1e-12))[:, None])
        if "log_area" in derived_features: arrays.append(np.log1p(_n("width") * _n("height"))[:, None])
        if "score_area" in derived_features: arrays.append((_n("score") * _n("width") * _n("height"))[:, None])
        if "method_one_hot" in derived_features:
            ms = df.get("method", pd.Series([""] * len(df))).astype(str).to_numpy()
            for lv in method_levels: arrays.append((ms == lv).astype(np.float64)[:, None])
        if not arrays:
            raise ValueError("Nessuna feature disponibile per l'inferenza ML.")
        X = np.hstack(arrays).astype(np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), []

    def _get_scores(pipe, X):
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(X)[:, 1]
        d = pipe.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))

def load_16bit_image_as_rgb(path):
    img = Image.open(path)
    if img.mode in ("I;16","I","F"):
        a = np.array(img, dtype=np.float32)
        mn, mx = a.min(), a.max()
        a = (a-mn)/(mx-mn)*255.0 if mx>mn else a*0
        return Image.fromarray(a.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


# ── Anteprime delle feature dell'image processing ─────────────────────────────
# Il C++ salva nel CSV il crop 96x96 effettivamente usato per GLCM/LBP/HOG.
# Le anteprime sotto partono proprio da quel crop, non dalla radiografia intera.
def _feature_input_crop(roi, fallback_image=None):
    raw_png = roi.get("_feature_image_png")
    if raw_png:
        try:
            return Image.open(BytesIO(raw_png)).convert("L")
        except Exception:
            pass

    if fallback_image is None:
        return Image.new("L", (96, 96), 0)

    x = int(max(0, float(roi.get("x", 0))))
    y = int(max(0, float(roi.get("y", 0))))
    w = int(max(1, float(roi.get("width", 1))))
    h = int(max(1, float(roi.get("height", 1))))
    return fallback_image.convert("L").crop((x, y, x + w, y + h)).resize((96, 96), Image.Resampling.BILINEAR)


def _lbp_feature_preview(gray_roi):
    """Mappa LBP con la stessa codifica a 8 vicini usata in main.cpp."""
    arr = np.asarray(gray_roi.convert("L"), dtype=np.uint8)
    lbp = np.zeros_like(arr, dtype=np.uint8)
    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
        c = arr[1:-1, 1:-1]
        lbp[1:-1, 1:-1] = (
            ((arr[:-2, :-2] >= c).astype(np.uint8) << 7)
            | ((arr[:-2, 1:-1] >= c).astype(np.uint8) << 6)
            | ((arr[:-2, 2:] >= c).astype(np.uint8) << 5)
            | ((arr[1:-1, 2:] >= c).astype(np.uint8) << 4)
            | ((arr[2:, 2:] >= c).astype(np.uint8) << 3)
            | ((arr[2:, 1:-1] >= c).astype(np.uint8) << 2)
            | ((arr[2:, :-2] >= c).astype(np.uint8) << 1)
            | ((arr[1:-1, :-2] >= c).astype(np.uint8) << 0)
        )
    return Image.fromarray(lbp, mode="L").convert("RGB")


def _glcm_feature_preview(gray_roi, levels=16):
    """Heatmap visuale della GLCM simmetrica a 16 livelli usata dal C++."""
    arr = np.asarray(gray_roi.convert("L"), dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        q = np.floor(((arr - mn) / (mx - mn)) * levels).astype(np.int32)
        q = np.clip(q, 0, levels - 1)
    else:
        q = np.zeros(arr.shape, dtype=np.int32)

    glcm = np.zeros((levels, levels), dtype=np.float64)
    for dx, dy in ((1, 0), (1, -1), (0, -1), (-1, -1)):
        y0, y1 = max(0, -dy), min(q.shape[0], q.shape[0] - dy)
        x0, x1 = max(0, -dx), min(q.shape[1], q.shape[1] - dx)
        a = q[y0:y1, x0:x1].ravel()
        b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        np.add.at(glcm, (a, b), 1.0)
        np.add.at(glcm, (b, a), 1.0)
    if glcm.sum() > 0:
        glcm /= glcm.sum()
    display = np.log1p(glcm * 1000.0)
    if display.max() > 0:
        display = display / display.max() * 255.0
    return Image.fromarray(display.astype(np.uint8), mode="L").resize((96, 96), Image.Resampling.NEAREST).convert("RGB")


def _gradient_feature_preview(gray_roi):
    """Anteprima del gradiente: supporto visuale dell'informazione HOG."""
    arr = np.asarray(gray_roi.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(arr)
    magnitude = np.hypot(gx, gy)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0
    return Image.fromarray(magnitude.astype(np.uint8), mode="L").convert("RGB")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _image_group_for_inference(img_path: Path) -> str:
    return f"sha256:{_sha256_file(Path(img_path).resolve())}"


def _raise_if_training_image(pkg: dict, img_path: Path) -> None:
    train_groups = set(pkg.get("split_image_groups", {}).get("train", []))
    if train_groups and _image_group_for_inference(img_path) in train_groups:
        raise ValueError(
            "Test bloccato: questa immagine appartiene al training set del modello. "
            "Scegli un'immagine non utilizzata nell'addestramento."
        )
    if not train_groups:
        print("[WARN] Il modello non salva gli identificativi delle immagini di training: "
              "controllo anti-leakage UI non disponibile.")


def _path_is_inside(path: Path, parent: Path) -> bool:
    """True solo se path e' dentro parent. Non crea, non copia e non cancella file."""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except Exception:
        return False


def _safe_existing_roots(*roots: Path):
    """Restituisce cartelle esistenti deduplicate per LETTURA.

    Le cartelle sotto ML/risultati_pipeline sono usate per file temporanei/output.
    IPA_CSV_FEATURE_DIR e' accettata solo come sorgente CSV in sola lettura.
    """
    out = []
    seen = set()
    for root in roots:
        try:
            root = Path(root)
            if not root.exists() or not root.is_dir():
                continue
            # Per la lettura dei CSV consento anche IPA_CSV_FEATURE_DIR, ma non ci scrivo mai.
            if not (_path_is_inside(root, PIPELINE_RESULTS_DIR) or root.resolve() == PIPELINE_RESULTS_DIR.resolve() or root.resolve() == IPA_CSV_FEATURE_DIR.resolve()):
                continue
            key = str(root.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(root.resolve())
        except Exception:
            continue
    return out


def _find_feature_csv_output(img_path: Path | None = None):
    """Trova un CSV feature gia' presente.

    La GUI non avvia piu' il C++ e non modifica la cartella IPA. Per comodita'
    puo' LEGGERE in sola lettura i CSV ufficiali in IPA/CSV_feature oppure i CSV
    copiati sotto ML/risultati_pipeline. Output e temporanei restano sempre in ML.
    """
    wanted_names = [
        "roi_feature_glcm_lbp_hog_labeled.csv",
        "roi_lbp_glcm_hog_features_labeled.csv",
        "roi_feature_glcm_labeled.csv",
        "roi_feature_hog_labeled.csv",
        "roi_feature_lbp_labeled.csv",
        "fratture_mancate_FN.csv",
    ]
    roots = _safe_existing_roots(
        CSV_FEATURE_DIR,
        PIPELINE_RUNTIME_DIR,
        PIPELINE_RUNTIME_DIR / "CSV_feature",
        PIPELINE_RUNTIME_DIR / "CSV",
        PIPELINE_RESULTS_DIR / "csv",
        PIPELINE_RESULTS_DIR / "CSV",
        PIPELINE_RESULTS_DIR,
        IPA_CSV_FEATURE_DIR,
    )
    if not roots:
        return None

    image_stem = ""
    try:
        image_stem = Path(img_path).stem.lower() if img_path else ""
    except Exception:
        image_stem = ""

    candidates = []
    seen = set()

    # Prima i nomi standard nelle cartelle principali.
    for root in roots:
        for name in wanted_names:
            candidate = root / name
            try:
                if candidate.exists() and candidate.is_file():
                    key = str(candidate.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        candidates.append(candidate.resolve())
            except Exception:
                continue

    # Poi qualunque CSV dentro le cartelle di lettura consentite.
    for root in roots:
        try:
            for candidate in root.rglob("*.csv"):
                if not candidate.is_file():
                    continue
                if not any(_path_is_inside(candidate, root) or candidate.resolve() == root.resolve() for root in roots):
                    continue
                key = str(candidate.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate.resolve())
        except Exception:
            continue

    if not candidates:
        return None

    def _csv_rank(path: Path):
        try:
            stat = path.stat()
            mtime = float(stat.st_mtime)
        except Exception:
            mtime = 0.0
        name = path.name.lower()
        stem_match = 1 if image_stem and image_stem in name else 0
        standard_name = 1 if path.name in wanted_names else 0
        # Ordine decrescente: match immagine, nome standard, piu' recente.
        return (stem_match, standard_name, mtime)

    return sorted(candidates, key=_csv_rank, reverse=True)[0]


def _feature_image_candidate_paths(raw_path: str, base_dir: Path | None = None, runtime_roi_dir: Path | None = None):
    """Percorsi possibili del crop ROI, limitati a ML/risultati_pipeline.

    Se il CSV contiene un path assoluto fuori da ML/risultati_pipeline, viene
    ignorato: in questo modo la GUI non legge, non rinomina e non modifica file
    nella cartella IPA o in altre cartelle esterne.
    """
    raw_path = str(raw_path or "").strip().strip('"').replace("\\", "/")
    if not raw_path:
        return []

    base_dir = Path(base_dir or PIPELINE_RESULTS_DIR)
    runtime_roi_dir = Path(runtime_roi_dir or ROI_CLASSIFICATE_DIR)
    allowed_roots = _safe_existing_roots(
        PIPELINE_RESULTS_DIR,
        PIPELINE_RUNTIME_DIR,
        CSV_FEATURE_DIR,
        ROI_CLASSIFICATE_DIR,
        base_dir,
        runtime_roi_dir,
    )

    out = []
    p = Path(raw_path)
    parts = [part for part in raw_path.split("/") if part not in ("", ".")]
    basename = Path(parts[-1]).name if parts else p.name

    if p.is_absolute():
        # Usa un path assoluto solo se rimane dentro ML/risultati_pipeline.
        if any(_path_is_inside(p, root) or p.resolve() == root.resolve() for root in allowed_roots):
            out.append(p)
    else:
        for root in allowed_roots:
            if parts:
                out.append(root.joinpath(*parts))
            if basename:
                out.append(root / basename)

    # Ricerca per basename solo dentro risultati_pipeline.
    if basename:
        for root in allowed_roots:
            try:
                for candidate in root.rglob(basename):
                    if _path_is_inside(candidate, PIPELINE_RESULTS_DIR):
                        out.append(candidate)
            except Exception:
                pass

    unique = []
    seen = set()
    for candidate in out:
        try:
            candidate = Path(candidate)
            if not candidate.exists() or not candidate.is_file():
                continue
            if not _path_is_inside(candidate, PIPELINE_RESULTS_DIR):
                continue
            key = str(candidate.resolve()).lower()
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(candidate.resolve())
    return unique


def _read_feature_image_bytes_for_roi(roi: dict, base_dir: Path | None = None, runtime_roi_dir: Path | None = None):
    """Legge il crop feature solo da ML/risultati_pipeline, senza toccare IPA."""
    rel_path = str(roi.get("roi_file", "")).strip()
    for feature_file in _feature_image_candidate_paths(rel_path, base_dir, runtime_roi_dir):
        try:
            return feature_file.read_bytes()
        except OSError:
            continue
    return None


def run_pipeline_inference(img_path):
    """Analisi ML/NMS da CSV feature gia' presente.

    Questa funzione mantiene il nome storico per non rompere la GUI, ma NON
    esegue piu' la pipeline C++ e NON avvia eseguibili esterni. Puo' leggere in
    sola lettura i CSV ufficiali dentro IPA/CSV_feature, ma non scrive, rinomina,
    cancella o modifica file in IPA. Tutti i temporanei/log/output restano sotto:
        <progetto>/ML/risultati_pipeline
    """
    _ensure_pipeline_output_dirs()
    img_path = Path(img_path)

    csv = _find_feature_csv_output(img_path)
    if csv is None:
        raise FileNotFoundError(
            "CSV feature non trovato.\n\n"
            "La GUI non esegue piu' la pipeline C++ e non modifica la cartella IPA. "
            "Cerca i CSV ufficiali in sola lettura qui:\n"
            f"  {IPA_CSV_FEATURE_DIR}\n"
            "oppure sotto ML/risultati_pipeline qui:\n"
            f"  {CSV_FEATURE_DIR}\n"
            f"  {PIPELINE_RUNTIME_DIR}\n"
        )

    print(f"[INFO] CSV feature caricato: {csv}")
    df = pd.read_csv(csv)
    df = _filter_feature_df_for_image(df, img_path)
    if len(df) == 0:
        print("[WARN] CSV feature vuoto, nessuna ROI disponibile.")
        gt_boxes = _load_label_boxes_for_image(img_path)
        return {
            "rois": [],
            "gt_boxes": gt_boxes,
            "nms_rois": [],
            "post_metrics": {},
            "postprocess_results": {},
            "postprocess_metrics": {},
            "postprocess_thresholds": {},
            "selected_postprocess_technique": NMS_TECHNIQUE,
            "ml_min_score": float(ML_MIN_SCORE),
            "feature_columns": [],
        }

    label_in_csv = "label" in df.columns
    label_count = int(df["label"].notna().sum()) if label_in_csv else 0

    # Carica le ground truth dal file YOLO corrispondente alla radiografia.
    gt_boxes = _load_label_boxes_for_image(img_path)
    if gt_boxes:
        src = gt_boxes[0].get("_source", "<unknown>")
        print(f"[INFO] Ground truth caricato da file label: {len(gt_boxes)} box da {src}.")
    else:
        print("[WARN] Nessun ground truth trovato nelle cartelle label/labels conosciute.")

    if label_in_csv and label_count > 0:
        pos = int((df["label"] == 1).sum())
        neg = int((df["label"] == 0).sum())
        print(f"[INFO] CSV Input: {len(df)} ROI, label presenti: {label_in_csv}, positivi: {pos}, negativi: {neg}.")

    # Modello e protezione anti-leakage per una valutazione indipendente.
    mp = TRAINING_MODEL_PATH
    with mp.open("rb") as f:
        pkg = pickle.load(f)
    _raise_if_training_image(pkg, img_path)

    # Compatibilita' con il nuovo package SVM_RBF e con modelli precedenti.
    if "feature_spec" in pkg:
        fs = pkg["feature_spec"]
        cols = fs.get("columns", [])
        der = fs.get("derived", [])
        met = fs.get("method_levels", [])
        sel = pkg.get("model_type", "single_best")
        ns = pkg.get("selected_names") or [pkg.get("selected_name")]
        pipes = pkg.get("pipelines", {})
        meta = pkg.get("stacking_meta_clf")
        if not ns or ns[0] not in pipes:
            raise ValueError("Package modello nuovo non valido: pipeline SVM_RBF mancante.")
    else:
        cols = pkg.get("feature_cols", [])
        der = pkg.get("derived_features", [])
        met = pkg.get("method_levels", [])
        pipeline = pkg.get("pipeline")
        if pipeline is None:
            raise ValueError("Non trovo una pipeline utilizzabile nel modello.")
        sel = "single"
        ns = ["single_model"]
        pipes = {"single_model": pipeline}
        meta = None

    X, _ = _extract_features(df, cols, der, met)
    thr = float(pkg.get("threshold", 0.5))

    # ── Inferenza ML ─────────────────────────────────────────────────────────
    if sel == "two_stage":
        stage1_name, stage2_name = ns[:2]
        stage1_scores = _get_scores(pipes[stage1_name], X)
        stage2_scores = _get_scores(pipes[stage2_name], X)
        first_thr = float(pkg.get("first_stage_threshold", thr))
        thr = float(pkg.get("second_stage_threshold", thr))
        scores = np.where(stage1_scores >= first_thr, stage2_scores, 0.0)
    elif sel in ("stacking", "stacking_advanced") and meta:
        mX = np.column_stack([_get_scores(pipes[n_], X) for n_ in ns])
        scores = meta.predict_proba(mX)[:, 1]
    elif sel == "weighted" and pkg.get("ensemble_weights"):
        ew = np.array(pkg["ensemble_weights"])
        scores = sum(w_ * _get_scores(pipes[n_], X) for n_, w_ in zip(ns, ew))
    elif sel == "ensemble":
        scores = np.mean([_get_scores(pipes[n_], X) for n_ in ns], axis=0)
    else:
        scores = _get_scores(pipes[ns[0]], X)

    thr = 0.50
    df["ml_score"] = scores
    df["ml_pred"] = (scores >= thr).astype(int)

    # ── POST-PROCESSING ML: NMS ─────────────────────────────────────────────
    _refresh_best_postprocess_params(verbose=True)
    effective_ml_threshold = float(ML_MIN_SCORE)
    df["nms_keep"] = (df["ml_score"] >= effective_ml_threshold).astype(int)
    raw_rois = df.to_dict(orient="records")

    # Carica eventuali crop feature solo da risultati_pipeline.
    for roi in raw_rois:
        feature_bytes = _read_feature_image_bytes_for_roi(roi, PIPELINE_RESULTS_DIR, ROI_CLASSIFICATE_DIR)
        if feature_bytes:
            roi["_feature_image_png"] = feature_bytes

    technique_results = {}
    technique_metrics = {}
    technique_thresholds = {}
    for technique in POSTPROCESS_TECHNIQUES:
        technique_threshold = _threshold_for_technique(technique, effective_ml_threshold)
        technique_thresholds[technique] = float(technique_threshold)
        dets_raw = postprocess_rois_by_technique(
            raw_rois,
            technique=technique,
            min_ml_score=technique_threshold,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
        )
        dets, _eval_gts, metrics = _with_fresh_detection_labels(dets_raw, gt_boxes)
        metrics["technique"] = technique
        metrics["technique_label"] = TECHNIQUE_LABELS.get(technique, technique)
        metrics["ml_min_score"] = float(technique_threshold)
        metrics["threshold"] = float(technique_threshold)
        metrics["discarded_low_ml_score"] = int((df["ml_score"] < technique_threshold).sum())
        technique_results[technique] = dets
        technique_metrics[technique] = metrics

    selected_technique = NMS_TECHNIQUE
    nms_rois = [dict(d) for d in technique_results.get(selected_technique, [])]
    post_metrics = dict(technique_metrics.get(selected_technique, {}))
    post_metrics["selected_technique"] = selected_technique

    print(
        "[INFO] Analisi ML/NMS da CSV completata: "
        f"soglia ML={effective_ml_threshold:.3f}, "
        f"ROI sopra soglia={int(df['nms_keep'].sum())}, "
        f"DET={post_metrics.get('detections', 0)} P={post_metrics.get('precision', 0.0):.3f} R={post_metrics.get('recall', 0.0):.3f}"
    )

    return {
        "rois": raw_rois,
        "gt_boxes": gt_boxes,
        "nms_rois": nms_rois,
        "post_metrics": post_metrics,
        "postprocess_results": technique_results,
        "postprocess_metrics": technique_metrics,
        "postprocess_thresholds": technique_thresholds,
        "selected_postprocess_technique": selected_technique,
        "ml_min_score": float(effective_ml_threshold),
        "feature_columns": list(cols),
    }

def save_annotated_image(img_path, rois_list, active_tab, gt_boxes=None):
    img = load_16bit_image_as_rgb(img_path)
    draw = ImageDraw.Draw(img)
    lw = max(2, int(min(img.size)/250))
    gt_boxes = gt_boxes or []
    if gt_boxes:
        for roi in gt_boxes:
            rx,ry,rw,rh = int(roi["x"]),int(roi["y"]),int(roi["width"]),int(roi["height"])
            # Nella vista post-nms una GT non abbinata e' un FN -> rossa; altrimenti viola.
            is_fn = active_tab != "cpp" and not roi.get("matched", False)
            gt_color = ACCENT_RED if is_fn else ACCENT_PURPLE
            draw.rectangle([rx,ry,rx+rw,ry+rh], outline=gt_color, width=lw)

    df = pd.DataFrame(rois_list)
    if len(df) > 0:
        if active_tab == "cpp":
            for _, row in df.iterrows():
                rx,ry,rw,rh = int(row["x"]),int(row["y"]),int(row["width"]),int(row["height"])
                draw.rectangle([rx,ry,rx+rw,ry+rh], outline=ACCENT_ORANGE, width=lw)
            sfx = "candidates"
        else:
            for _, row in df.iterrows():
                rx,ry,rw,rh = int(row["x"]),int(row["y"]),int(row["width"]),int(row["height"])
                post_label = row.get("post_label", None)
                if post_label == "TP":
                    draw.rectangle([rx,ry,rx+rw,ry+rh], outline=ACCENT_GREEN, width=lw)
                elif post_label == "FP":
                    draw.rectangle([rx,ry,rx+rw,ry+rh], outline=ACCENT_ORANGE, width=lw)
                elif post_label == "IGNORE":
                    continue   # detection su una GT gia' assegnata: ignorata, non disegnata
                else:
                    has_gt = "label" in row and not pd.isna(row["label"])
                    gt = int(row["label"]) if has_gt else None
                    pred = int(row["ml_pred"])
                    best_iou = _safe_best_gt_iou(row)
                    if pred == 1:
                        # In vista ML una detection e' verde solo se supera davvero il criterio TP IoU>=0.50.
                        color = ACCENT_GREEN if _is_tp_iou_match(best_iou) else ACCENT_ORANGE
                        draw.rectangle([rx,ry,rx+rw,ry+rh], outline=color, width=lw)
                    elif has_gt and gt == 1:
                        draw.rectangle([rx,ry,rx+rw,ry+rh], outline=ACCENT_RED, width=lw)
            sfx = "ml_filtered"
    dest = PIPELINE_ANNOTATED_DIR
    dest.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dest / f"{img_path.name.split('.')[0]}_{sfx}_{ts}.png"
    out = _safe_save_image(img, out)
    return out

# ── Floating Mouse Tooltip ─────────────────────────────────────────────────────
class ToolTip:
    def __init__(self, canvas):
        self.canvas = canvas
        self.tip_window = None
        self.text = ""
        self.label = None

    def show(self, text, x, y):
        cx = self.canvas.winfo_rootx() + x + 15
        cy = self.canvas.winfo_rooty() + y + 15
        if self.tip_window:
            if self.text == text:
                self.tip_window.wm_geometry(f"+{cx}+{cy}")
                return
            else:
                self.text = text
                if self.label: self.label.configure(text=text)
                self.tip_window.wm_geometry(f"+{cx}+{cy}")
                return
        self.text = text
        self.tip_window = tw = tk.Toplevel(self.canvas)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{cx}+{cy}")
        frame = tk.Frame(tw, bg="#1E293B", bd=0,
                         highlightbackground="#38BDF8", highlightthickness=1)
        frame.pack()
        self.label = tk.Label(frame, text=text, justify='left',
                              background="#1E293B", fg="#E2E8F0",
                              font=("Menlo", 9), padx=12, pady=10)
        self.label.pack()

    def hide(self):
        tw = self.tip_window
        self.tip_window = None
        self.text = ""
        self.label = None
        if tw:
            try: tw.destroy()
            except Exception: pass

# ── LED Dot Widget (canvas-based pulsing indicator) ───────────────────────────
class LEDDot(tk.Canvas):
    """A small circular LED that can pulse between two colors."""
    def __init__(self, parent, color_on, color_off="#CBD5E1", size=10, bg_color=BG_SIDEBAR, **kwargs):
        super().__init__(parent, width=size, height=size,
                         bg=bg_color, highlightthickness=0, **kwargs)
        self.color_on  = color_on
        self.color_off = color_off
        self.size = size
        self._dot = self.create_oval(1, 1, size-1, size-1,
                                     fill=color_off, outline="")
        self._state = False

    def set_state(self, active: bool):
        self._state = active
        self.itemconfig(self._dot, fill=self.color_on if active else self.color_off)

    def set_color(self, color: str):
        self.itemconfig(self._dot, fill=color)


# ── Main Application Window ────────────────────────────────────────────────────
class FractureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("IPA Clinical Review · Fracture Detection")
        self.geometry("1480x920")
        self.resizable(True, True)
        self.minsize(1220, 820)  # workstation diagnostica: navigazione e metriche sempre leggibili
        self.configure(fg_color=BG_MAIN)

        # State
        self.img_path        = None
        self.base_image      = None
        self.tk_image        = None
        self.rois            = []
        self.gt_boxes        = []
        self.nms_rois     = []
        self.post_metrics    = {}
        self.postprocess_results = {}
        self.postprocess_metrics = {}
        self.postprocess_thresholds = {}
        self.evaluate_best_rows_by_technique = _load_evaluate_best_rows_by_technique()
        self.selected_postprocess_technique = DEFAULT_POSTPROCESS_TECHNIQUE
        self.config_buttons = {}
        self.feature_columns = []          # colonne impiegate dal modello, mostrate nel pannello feature
        self.feature_roi_index = 0         # ROI C++ selezionata nella vista image processing
        self.feature_tk_images = {}        # riferimenti immagini Tk del pannello, evita garbage collection
        self.active_tab      = "ml"       # "cpp" = candidati image processing; "ml" = risultato finale
        self.is_running      = False
        self.image_folder    = None       # cartella attualmente sfogliata
        self.image_files     = []         # immagini disponibili nella cartella
        self.image_index     = -1         # posizione immagine corrente
        self.pending_image_index = None   # ultima navigazione richiesta durante l'analisi
        self.auto_analysis   = True       # ogni cambio immagine avvia la pipeline
        self._analysis_token = 0
        self._analysis_events = queue.Queue()  # worker -> thread grafico, in sicurezza
        self.curve_window    = None       # finestra riutilizzabile per grafici/curve di valutazione
        self.curve_figure    = None
        self.curve_ax        = None
        self.curve_canvas    = None
        self.curve_plot_host = None
        self.curve_motion_cid = None
        self.curve_click_cid = None
        self.curve_leave_cid = None
        self._curve_resize_after = None
        self._curve_last_canvas_size = None
        self._curve_interactive_series = []
        self.pulse           = 0          # integer tick counter
        self.phase           = 0.0        # float phase for smooth color cycle
        self.model_info      = self._load_model_info()
        self.global_metrics  = self._load_global_test_metrics()
        self.tooltip         = ToolTip(None)
        self.hovering_viewer = False
        self.hovering_dz     = False
        self.hovering_run    = False
        # Ripple rings: list of dicts {r, max_r, alpha_factor}
        self._ripples        = []
        self._ripple_tick    = 0          # spawn a new ring every N ticks

        self._build_ui()
        self._build_menu_bar()
        self.bind("<Control-o>", lambda _e: self.pick_file())
        self.bind("<Left>", lambda _e: self.previous_image())
        self.bind("<Right>", lambda _e: self.next_image())
        self.bind("<Key-i>", lambda _e: self.switch_tab("cpp"))
        self.bind("<Key-m>", lambda _e: self.switch_tab("ml"))
        self.bind("<Key-2>", lambda _e: self.select_postprocess_config(NMS_TECHNIQUE))
        self.after(70, self._poll_analysis_events)
        self._animate()

        self.lift()
        self.attributes("-topmost", True)
        self.after(100, lambda: self.attributes("-topmost", False))

    # ── Model Info ────────────────────────────────────────────────────────────
    def _load_model_info(self):
        mp = TRAINING_MODEL_PATH
        if not mp.exists():
            return {"active": False, "name": "Assente"}
        try:
            with mp.open("rb") as f: pkg = pickle.load(f)
            return {"active": True, "name": pkg.get("model_type", "single").upper()}
        except Exception:
            return {"active": False, "name": "Errore"}

    def _load_global_test_metrics(self):
        """Legge le metriche globali post-nms salvate dallo script evaluate."""
        defaults = {
            "TP": "—",
            "FP": "—",
            "FN": "—",
            "GT": "—",
            "DETECTIONS": "—",
            "PRECISION_PERCENT": "—",
            "RECALL_PERCENT": "—",
        }

        if not GLOBAL_METRICS_FILE.exists():
            return defaults

        try:
            values = {}
            with GLOBAL_METRICS_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    values[key.strip()] = value.strip()

            defaults.update(values)
            return defaults

        except Exception as exc:
            print(f"[WARN] Impossibile leggere metriche globali GUI: {exc}")
            return defaults
        
    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()

    # ── Application menu & performance curve viewer ──────────────────────────
    def _build_menu_bar(self):
        """La navigazione è integrata nell'interfaccia, non nella barra nativa OS.

        Le barre menu Tk native appaiono diverse su Windows/macOS e non permettono
        uno styling clinico coerente. I comandi principali sono quindi nella
        command bar professionale costruita in ``_build_main_area``.
        """
        try:
            self.configure(menu="")
        except tk.TclError:
            pass
        self.bind("<Control-m>", lambda _e: self.show_curve("FROC"))

    def _close_curve_window(self):
        if self.curve_window is not None and self.curve_window.winfo_exists():
            try:
                self.curve_window.grab_release()
            except tk.TclError:
                pass
            if getattr(self, "_curve_resize_after", None) is not None:
                try:
                    self.after_cancel(self._curve_resize_after)
                except tk.TclError:
                    pass
                self._curve_resize_after = None
            for cid_name in ("curve_motion_cid", "curve_click_cid", "curve_leave_cid"):
                cid = getattr(self, cid_name, None)
                if cid is not None and getattr(self, "curve_canvas", None) is not None:
                    try:
                        self.curve_canvas.mpl_disconnect(cid)
                    except Exception:
                        pass
                setattr(self, cid_name, None)
            self.curve_window.destroy()
        self.curve_window = None
        self.curve_figure = None
        self.curve_ax = None
        self.curve_canvas = None
        self.curve_plot_host = None
        self.curve_cross_v = None
        self.curve_cross_h = None
        self.curve_annotation = None
        self._curve_resize_after = None
        self._curve_last_canvas_size = None
        self._curve_interactive_series = []

    def _center_dialog(self, window, width, height):
        self.update_idletasks()
        x = self.winfo_rootx() + max(20, (self.winfo_width() - width) // 2)
        y = self.winfo_rooty() + max(20, (self.winfo_height() - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")

    def _style_curve_axes(self, ax, model):
        fig = ax.figure
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.set_title(model["title"], fontsize=14, fontweight="bold", color=TEXT_MAIN, pad=12)
        ax.set_xlabel(model["xlabel"], fontsize=11, color=TEXT_SUB)
        ax.set_ylabel(model["ylabel"], fontsize=11, color=TEXT_SUB)
        ax.set_xlim(*model["xlim"])
        ax.set_ylim(*model["ylim"])
        # Asse Y in scala logaritmica (FROC): tacche in percentuale leggibili.
        if model.get("y_scale") == "log":
            import matplotlib.ticker as _mticker
            ax.set_yscale("log")
            y_lo = model["ylim"][0]
            y_ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0) if t >= y_lo * 0.99]
            ax.yaxis.set_major_locator(_mticker.FixedLocator(y_ticks))
            ax.yaxis.set_major_formatter(_mticker.FuncFormatter(lambda v, _p: f"{v * 100:.0f}%"))
            ax.yaxis.set_minor_formatter(_mticker.NullFormatter())
            ax.grid(True, which="minor", color="#E6EDF4", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="major", color="#D7E0EA", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_MUTED, labelsize=10)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#8FA1B3")
            ax.spines[side].set_linewidth(1.0)
        for side in ("right", "top"):
            ax.spines[side].set_color("#D7E0EA")
            ax.spines[side].set_linewidth(0.8)

    def _schedule_curve_resize(self, event=None):
        if getattr(self, "curve_canvas", None) is None or getattr(self, "curve_figure", None) is None:
            return
        width = int(getattr(event, "width", 0) or self.curve_canvas.get_tk_widget().winfo_width())
        height = int(getattr(event, "height", 0) or self.curve_canvas.get_tk_widget().winfo_height())
        if width <= 1 or height <= 1:
            return
        if getattr(self, "_curve_resize_after", None) is not None:
            try:
                self.after_cancel(self._curve_resize_after)
            except tk.TclError:
                pass
        self._curve_resize_after = self.after(45, lambda w=width, h=height: self._resize_curve_figure(w, h))

    def _resize_curve_figure(self, width=None, height=None):
        self._curve_resize_after = None
        if getattr(self, "curve_canvas", None) is None or getattr(self, "curve_figure", None) is None:
            return
        widget = self.curve_canvas.get_tk_widget()
        width = int(width or widget.winfo_width())
        height = int(height or widget.winfo_height())
        if width <= 1 or height <= 1:
            return
        width = max(360, width)
        height = max(260, height)
        size_key = (width // 4, height // 4)
        if size_key == getattr(self, "_curve_last_canvas_size", None):
            return
        self._curve_last_canvas_size = size_key
        dpi = float(self.curve_figure.get_dpi() or 100.0)
        # Adatto la figura alla dimensione esatta del widget; constrained_layout
        # ridispone il contenuto al draw successivo, riempiendo lo spazio senza tagli.
        self.curve_figure.set_size_inches(width / dpi, height / dpi, forward=True)
        self.curve_canvas.draw_idle()

    def _hide_curve_cursor(self, reset_label=True):
        changed = False
        for artist in (getattr(self, "curve_cross_v", None), getattr(self, "curve_cross_h", None), getattr(self, "curve_annotation", None)):
            if artist is not None and artist.get_visible():
                artist.set_visible(False)
                changed = True
        if reset_label and getattr(self, "curve_coord_label", None) is not None:
            self.curve_coord_label.configure(text="X=—   Y=—")
        if changed and getattr(self, "curve_canvas", None) is not None:
            self.curve_canvas.draw_idle()

    def _on_curve_motion(self, event):
        ax = getattr(self, "curve_ax", None)
        if ax is None or event is None or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            self._hide_curve_cursor()
            return
        best = None
        for item in getattr(self, "_curve_interactive_series", []):
            x = item["x"]
            y = item["y"]
            if x.size == 0:
                continue
            xmin, xmax = float(x.min()), float(x.max())
            if float(event.xdata) < xmin or float(event.xdata) > xmax:
                continue
            if x.size == 1:
                curve_x = float(x[0])
                curve_y = float(y[0])
            else:
                curve_x = float(event.xdata)
                curve_y = float(np.interp(curve_x, x, y))
            px, py = ax.transData.transform((curve_x, curve_y))
            dist = ((float(event.x) - px) ** 2 + (float(event.y) - py) ** 2) ** 0.5
            if best is None or dist < best[0]:
                best = (dist, item["label"], curve_x, curve_y)

        if best is None or best[0] > 28.0:
            self._hide_curve_cursor()
            return

        _dist, label, curve_x, curve_y = best
        # Sulla FROC l'asse Y e' la sensibilita' (scala log): la mostro in %.
        if getattr(self, "_curve_y_kind", None) == "frac":
            y_text = f"{curve_y * 100:.1f}%"
        else:
            y_text = f"{curve_y:.4f}"
        self.curve_cross_v.set_xdata([curve_x, curve_x])
        self.curve_cross_h.set_ydata([curve_y, curve_y])
        self.curve_cross_v.set_visible(True)
        self.curve_cross_h.set_visible(True)
        self.curve_annotation.xy = (curve_x, curve_y)
        self.curve_annotation.set_text(f"{label}\nX={curve_x:.4f}\nY={y_text}")
        self.curve_annotation.set_visible(True)
        self.curve_coord_label.configure(text=f"{label}   X={curve_x:.4f}   Y={y_text}")
        self.curve_canvas.draw_idle()

    def _draw_interactive_curve(self, model):
        Figure, FigureCanvasTkAgg = _ensure_matplotlib_tk()
        if getattr(self, "curve_figure", None) is None:
            # constrained_layout: ridispone da solo titolo/assi/etichette/legenda a
            # QUALSIASI dimensione di finestra e DPI, senza mai tagliare nulla.
            self.curve_figure = Figure(figsize=(8.8, 6.0), dpi=100, facecolor="#FFFFFF",
                                       constrained_layout=True)
            try:
                self.curve_figure.set_constrained_layout_pads(
                    w_pad=0.10, h_pad=0.10, wspace=0.0, hspace=0.0)
            except Exception:
                pass
            self.curve_ax = self.curve_figure.add_subplot(111)
            self.curve_canvas = FigureCanvasTkAgg(self.curve_figure, master=self.curve_plot_host)
            widget = self.curve_canvas.get_tk_widget()
            widget.configure(bg="#FFFFFF", highlightthickness=0, bd=0)
            widget.grid(row=0, column=0, sticky="nsew")
            widget.bind("<Configure>", self._schedule_curve_resize, add="+")
            self.curve_motion_cid = self.curve_canvas.mpl_connect("motion_notify_event", self._on_curve_motion)
            self.curve_click_cid = self.curve_canvas.mpl_connect("button_press_event", self._on_curve_motion)
            self.curve_leave_cid = self.curve_canvas.mpl_connect("figure_leave_event", lambda _e: self._hide_curve_cursor())

        ax = self.curve_ax
        ax.clear()
        self._curve_interactive_series = []
        self._curve_y_kind = model.get("y_kind")
        self._style_curve_axes(ax, model)

        y_floor = model.get("ylim", (0.0, 1.0))[0]
        for series in model.get("series", []):
            x = np.asarray(series.get("x", []), dtype=float)
            y = np.asarray(series.get("y", []), dtype=float)
            if x.size == 0 or y.size == 0:
                continue
            col = series.get("color", ACCENT_BLUE)
            # Area evidenziata sotto la curva (riempimento tenue). Disattivata quando
            # si confrontano piu' curve (es. le 3 FROC di validation) per leggibilita'.
            if model.get("fill", True):
                ax.fill_between(x, y, y_floor, color=col, alpha=0.14, zorder=1)
            line, = ax.plot(
                x, y,
                color=col,
                linewidth=2.7,
                solid_capstyle="round",
                label=series.get("label", "NMS"),
                zorder=3,
            )
            self._curve_interactive_series.append({
                "artist": line,
                "label": series.get("label", "NMS"),
                "x": x,
                "y": y,
            })

        # Pallini sui punti FP/img fissi su cui si legge la sensibilita' per il CPM.
        markers = model.get("markers")
        if markers and markers.get("x"):
            ax.scatter(markers["x"], markers["y"], s=62, zorder=6,
                       color=markers.get("color", ACCENT_BLUE),
                       edgecolor="#FFFFFF", linewidth=1.4,
                       label=markers.get("label"))

        if self._curve_interactive_series:
            legend = ax.legend(loc="best", frameon=True, facecolor="#FFFFFF", edgecolor="#D7E0EA", framealpha=0.95)
            for text in legend.get_texts():
                text.set_color(TEXT_MAIN)
        else:
            ax.text(0.5, 0.5, model.get("empty_text", "Nessun punto curva disponibile"),
                    transform=ax.transAxes, ha="center", va="center", color=TEXT_MUTED, fontsize=12)

        self.curve_cross_v = ax.axvline(0, color="#64748B", linewidth=0.9, alpha=0.55, visible=False)
        self.curve_cross_h = ax.axhline(0, color="#64748B", linewidth=0.9, alpha=0.55, visible=False)
        self.curve_annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(14, 14),
            textcoords="offset points",
            fontsize=9.5,
            color=TEXT_MAIN,
            # Sfondo SOLIDO e opaco (niente trasparenza): il riquadro copre griglia e
            # curva sottostanti e resta sopra a tutto (zorder alto).
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F1F6FB", "edgecolor": ACCENT_BLUE,
                  "linewidth": 1.3, "alpha": 1.0},
            arrowprops={"arrowstyle": "-", "color": ACCENT_BLUE, "linewidth": 0.9},
            visible=False,
            zorder=30,
        )
        self.curve_canvas.draw_idle()
        self._schedule_curve_resize()
        self.curve_coord_label.configure(text="X=—   Y=—")
        self.curve_path_label.configure(text=model.get("footer", ""))

    def show_curve(self, curve_name):
        """Mostra le curve come dialog professionale sempre sopra la workstation."""
        curve_model = _continuous_curve_model(curve_name)
        if not curve_model:
            expected = "\n".join(str(p) for p in CURVE_DATA_CANDIDATES.get(curve_name, []))
            messagebox.showerror(
                "Metrica non disponibile",
                f"Non trovo i dati della curva {curve_name}.\n\nPercorsi cercati:\n{expected}"
            )
            return

        if self.curve_window is None or not self.curve_window.winfo_exists():
            self.curve_window = ctk.CTkToplevel(self)
            self.curve_window.withdraw()
            self.curve_window.title("Metriche di valutazione · IPA Clinical Review")
            self.curve_window.minsize(760, 600)
            self.curve_window.configure(fg_color="#F5F7FA")
            self.curve_window.transient(self)
            self.curve_window.protocol("WM_DELETE_WINDOW", self._close_curve_window)
            self.curve_window.grid_columnconfigure(0, weight=1)
            self.curve_window.grid_rowconfigure(2, weight=1)

            titlebar = ctk.CTkFrame(
                self.curve_window, height=74, fg_color="#FFFFFF",
                corner_radius=0, border_width=0
            )
            titlebar.grid(row=0, column=0, sticky="ew")
            titlebar.grid_columnconfigure(0, weight=1)
            titlebar.grid_propagate(False)
            ctk.CTkLabel(
                titlebar, text="METRICHE POST-PROCESSING", text_color=TEXT_MUTED,
                font=("Segoe UI", 10, "bold"), anchor="w"
            ).grid(row=0, column=0, padx=(26, 10), pady=(13, 0), sticky="w")
            self.curve_title = ctk.CTkLabel(
                titlebar, text="", text_color=TEXT_MAIN,
                font=("Segoe UI", 19, "bold"), anchor="w"
            )
            self.curve_title.grid(row=1, column=0, padx=(26, 10), pady=(0, 13), sticky="w")
            close_btn = ctk.CTkButton(
                titlebar, text="Chiudi", width=82, height=34, corner_radius=12,
                fg_color="#EEF3F7", hover_color="#E3EAF0", text_color=TEXT_SUB,
                font=("Segoe UI", 11, "bold"), command=self._close_curve_window
            )
            close_btn.grid(row=0, column=1, rowspan=2, padx=(0, 24), pady=18)

            metric_tabs = ctk.CTkFrame(
                self.curve_window, height=55, fg_color="#F5F7FA", corner_radius=0
            )
            metric_tabs.grid(row=1, column=0, sticky="ew", padx=22, pady=(14, 8))
            for col, (label, key) in enumerate([
                ("FROC TEST", "FROC"),
                ("FROC VALIDATION", "FROC-validation"),
                ("PRECISION–RECALL", "Precision-Recall"),
            ]):
                btn_width = 154 if key == "Precision-Recall" else (140 if key == "FROC-validation" else 104)
                btn = ctk.CTkButton(
                    metric_tabs, text=label, width=btn_width, height=38,
                    corner_radius=12, fg_color="#FFFFFF", hover_color="#DDECF0",
                    border_width=1, border_color=GLASS_BORDER, text_color=ACCENT_BLUE,
                    font=("Segoe UI", 11, "bold"),
                    command=lambda name=key: self.show_curve(name)
                )
                btn.grid(row=0, column=col, padx=(0, 8), pady=8, sticky="w")

            curve_card = ctk.CTkFrame(
                self.curve_window, fg_color="#FFFFFF", corner_radius=22,
                border_width=1, border_color=GLASS_BORDER
            )
            curve_card.grid(row=2, column=0, sticky="nsew", padx=22, pady=(0, 22))
            curve_card.grid_columnconfigure(0, weight=1)
            curve_card.grid_rowconfigure(0, weight=1)
            self.curve_plot_host = tk.Frame(curve_card, bg="#FFFFFF", highlightthickness=0, bd=0)
            self.curve_plot_host.grid(row=0, column=0, padx=18, pady=(18, 4), sticky="nsew")
            self.curve_plot_host.grid_columnconfigure(0, weight=1)
            self.curve_plot_host.grid_rowconfigure(0, weight=1)
            self.curve_coord_label = ctk.CTkLabel(
                curve_card, text="X=—   Y=—",
                text_color=TEXT_SUB, font=("Segoe UI", 11, "bold"), anchor="w"
            )
            self.curve_coord_label.grid(row=1, column=0, padx=20, pady=(4, 0), sticky="ew")
            self.curve_path_label = ctk.CTkLabel(
                curve_card, text=_best_postprocess_params_summary(),
                text_color=TEXT_MUTED, font=("Segoe UI", 10), anchor="w"
            )
            self.curve_path_label.grid(row=2, column=0, padx=20, pady=(4, 16), sticky="ew")

        display_names = {
            "Precision-Recall": "Precision–Recall",
            "FROC": "FROC (test)",
            "FROC-validation": "FROC validation (confronto NMS)",
        }
        display_name = display_names.get(curve_name, curve_name)
        self.curve_title.configure(text=f"Curva {display_name}")
        self.curve_window.title(f"Curva {display_name} · Metriche")
        self._draw_interactive_curve(curve_model)

        # L'ordine è intenzionale: sul primo click Windows deve mostrare subito
        # il dialog davanti alla finestra principale, senza flash in secondo piano.
        # Apro la finestra gia' ampia (adattata allo schermo) cosi' il grafico si
        # vede per intero da subito, senza doverla allargare a mano.
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        win_w = max(880, min(1280, sw - 120))
        win_h = max(700, min(1020, sh - 120))
        self._center_dialog(self.curve_window, win_w, win_h)
        self.curve_window.attributes("-topmost", True)
        self.curve_window.deiconify()
        self.curve_window.lift()
        self.curve_window.focus_force()
        self.curve_window.update_idletasks()
        if getattr(self, "curve_canvas", None) is not None:
            self._curve_last_canvas_size = None
            self._resize_curve_figure()
        try:
            self.curve_window.grab_set()
        except tk.TclError:
            pass
        self.curve_window.after(260, lambda: self.curve_window and self.curve_window.winfo_exists() and self.curve_window.attributes("-topmost", False))

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self, width=292, fg_color=BG_SIDEBAR, border_width=0, corner_radius=0
        )
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)

        self._sidebar_accent = tk.Frame(self.sidebar, bg="#18334D", height=1)
        self._sidebar_accent.place(x=0, y=0, relwidth=1.0)
        self.side_leds = []

        # ── Identità applicazione: pulita, senza etichette sovrapposte ────────
        brand = ctk.CTkFrame(
            self.sidebar, width=252, height=82, fg_color="#102237",
            border_width=1, border_color="#1C3752", corner_radius=20
        )
        brand.place(x=20, y=22)
        brand.grid_propagate(False)
        icon = ctk.CTkFrame(brand, width=42, height=42, fg_color="#0E7187", corner_radius=13)
        icon.place(x=16, y=19)
        icon.grid_propagate(False)
        ctk.CTkLabel(
            icon, text="IPA", text_color="#FFFFFF", font=("Segoe UI", 11, "bold")
        ).place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(
            brand, text="Clinical Review", text_color="#F8FAFC",
            font=("Segoe UI", 16, "bold"), anchor="w"
        ).place(x=70, y=18)
        ctk.CTkLabel(
            brand, text="FRACTURE DETECTION WORKSTATION", text_color="#8EA5BA",
            font=("Segoe UI", 8, "bold"), anchor="w"
        ).place(x=70, y=45)
        self.logo_led = LEDDot(brand, ACCENT_GREEN, "#315068", size=8, bg_color="#102237")
        self.logo_led.place(x=224, y=36)

        # ── Caricamento radiografia ──────────────────────────────────────────
        self.dropzone = ctk.CTkFrame(
            self.sidebar, width=252, height=286, fg_color="#FFFFFF",
            border_width=1, border_color="#D9E2EA", corner_radius=22
        )
        self.dropzone.place(x=20, y=126)
        self.dropzone.grid_propagate(False)
        self._dz_canvas = tk.Canvas(self.dropzone, bg="#FFFFFF", highlightthickness=0)
        self._dz_canvas.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
        self._dz_canvas.bind("<Configure>", self._draw_dz_dashes)

        self.dz_icon = tk.Label(
            self.dropzone, text="＋", bg="#FFFFFF", fg=ACCENT_BLUE,
            font=("Segoe UI", 32, "normal")
        )
        self.dz_icon.place(relx=0.5, rely=0.29, anchor="center")
        self.dz_main = tk.Label(
            self.dropzone, text="Apri radiografia", bg="#FFFFFF", fg=TEXT_MAIN,
            font=("Segoe UI", 14, "bold")
        )
        self.dz_main.place(relx=0.5, rely=0.47, anchor="center")
        self.dz_sub = tk.Label(
            self.dropzone, text="Analisi automatica e navigazione cartella", bg="#FFFFFF", fg=TEXT_MUTED,
            font=("Segoe UI", 10)
        )
        self.dz_sub.place(relx=0.5, rely=0.58, anchor="center")
        ctk.CTkLabel(
            self.dropzone, text="PNG  ·  JPG  ·  TIFF  ·  BMP", text_color="#8796A6",
            fg_color="#F4F7FA", corner_radius=10, height=24,
            font=("Segoe UI", 9, "bold")
        ).place(relx=0.5, rely=0.77, anchor="center")
        self.dz_filename = tk.Label(
            self.dropzone, text="", bg=GLASS_SOFT, fg=ACCENT_BLUE,
            font=("Segoe UI", 9, "bold"), padx=10, pady=3
        )
        for w in [self.dropzone, self._dz_canvas, self.dz_icon, self.dz_main, self.dz_sub]:
            w.bind("<ButtonPress-1>", self._on_dz_press)
            w.bind("<ButtonRelease-1>", self._on_dz_release)
            w.bind("<Enter>", self._on_dz_enter)
            w.bind("<Leave>", self._on_dz_leave)

        # ── Stato compatto: non è un pannello di navigazione/sequenza ───────
        status = ctk.CTkFrame(
            self.sidebar, width=252, height=112, fg_color="#102237",
            border_width=1, border_color="#1C3752", corner_radius=18
        )
        status.place(x=20, y=435)
        status.grid_propagate(False)
        ctk.CTkLabel(
            status, text="STATO ANALISI", text_color="#8EA5BA",
            font=("Segoe UI", 9, "bold"), anchor="w"
        ).place(x=16, y=14)
        pipe_row = tk.Frame(status, bg="#102237")
        pipe_row.place(x=16, y=40)
        self.pipe_led = LEDDot(pipe_row, ACCENT_GREEN, "#315068", size=8, bg_color="#102237")
        self.pipe_led.pack(side="left", padx=(0, 9), pady=4)
        self.pipe_led.set_state(False)
        self.pipe_lbl = tk.Label(
            pipe_row, text="In attesa di una radiografia", bg="#102237",
            fg="#D2DCE7", font=("Segoe UI", 10)
        )
        self.pipe_lbl.pack(side="left")
        ctk.CTkLabel(
            status, text="AUTO ANALYSIS", text_color="#D6F5F2", fg_color="#153C45",
            corner_radius=9, width=104, height=22, font=("Segoe UI", 8, "bold")
        ).place(x=16, y=74)

        # ── Azioni principali ────────────────────────────────────────────────
        self.sidebar_footer = ctk.CTkFrame(self.sidebar, width=252, height=166, fg_color="transparent")
        self.sidebar_footer.place(x=20, rely=1.0, y=-24, anchor="sw")
        self.btn_new = ctk.CTkButton(
            self.sidebar_footer, text="Nuova sessione", height=40, width=252,
            font=("Segoe UI", 12, "bold"), fg_color="#132840", hover_color="#18344F",
            text_color="#E3ECF4", border_width=1, border_color="#223D57", corner_radius=13,
            state="disabled", command=self.reset
        )
        self.btn_new.place(x=0, y=0)
        self.btn_save = ctk.CTkButton(
            self.sidebar_footer, text="Esporta risultato", height=40, width=252,
            font=("Segoe UI", 12, "bold"), fg_color="#132840", hover_color="#18344F",
            text_color="#E3ECF4", border_width=1, border_color="#223D57", corner_radius=13,
            state="disabled", command=self.save_results
        )
        self.btn_save.place(x=0, y=48)
        self.run_leds = []
        self.btn_run = ctk.CTkButton(
            self.sidebar_footer, text="Rianalizza immagine", height=54, width=252,
            font=("Segoe UI", 13, "bold"), fg_color=ACCENT_BLUE, hover_color="#075B73",
            text_color="#FFFFFF", border_width=0, corner_radius=15,
            state="disabled", command=self.run_analysis
        )
        self.btn_run.place(x=0, y=104)
        self.btn_run_sheen = tk.Frame(self.sidebar_footer, bg="#4EC0C5", height=1)
        self.btn_run_sheen.place(x=18, y=109, width=216)
        for btn, enter_cb, leave_cb in [
            (self.btn_new, self._on_btn_new_enter, self._on_btn_new_leave),
            (self.btn_save, self._on_btn_save_enter, self._on_btn_save_leave),
            (self.btn_run, self._on_btn_run_enter, self._on_btn_run_leave),
        ]:
            btn.bind("<Enter>", enter_cb)
            btn.bind("<Leave>", leave_cb)
        tk.Frame(self.sidebar, bg="#1A334C", height=1).place(
            x=0, rely=1.0, y=-1, relwidth=1.0, anchor="sw"
        )

    # ── Dashed border for dropzone ────────────────────────────────────────────
    def _draw_dz_dashes(self, event=None):
        c = self._dz_canvas
        c.delete("dash_border")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10: return
        r = 16
        dash = (8, 5)
        color = ACCENT_BLUE_L if self.hovering_dz else GLASS_BORDER
        c.create_line(18, 3, w - 18, 3, fill=GLASS_HILITE, width=1, tags="dash_border")
        # Draw dashed rounded rect approximated with lines
        c.create_line(r, 2, w-r, 2,           fill=color, dash=dash, width=1.5, tags="dash_border")
        c.create_line(w-2, r, w-2, h-r,        fill=color, dash=dash, width=1.5, tags="dash_border")
        c.create_line(w-r, h-2, r, h-2,        fill=color, dash=dash, width=1.5, tags="dash_border")
        c.create_line(2, h-r, 2, r,            fill=color, dash=dash, width=1.5, tags="dash_border")

    # ── MAIN AREA ─────────────────────────────────────────────────────────────
    def _build_main_area(self):
        self.main_area = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=26, pady=20)
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(5, weight=1)

        # ── Command bar interna: sostituisce il menu nativo non tematizzabile ─
        command_bar = ctk.CTkFrame(
            self.main_area, height=60, fg_color="#FFFFFF", border_width=1,
            border_color=GLASS_BORDER, corner_radius=18
        )
        command_bar.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        command_bar.grid_columnconfigure(1, weight=1)
        command_bar.grid_propagate(False)
        ctk.CTkLabel(
            command_bar, text="WORKLIST", text_color=TEXT_MUTED,
            font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=0, padx=(18, 14), pady=20)
        action_frame = ctk.CTkFrame(command_bar, fg_color="transparent")
        action_frame.grid(row=0, column=1, sticky="w")
        self.btn_open = ctk.CTkButton(
            action_frame, text="Seleziona radiografia", width=174, height=34, corner_radius=11,
            fg_color=ACCENT_BLUE, hover_color="#075B73", text_color="#FFFFFF",
            font=("Segoe UI", 11, "bold"), command=self.pick_file
        )
        self.btn_open.pack(side="left")

        tools_frame = ctk.CTkFrame(command_bar, fg_color="transparent")
        tools_frame.grid(row=0, column=2, padx=(10, 16), sticky="e")
        self.nav_hint = ctk.CTkLabel(
            tools_frame, text="Nessun esame aperto", width=154, anchor="e",
            text_color=TEXT_MUTED, font=("Segoe UI", 10)
        )
        self.nav_hint.pack(side="left", padx=(0, 10))
        self.btn_prev = ctk.CTkButton(
            tools_frame, text="‹", width=34, height=34, corner_radius=10,
            fg_color="#EEF3F7", hover_color="#DFE8F0", text_color=TEXT_MAIN,
            font=("Segoe UI", 20), state="disabled", command=self.previous_image
        )
        self.btn_prev.pack(side="left", padx=(0, 4))
        self.folder_counter = ctk.CTkLabel(
            tools_frame, text="— / —", width=72, height=34, corner_radius=10,
            fg_color="#F5F7FA", text_color=TEXT_SUB, font=("Segoe UI", 10, "bold")
        )
        self.folder_counter.pack(side="left", padx=(0, 4))
        self.btn_next = ctk.CTkButton(
            tools_frame, text="›", width=34, height=34, corner_radius=10,
            fg_color="#EEF3F7", hover_color="#DFE8F0", text_color=TEXT_MAIN,
            font=("Segoe UI", 20), state="disabled", command=self.next_image
        )
        self.btn_next.pack(side="left", padx=(0, 12))
        ctk.CTkButton(
            tools_frame, text="Metriche", width=100, height=34, corner_radius=11,
            fg_color="#E6F2F4", hover_color="#D6EBEE", text_color=ACCENT_BLUE,
            font=("Segoe UI", 11, "bold"), command=lambda: self.show_curve("FROC")
        ).pack(side="left")

        # ── Studio/header ─────────────────────────────────────────────────────
        hdr_frame = ctk.CTkFrame(
            self.main_area, fg_color="#FFFFFF", border_width=1,
            border_color=GLASS_BORDER, corner_radius=22, height=84
        )
        hdr_frame.grid(row=1, column=0, sticky="ew")
        hdr_frame.grid_columnconfigure(0, weight=1)
        hdr_frame.grid_columnconfigure(1, weight=0)
        hdr_frame.grid_propagate(False)
        self.hdr_title = ctk.CTkLabel(
            hdr_frame, text="Nessuna radiografia caricata", text_color=TEXT_MAIN,
            font=("Segoe UI", 22, "bold"), anchor="w"
        )
        self.hdr_title.grid(row=0, column=0, sticky="ew", padx=(22, 12), pady=(14, 0))
        self.hdr_sub = ctk.CTkLabel(
            hdr_frame, text="Seleziona una radiografia: le immagini della stessa cartella saranno disponibili per la revisione.",
            text_color=TEXT_MUTED, font=("Segoe UI", 11), anchor="w"
        )
        self.hdr_sub.grid(row=1, column=0, sticky="ew", padx=(22, 12), pady=(0, 14))
        chip_frame = ctk.CTkFrame(hdr_frame, fg_color="transparent")
        chip_frame.grid(row=0, column=1, rowspan=2, sticky="e", padx=(8, 20))
        self.model_chip = ctk.CTkLabel(
            chip_frame, text=f"MODEL · {self.model_info.get('name', 'N/D')}", width=132, height=32,
            fg_color="#EEF3F7", text_color=TEXT_SUB, corner_radius=16,
            font=("Segoe UI", 9, "bold")
        )
        self.model_chip.pack(side="left", padx=(0, 8))
        self.state_chip = ctk.CTkLabel(
            chip_frame, text="STANDBY", width=108, height=32, fg_color="#EEF3F7",
            text_color=TEXT_SUB, corner_radius=16, font=("Segoe UI", 9, "bold")
        )
        self.state_chip.pack(side="left")

        self.progress_bar = ctk.CTkProgressBar(
            self.main_area, height=3, fg_color="#E5ECF1", progress_color=ACCENT_BLUE
        )
        self.progress_bar.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.progress_bar.grid_remove()
        # ── Metriche globali del test set calcolate da evaluate ────────────────
        self.global_metrics_strip = ctk.CTkFrame(
            self.main_area,
            height=58,
            fg_color="#FFFFFF",
            border_width=1,
            border_color=GLASS_BORDER,
            corner_radius=18
        )
        self.global_metrics_strip.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self.global_metrics_strip.grid_columnconfigure(0, weight=1)
        self.global_metrics_strip.grid_propagate(False)

        ctk.CTkLabel(
            self.global_metrics_strip,
            text="TEST SET · NMS",
            text_color=TEXT_MAIN,
            font=("Segoe UI", 11, "bold"),
            anchor="w"
        ).grid(row=0, column=0, padx=(20, 12), pady=13, sticky="w")

        self.global_tp_label = ctk.CTkLabel(
            self.global_metrics_strip,
            text=f"TP {self.global_metrics['TP']}",
            width=74,
            height=31,
            fg_color="#DCFCE7",
            text_color=ACCENT_GREEN,
            corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.global_tp_label.grid(row=0, column=1, padx=4, pady=13)

        self.global_fp_label = ctk.CTkLabel(
            self.global_metrics_strip,
            text=f"FP {self.global_metrics['FP']}",
            width=74,
            height=31,
            fg_color="#FEF3C7",
            text_color=ACCENT_ORANGE,
            corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.global_fp_label.grid(row=0, column=2, padx=4, pady=13)

        self.global_fn_label = ctk.CTkLabel(
            self.global_metrics_strip,
            text=f"FN {self.global_metrics['FN']}",
            width=74,
            height=31,
            fg_color="#FEE2E2",
            text_color=ACCENT_RED,
            corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.global_fn_label.grid(row=0, column=3, padx=4, pady=13)

        self.global_precision_label = ctk.CTkLabel(
            self.global_metrics_strip,
            text=f"Precision {self.global_metrics['PRECISION_PERCENT']}%",
            width=140,
            height=31,
            fg_color="#DCECF1",
            text_color=ACCENT_BLUE,
            corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.global_precision_label.grid(row=0, column=4, padx=4, pady=13)

        self.global_recall_label = ctk.CTkLabel(
            self.global_metrics_strip,
            text=f"Recall {self.global_metrics['RECALL_PERCENT']}%",
            width=126,
            height=31,
            fg_color="#DCECF1",
            text_color=ACCENT_BLUE,
            corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.global_recall_label.grid(row=0, column=5, padx=(4, 16), pady=13)

        # ── Barra clinica di vista e di esito ────────────────────────────────
        # Un solo viewer centrale: il radiologo alterna i candidati prodotti
        # dall'image processing e il risultato finale filtrato dal modello.
        self.result_strip = ctk.CTkFrame(
            self.main_area, height=66, fg_color="#FFFFFF", border_width=1,
            border_color=GLASS_BORDER, corner_radius=18
        )
        self.result_strip.grid(row=4, column=0, sticky="ew", pady=(14, 14))
        self.result_strip.grid_columnconfigure(0, weight=1)
        self.result_title = ctk.CTkLabel(
            self.result_strip, text="RISULTATO FINALE · ML + NMS", text_color=TEXT_MAIN,
            font=("Segoe UI", 11, "bold"), anchor="w"
        )
        self.result_title.grid(row=0, column=0, padx=(20, 12), pady=16, sticky="w")

        self.view_selector = ctk.CTkFrame(
            self.result_strip, fg_color="#EEF3F7", corner_radius=12,
            width=296, height=40
        )
        self.view_selector.grid(row=0, column=1, padx=(8, 12), pady=12)
        self.view_selector.grid_propagate(False)
        self.btn_view_cpp = ctk.CTkButton(
            self.view_selector, text="IMAGE PROCESSING", width=158, height=34,
            corner_radius=10, fg_color="transparent", hover_color="#DDE8EF",
            text_color=TEXT_SUB, font=("Segoe UI", 9, "bold"),
            command=lambda: self.switch_tab("cpp")
        )
        self.btn_view_cpp.place(x=3, y=3)
        self.btn_view_ml = ctk.CTkButton(
            self.view_selector, text="RISULTATO ML", width=132, height=34,
            corner_radius=10, fg_color=ACCENT_BLUE, hover_color="#075B73",
            text_color="#FFFFFF", font=("Segoe UI", 9, "bold"),
            command=lambda: self.switch_tab("ml")
        )
        self.btn_view_ml.place(x=161, y=3)

        self.config_selector = ctk.CTkFrame(
            self.result_strip, fg_color="#EEF3F7", corner_radius=12,
            width=86, height=40
        )
        self.config_selector.grid(row=0, column=2, padx=(0, 12), pady=12)
        self.config_selector.grid_propagate(False)
        btn = ctk.CTkButton(
            self.config_selector, text="NMS", width=78, height=34, corner_radius=10,
            fg_color=ACCENT_BLUE, hover_color="#075B73", text_color="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            command=lambda: self.select_postprocess_config(NMS_TECHNIQUE)
        )
        btn.place(x=4, y=3)
        self.config_buttons[NMS_TECHNIQUE] = btn

        self.metric_detection = ctk.CTkLabel(
            self.result_strip, text="TP —  FP —  FN —", width=245, height=31,
            fg_color="#EEF3F7", text_color=TEXT_SUB, corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.metric_detection.grid(row=0, column=3, padx=4, pady=16)
        self.metric_precision = ctk.CTkLabel(
            self.result_strip, text="P —", width=90, height=31,
            fg_color="#EEF3F7", text_color=TEXT_SUB, corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.metric_precision.grid(row=0, column=4, padx=4, pady=16)
        self.metric_recall = ctk.CTkLabel(
            self.result_strip, text="R —", width=90, height=31,
            fg_color="#EEF3F7", text_color=TEXT_SUB, corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.metric_recall.grid(row=0, column=5, padx=4, pady=16)
        self.metric_f1 = ctk.CTkLabel(
            self.result_strip, text="F1 —", width=96, height=31,
            fg_color="#EEF3F7", text_color=TEXT_SUB, corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.metric_f1.grid(row=0, column=6, padx=4, pady=16)
        # F2 evidenziato in un chip colorato (viola/indaco): e' il criterio con cui
        # viene ottimizzato l'addestramento, quindi spicca rispetto agli altri.
        self.metric_f2 = ctk.CTkLabel(
            self.result_strip, text="F2 —", width=96, height=31,
            fg_color="#EDE9FE", text_color=ACCENT_INDIGO, corner_radius=11,
            font=("Segoe UI", 10, "bold")
        )
        self.metric_f2.grid(row=0, column=7, padx=(4, 16), pady=16)
        self.tabs_frame = self.result_strip
        self._sync_view_selector()

        # ── Viewer diagnostico ────────────────────────────────────────────────
        self.viewer_shell = ctk.CTkFrame(
            self.main_area, border_width=1, border_color="#CCD8E2",
            fg_color=BG_VIEWER, corner_radius=25
        )
        self.viewer_shell.grid(row=5, column=0, sticky="nsew", pady=(0, 4))
        self.viewer_shell.grid_rowconfigure(0, weight=1)
        self.viewer_shell.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.viewer_shell, bg=BG_VIEWER, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self._on_roi_feature_click)
        self.canvas.bind("<Enter>", self._on_viewer_enter)
        self.canvas.bind("<Leave>", self._on_viewer_leave)
        self.canvas.bind("<MouseWheel>", self._on_viewer_mousewheel)
        self.canvas.bind("<Button-4>", lambda _e: self.previous_image())
        self.canvas.bind("<Button-5>", lambda _e: self.next_image())
        self.viewer_shell.bind("<Enter>", self._on_viewer_enter)
        self.viewer_shell.bind("<Leave>", self._on_viewer_leave)
        self.tooltip.canvas = self.canvas

        # Pannello laterale visibile solo in IMAGE PROCESSING: abbina la ROI
        # selezionata all'immagine effettivamente usata dal C++ per le feature.
        self.viewer_shell.grid_columnconfigure(1, weight=0)
        self.feature_panel = ctk.CTkFrame(
            self.viewer_shell, width=328, fg_color="#0D1B2D",
            border_width=1, border_color="#243A52", corner_radius=18
        )
        self.feature_panel.grid(row=0, column=1, sticky="ns", padx=(0, 12), pady=12)
        self.feature_panel.grid_propagate(False)
        ctk.CTkLabel(
            self.feature_panel, text="FEATURE VIEW · IMAGE PROCESSING",
            text_color="#93C5D7", font=("Segoe UI", 9, "bold"), anchor="w"
        ).pack(fill="x", padx=18, pady=(16, 2))
        self.feature_roi_title = ctk.CTkLabel(
            self.feature_panel, text="Seleziona una ROI", text_color="#F8FAFC",
            font=("Segoe UI", 15, "bold"), anchor="w"
        )
        self.feature_roi_title.pack(fill="x", padx=18, pady=(0, 10))

        selector = ctk.CTkFrame(self.feature_panel, fg_color="transparent")
        selector.pack(fill="x", padx=18, pady=(0, 10))
        self.feature_prev_btn = ctk.CTkButton(
            selector, text="‹ ROI", width=76, height=30, fg_color="#142B41",
            hover_color="#193A55", text_color="#E2E8F0",
            command=lambda: self.select_feature_roi(self.feature_roi_index - 1)
        )
        self.feature_prev_btn.pack(side="left")
        self.feature_index_label = ctk.CTkLabel(
            selector, text="— / —", text_color="#A7BCCF", font=("Segoe UI", 10, "bold")
        )
        self.feature_index_label.pack(side="left", expand=True)
        self.feature_next_btn = ctk.CTkButton(
            selector, text="ROI ›", width=76, height=30, fg_color="#142B41",
            hover_color="#193A55", text_color="#E2E8F0",
            command=lambda: self.select_feature_roi(self.feature_roi_index + 1)
        )
        self.feature_next_btn.pack(side="right")

        previews = ctk.CTkFrame(self.feature_panel, fg_color="transparent")
        previews.pack(fill="x", padx=13)
        self.feature_preview_labels = {}
        for grid_row, grid_col, key, title in [
            (0, 0, "input", "ROI FEATURE"),
            (0, 1, "lbp", "LBP"),
            (1, 0, "glcm", "GLCM"),
            (1, 1, "grad", "GRAD / HOG"),
        ]:
            tile = ctk.CTkFrame(previews, width=146, height=145, fg_color="#102338", corner_radius=12)
            tile.grid(row=grid_row, column=grid_col, padx=4, pady=4)
            tile.grid_propagate(False)
            ctk.CTkLabel(
                tile, text=title, text_color="#91A8BD", font=("Segoe UI", 8, "bold")
            ).pack(pady=(7, 2))
            img_label = ctk.CTkLabel(tile, text="", width=108, height=108)
            img_label.pack(pady=(0, 7))
            self.feature_preview_labels[key] = img_label

        ctk.CTkLabel(
            self.feature_panel, text="VALORI ESTRATTI", text_color="#91A8BD",
            font=("Segoe UI", 8, "bold"), anchor="w"
        ).pack(fill="x", padx=18, pady=(12, 4))
        self.feature_stats = ctk.CTkTextbox(
            self.feature_panel, height=152, fg_color="#102338", border_width=0,
            text_color="#DAE7F1", font=("Consolas", 10), corner_radius=10
        )
        self.feature_stats.pack(fill="x", padx=14, pady=(0, 14))
        self.feature_stats.configure(state="disabled")
        self.feature_panel.grid_remove()
        self._draw_empty_placeholder()

    # ── Dropzone interaction ──────────────────────────────────────────────────
    def _on_dz_enter(self, event):
        self.hovering_dz = True
        self.dropzone.configure(border_color=ACCENT_BLUE_L)
        self._dz_canvas.configure(bg=GLASS_PANEL)
        self._draw_dz_dashes()
        self.dz_icon.configure(fg=ACCENT_CYAN)

    def _on_dz_leave(self, event):
        self.hovering_dz = False
        self.dropzone.configure(border_color=GLASS_BORDER)
        self._draw_dz_dashes()

    def _on_dz_press(self, event):
        """Strong press effect: shrink + darken + heavy border glow."""
        press_color = GLASS_SOFT
        self.dropzone.configure(
            fg_color=press_color,
            border_color=ACCENT_BLUE,
            border_width=2
        )
        self._dz_canvas.configure(bg=press_color)
        for lbl in [self.dz_icon, self.dz_main, self.dz_sub]:
            lbl.configure(bg=press_color)
        # Scale-down illusion: shift icon/text slightly inward
        self.dz_icon.place(relx=0.5, rely=0.30, anchor="center")
        self.dz_main.place(relx=0.5, rely=0.48, anchor="center")
        self.dz_sub.place(relx=0.5,  rely=0.59, anchor="center")

    def _on_dz_release(self, event):
        """Restore normal state after press."""
        self.dropzone.configure(fg_color=GLASS_PANEL, border_width=1)
        self._dz_canvas.configure(bg=GLASS_PANEL)
        for lbl in [self.dz_icon, self.dz_main, self.dz_sub]:
            lbl.configure(bg=GLASS_PANEL)
        # Restore positions
        self.dz_icon.place(relx=0.5, rely=0.29, anchor="center")
        self.dz_main.place(relx=0.5, rely=0.47, anchor="center")
        self.dz_sub.place(relx=0.5,  rely=0.58, anchor="center")
        if self.hovering_dz:
            self.dropzone.configure(border_color=ACCENT_BLUE_L)
        else:
            self.dropzone.configure(border_color=GLASS_BORDER)
        self.pick_file()

    # ── Viewer interaction ────────────────────────────────────────────────────
    def _on_viewer_enter(self, event):
        self.hovering_viewer = True
        self.viewer_shell.configure(border_color=ACCENT_CYAN)

    def _on_viewer_leave(self, event):
        self.hovering_viewer = False

    def _on_viewer_mousewheel(self, event):
        if getattr(event, "delta", 0) > 0:
            self.previous_image()
        elif getattr(event, "delta", 0) < 0:
            self.next_image()
        return "break"

    # ── Feature view interaction ──────────────────────────────────────────────
    def _set_feature_panel_visible(self):
        visible = self.active_tab == "cpp" and bool(self.rois)
        if visible:
            self.feature_panel.grid()
            self._update_feature_panel()
        else:
            self.feature_panel.grid_remove()

    def _on_roi_feature_click(self, event):
        """In IMAGE PROCESSING un click sul box apre le feature della stessa ROI."""
        if self.active_tab != "cpp" or not self.rois or not self.base_image:
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.base_image.size
        scale = min(w / iw, h / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox, oy = (w - dw) // 2, (h - dh) // 2
        for idx in range(len(self.rois) - 1, -1, -1):
            roi = self.rois[idx]
            rx = ox + float(roi.get("x", 0.0)) * scale
            ry = oy + float(roi.get("y", 0.0)) * scale
            rw = float(roi.get("width", 0.0)) * scale
            rh = float(roi.get("height", 0.0)) * scale
            if rx <= event.x <= rx + rw and ry <= event.y <= ry + rh:
                self.select_feature_roi(idx)
                return "break"

    def select_feature_roi(self, index):
        if not self.rois:
            return
        self.feature_roi_index = int(index) % len(self.rois)
        self._update_feature_panel()
        self.redraw()

    @staticmethod
    def _safe_feature_value(roi, name):
        try:
            value = float(roi.get(name, float("nan")))
            return value if np.isfinite(value) else None
        except Exception:
            return None

    def _update_feature_panel(self):
        if self.active_tab != "cpp" or not self.rois:
            return
        self.feature_roi_index %= len(self.rois)
        roi = self.rois[self.feature_roi_index]
        roi_id = roi.get("roi_id", self.feature_roi_index + 1)
        method = str(roi.get("method", "N/D"))
        self.feature_roi_title.configure(text=f"ROI {roi_id}  ·  {method}")
        self.feature_index_label.configure(text=f"{self.feature_roi_index + 1} / {len(self.rois)}")
        self.feature_prev_btn.configure(state="normal" if len(self.rois) > 1 else "disabled")
        self.feature_next_btn.configure(state="normal" if len(self.rois) > 1 else "disabled")

        feature_input = _feature_input_crop(roi, self.base_image)
        previews = {
            "input": feature_input.convert("RGB"),
            "lbp": _lbp_feature_preview(feature_input),
            "glcm": _glcm_feature_preview(feature_input),
            "grad": _gradient_feature_preview(feature_input),
        }
        self.feature_tk_images = {}
        for key, image in previews.items():
            thumb = image.resize((108, 108), Image.Resampling.NEAREST if key in ("lbp", "glcm") else Image.Resampling.LANCZOS)
            tk_img = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=(108, 108))
            self.feature_tk_images[key] = tk_img
            self.feature_preview_labels[key].configure(image=tk_img, text="")

        shown = [
            ("score", "Score C++"),
            ("glcm_contrast", "GLCM contrast"),
            ("glcm_homogeneity", "GLCM homog."),
            ("glcm_energy", "GLCM energy"),
            ("glcm_entropy", "GLCM entropy"),
            ("glcm_correlation", "GLCM correl."),
        ]
        lines = []
        for key, label in shown:
            value = self._safe_feature_value(roi, key)
            if value is not None:
                lines.append(f"{label:<16} {value:>9.4f}")
        lbp_values = [
            self._safe_feature_value(roi, f"lbp_{idx}") for idx in range(256)
            if f"lbp_{idx}" in roi
        ]
        lbp_values = [value for value in lbp_values if value is not None]
        if lbp_values:
            peak_bin = int(np.argmax(np.asarray(lbp_values, dtype=float)))
            lines.append(f"{'LBP bin picco':<16} {peak_bin:>9d}")
            lines.append(f"{'LBP valore picco':<16} {max(lbp_values):>9.4f}")

        hog_values = [
            self._safe_feature_value(roi, key) for key in roi
            if str(key).startswith("hog_")
        ]
        hog_values = [value for value in hog_values if value is not None]
        if hog_values:
            hv = np.asarray(hog_values, dtype=float)
            lines.append(f"{'HOG descrittori':<16} {len(hog_values):>9d}")
            lines.append(f"{'HOG max':<16} {float(hv.max()):>9.4f}")

        used = set(self.feature_columns or [])
        used_texture = [label for key, label in shown if key in used]
        if used_texture:
            lines.append("")
            lines.append("Usate dal modello: " + ", ".join(used_texture))
        if "_feature_image_png" not in roi:
            lines.append("")
            lines.append("Anteprima ROI ricostruita dal box.")
        self.feature_stats.configure(state="normal")
        self.feature_stats.delete("1.0", "end")
        self.feature_stats.insert("1.0", "\n".join(lines) if lines else "Feature numeriche non disponibili.")
        self.feature_stats.configure(state="disabled")

    # ── Button hover effects ──────────────────────────────────────────────────
    def _on_btn_new_enter(self, event):
        if self.btn_new.cget("state") == "normal":
            self.btn_new.configure(border_color=ACCENT_BLUE, text_color=ACCENT_BLUE)

    def _on_btn_new_leave(self, event):
        self.btn_new.configure(border_color=GLASS_BORDER, text_color=TEXT_SUB)

    def _on_btn_save_enter(self, event):
        if self.btn_save.cget("state") == "normal":
            self.btn_save.configure(border_color=ACCENT_BLUE, text_color=ACCENT_BLUE)

    def _on_btn_save_leave(self, event):
        self.btn_save.configure(border_color=GLASS_BORDER, text_color=TEXT_SUB)

    def _on_btn_run_enter(self, event):
        self.hovering_run = True
        if self.btn_run.cget("state") == "normal":
            self.btn_run.configure(fg_color="#007AFF", border_color=ACCENT_BLUE_L)

    def _on_btn_run_leave(self, event):
        self.hovering_run = False
        self.btn_run.configure(fg_color=ACCENT_BLUE, border_color="#4DA3FF")

    # ── Placeholder & Grid ────────────────────────────────────────────────────
    def _draw_tech_grid(self, w, h):
        gs = 52
        for x in range(0, w, gs):
            major = (x // gs) % 4 == 0
            self.canvas.create_line(x, 0, x, h, fill="#202A38" if major else "#171F2C", width=1)
        for y in range(0, h, gs):
            major = (y // gs) % 4 == 0
            self.canvas.create_line(0, y, w, y, fill="#202A38" if major else "#171F2C", width=1)

    def _draw_ambient_lights(self, w, h):
        self.canvas.create_rectangle(0, 0, w, 1, fill="#3A3A3C", outline="")
        self.canvas.create_rectangle(0, h - 1, w, h, fill="#1C1C1E", outline="")
        self.canvas.create_rectangle(18, 18, w - 18, h - 18, outline="#2C2C2E", width=1)

    def _draw_viewer_hud(self, w, h, status_text="STANDBY"):
        accent = ACCENT_BLUE_L
        dim = "#3A3A3C"
        panel = "#161618"

        panel_w = min(300, max(220, w - 140))
        x0 = (w - panel_w) // 2
        y0 = 18
        self.canvas.create_rectangle(x0, y0, x0 + panel_w, y0 + 32, fill=panel, outline=dim, width=1)
        self.canvas.create_text(
            x0 + 16, y0 + 16, text="DIAGNOSTIC VIEW",
            fill="#EEF5FF", font=("SF Pro Text", 10, "bold"), anchor="w"
        )
        self.canvas.create_text(
            x0 + panel_w - 16, y0 + 16, text=status_text,
            fill=accent, font=("SF Pro Text", 9, "bold"), anchor="e"
        )

        cap = 28
        for bx, by, sx, sy in [
            (24, 24, 1, 1), (w - 24, 24, -1, 1),
            (24, h - 24, 1, -1), (w - 24, h - 24, -1, -1),
        ]:
            self.canvas.create_line(bx, by, bx + sx * cap, by, fill=dim, width=1)
            self.canvas.create_line(bx, by, bx, by + sy * cap, fill=dim, width=1)

        if w > 560 and h > 260:
            total = len(self.rois) if hasattr(self, "rois") else 0
            confirmed = len([r for r in self.rois if int(r.get("ml_pred", 0)) == 1]) if total else 0
            gt_count = len(self.gt_boxes) if hasattr(self, "gt_boxes") else 0
            card_w, card_h = 200, 68
            cx0, cy0 = w - card_w - 28, h - card_h - 34
            self.canvas.create_rectangle(cx0, cy0, cx0 + card_w, cy0 + card_h, fill="#0B1728", outline=dim, width=1)
            self.canvas.create_text(cx0 + 14, cy0 + 16, text="Sessione", fill=TEXT_MUTED, font=("SF Pro Text", 9, "bold"), anchor="w")
            self.canvas.create_text(cx0 + card_w - 14, cy0 + 16, text=status_text, fill=accent, font=("SF Pro Text", 9, "bold"), anchor="e")
            self.canvas.create_text(cx0 + 14, cy0 + 36, text=f"ROI {total:03d}", fill="#E0F2FE", font=("SF Pro Text", 10, "bold"), anchor="w")
            self.canvas.create_text(cx0 + card_w - 14, cy0 + 36, text=f"ML {confirmed:03d}", fill=ACCENT_GREEN, font=("SF Pro Text", 10, "bold"), anchor="e")
            self.canvas.create_text(cx0 + 14, cy0 + 56, text=f"GT {gt_count:03d}", fill=ACCENT_PURPLE, font=("SF Pro Text", 10, "bold"), anchor="w")

    def _draw_empty_placeholder(self):
        """Draw the empty viewer state: only the clinical grid."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50: w = 780
        if h < 50: h = 480

        self._draw_tech_grid(w, h)
        self.canvas.create_rectangle(18, 18, w - 18, h - 18, outline="#2C2C2E", width=1)

    # ── File I/O & continuous series browsing ─────────────────────────────────
    @staticmethod
    def _natural_sort_key(path):
        """Ordina ad esempio RX_2 prima di RX_10."""
        return [int(part) if part.isdigit() else part.casefold()
                for part in re.split(r"(\d+)", str(path))]

    def _collect_series(self, root, recursive=False):
        iterator = root.rglob("*") if recursive else root.iterdir()
        images = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(images, key=lambda p: self._natural_sort_key(p.relative_to(root)))

    def _update_browser_controls(self):
        total = len(self.image_files)
        current = self.image_index + 1 if total and self.image_index >= 0 else 0
        self.folder_counter.configure(text=f"{current} / {total}" if total else "— / —")
        base_index = self.pending_image_index if self.pending_image_index is not None else self.image_index
        enabled = total > 1
        self.btn_prev.configure(state="normal" if enabled and base_index > 0 else "disabled")
        self.btn_next.configure(state="normal" if enabled and base_index < total - 1 else "disabled")
        if self.image_folder and total:
            folder_name = self.image_folder.name or str(self.image_folder)
            if len(folder_name) > 28:
                folder_name = folder_name[:25] + "…"
            self.nav_hint.configure(text=f"{folder_name}")
        else:
            self.nav_hint.configure(text="Nessun esame aperto")

    def pick_file(self):
        """Apre un file e crea comunque una serie usando le immagini sorelle."""
        if self.is_running:
            return
        self.dz_main.configure(text="Selezione in corso…")
        self.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            parent=self,
            title="Seleziona radiografia",
            filetypes=[("Immagini cliniche", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("Tutti", "*.*")]
        )
        self.attributes("-topmost", False)
        self.lift()
        self.dz_main.configure(text="Seleziona radiografia")
        if not path:
            return
        selected = Path(path)
        self.image_folder = selected.parent
        self.image_files = self._collect_series(selected.parent, recursive=False)
        if selected not in self.image_files:
            self.image_files.append(selected)
            self.image_files.sort(key=lambda p: self._natural_sort_key(p.name))
        self.image_index = self.image_files.index(selected)
        self.pending_image_index = None
        self._update_browser_controls()
        self.load_image(selected, auto_analyze=True)


    def _select_series_image(self, new_index):
        if not self.image_files:
            return
        new_index = max(0, min(int(new_index), len(self.image_files) - 1))
        if self.is_running:
            self.pending_image_index = new_index
            self.nav_hint.configure(text=f"In coda · {new_index + 1}/{len(self.image_files)}")
            self._update_browser_controls()
            return
        if new_index == self.image_index and self.img_path == self.image_files[new_index]:
            return
        self.image_index = new_index
        self.pending_image_index = None
        self._update_browser_controls()
        self.load_image(self.image_files[self.image_index], auto_analyze=True)

    def previous_image(self):
        base = self.pending_image_index if self.pending_image_index is not None else self.image_index
        self._select_series_image(base - 1)

    def next_image(self):
        base = self.pending_image_index if self.pending_image_index is not None else self.image_index
        self._select_series_image(base + 1)

    def load_image(self, path, auto_analyze=True):
        self.img_path = Path(path)
        self.base_image = load_16bit_image_as_rgb(self.img_path)
        self.rois = []
        self.gt_boxes = []
        self.nms_rois = []
        self.post_metrics = {}
        self.feature_columns = []
        self.feature_roi_index = 0
        self.feature_tk_images = {}
        if hasattr(self, "feature_panel"):
            self.feature_panel.grid_remove()
        self.tooltip.hide()

        name = self.img_path.name
        short_name = name if len(name) <= 30 else name[:16] + "…" + name[-11:]
        self.dz_icon.configure(text="✓", fg=ACCENT_GREEN)
        self.dz_main.configure(text="Radiografia caricata", fg=TEXT_MAIN)
        self.dz_sub.configure(text=short_name, fg=ACCENT_BLUE)

        self.hdr_title.configure(text=name)
        total = len(self.image_files)
        current = self.image_index + 1 if total else 1
        self.hdr_sub.configure(
            text=f"Immagine {current} di {max(total, 1)} · analisi automatica · usa ‹ › per la navigazione."
        )
        self.metric_detection.configure(text="TP —  FP —  FN —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self.metric_precision.configure(text="P —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self.metric_recall.configure(text="R —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self.metric_f1.configure(text="F1 —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self.metric_f2.configure(text="F2 —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self._sync_view_selector()
        self.pipe_led.set_state(True)
        self.pipe_lbl.configure(text="Pronta per analisi", fg=TEXT_SIDEBAR)
        self.state_chip.configure(text="AUTO READY", fg_color="#DCECF1", text_color=ACCENT_BLUE)

        self.btn_new.configure(state="normal")
        self.btn_save.configure(state="disabled")
        self.btn_run.configure(state="normal")
        self._update_browser_controls()
        self.redraw()
        if auto_analyze and self.auto_analysis:
            self.after(40, self.run_analysis)

    def reset(self):
        self.img_path   = None
        self.base_image = None
        self.tk_image   = None
        self.rois       = []
        self.gt_boxes   = []
        self.nms_rois = []
        self.post_metrics = {}
        self.feature_columns = []
        self.feature_roi_index = 0
        self.feature_tk_images = {}
        self.active_tab = "ml"
        if hasattr(self, "feature_panel"):
            self.feature_panel.grid_remove()
        self.image_folder = None
        self.image_files = []
        self.image_index = -1
        self.pending_image_index = None
        self.tooltip.hide()
        self._update_browser_controls()

        # Reset dropzone
        self.dz_icon.configure(text="＋", fg=ACCENT_BLUE)
        self.dz_main.configure(text="Apri radiografia", fg=TEXT_MAIN)
        self.dz_sub.configure(text="Analisi automatica all'apertura", fg=TEXT_MUTED)

        # Reset header
        self.hdr_title.configure(text="Nessuna radiografia caricata")
        self.hdr_sub.configure(text="Seleziona una radiografia per avviare l'analisi e navigare le immagini della cartella.")

        # Reset indicatore analisi
        self.pipe_led.set_state(False)
        self.pipe_lbl.configure(text="In attesa", fg="#94A3B8")
        self.state_chip.configure(text="STANDBY", fg_color=GLASS_SOFT, text_color=TEXT_SUB)

        self.btn_new.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.btn_run.configure(state="disabled")
        self.metric_detection.configure(text="TP —  FP —  FN —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self.metric_precision.configure(text="P —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self.metric_recall.configure(text="R —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self.metric_f1.configure(text="F1 —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self.metric_f2.configure(text="F2 —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        self._sync_view_selector()
        self._draw_empty_placeholder()

    def _sync_global_metrics_strip(self):
        """Aggiorna la barra TEST SET con i valori evaluate NMS."""
        if not hasattr(self, "global_tp_label"):
            return
        technique = _canonical_technique_name(getattr(self, "selected_postprocess_technique", DEFAULT_POSTPROCESS_TECHNIQUE))
        row = _evaluate_best_row_for_technique(technique)
        if row:
            try:
                self.global_tp_label.configure(text=f"TP {_format_int(row.get('TP'))}")
                self.global_fp_label.configure(text=f"FP {_format_int(row.get('FP'))}")
                self.global_fn_label.configure(text=f"FN {_format_int(row.get('FN'))}")
                self.global_precision_label.configure(text=f"Precision {_format_percent(row.get('precision'), 1)}")
                self.global_recall_label.configure(text=f"Recall {_format_percent(row.get('recall'), 1)}")
            except Exception:
                pass
            return
        try:
            self.global_tp_label.configure(text=f"TP {self.global_metrics['TP']}")
            self.global_fp_label.configure(text=f"FP {self.global_metrics['FP']}")
            self.global_fn_label.configure(text=f"FN {self.global_metrics['FN']}")
            self.global_precision_label.configure(text=f"Precision {self.global_metrics['PRECISION_PERCENT']}%")
            self.global_recall_label.configure(text=f"Recall {self.global_metrics['RECALL_PERCENT']}%")
        except Exception:
            pass


    def _sync_view_selector(self):
        """Aggiorna selettore e metriche in base alla vista richiesta."""
        if not hasattr(self, "btn_view_cpp"):
            return
        is_cpp = self.active_tab == "cpp"
        self._sync_config_selector()
        self._sync_global_metrics_strip()
        self.btn_view_cpp.configure(
            fg_color=ACCENT_BLUE if is_cpp else "transparent",
            hover_color="#075B73" if is_cpp else "#DDE8EF",
            text_color="#FFFFFF" if is_cpp else TEXT_SUB,
        )
        self.btn_view_ml.configure(
            fg_color="transparent" if is_cpp else ACCENT_BLUE,
            hover_color="#DDE8EF" if is_cpp else "#075B73",
            text_color=TEXT_SUB if is_cpp else "#FFFFFF",
        )
        if is_cpp:
            self.result_title.configure(text="CSV FEATURE · TUTTE LE ZONE CANDIDATE")
            candidate_count = len(self.rois)
            self.metric_detection.configure(
                text=f"Candidati {candidate_count}" if self.rois else "Candidati —",
                text_color=CANDIDATE_CLR if self.rois else TEXT_SUB,
                fg_color="#DDF4F6" if self.rois else GLASS_SOFT,
            )
            self.metric_precision.configure(text="PRE-FILTRO", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
            self.metric_recall.configure(text="ROI CSV", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
            self.metric_f1.configure(text="F1 —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
            self.metric_f2.configure(text="F2 —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
        else:
            active_tech = getattr(self, 'selected_postprocess_technique', DEFAULT_POSTPROCESS_TECHNIQUE)
            self.result_title.configure(text=f"RISULTATO FINALE · {TECHNIQUE_LABELS.get(active_tech, 'ML + NMS')} · TP {TP_IOU_RULE_TEXT}")
            if self.post_metrics:
                n_nms = len(self.nms_rois)
                prec = float(self.post_metrics.get("precision", 0.0)) * 100.0
                rec = float(self.post_metrics.get("recall", 0.0)) * 100.0
                tp = int(self.post_metrics.get("TP", 0))
                fp = int(self.post_metrics.get("FP", 0))
                fn = int(self.post_metrics.get("FN", 0))
                f1 = float(self.post_metrics.get("F1", 0.0))
                self.metric_detection.configure(text=f"TP {tp}  FP {fp}  FN {fn}", text_color=ACCENT_BLUE, fg_color="#DCECF1")
                self.metric_precision.configure(text=f"P {prec:.1f}%", text_color=ACCENT_GREEN, fg_color="#DCFCE7")
                self.metric_recall.configure(text=f"R {rec:.1f}%", text_color=ACCENT_GREEN, fg_color="#DCFCE7")
                self.metric_f1.configure(text=f"F1 {f1:.3f}", text_color=ACCENT_GREEN, fg_color="#DCFCE7")
                f2 = float(self.post_metrics.get("F2", 0.0))
                self.metric_f2.configure(text=f"F2 {f2:.3f}", text_color=ACCENT_INDIGO, fg_color="#EDE9FE")
            else:
                self.metric_detection.configure(text="TP —  FP —  FN —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
                self.metric_precision.configure(text="P —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
                self.metric_recall.configure(text="R —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
                self.metric_f1.configure(text="F1 —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)
                self.metric_f2.configure(text="F2 —", text_color=TEXT_SUB, fg_color=GLASS_SOFT)

    def _sync_config_selector(self):
        """Aggiorna il pulsante NMS attivo."""
        selected = _canonical_technique_name(getattr(self, "selected_postprocess_technique", DEFAULT_POSTPROCESS_TECHNIQUE))
        for technique, btn in getattr(self, "config_buttons", {}).items():
            active = technique == selected
            try:
                btn.configure(
                    fg_color=ACCENT_BLUE if active else "transparent",
                    hover_color="#075B73" if active else "#DDE8EF",
                    text_color="#FFFFFF" if active else TEXT_SUB,
                    border_width=0,
                )
            except Exception:
                pass

    def _current_image_metric_line(self, technique=None):
        technique = _canonical_technique_name(technique or getattr(self, "selected_postprocess_technique", DEFAULT_POSTPROCESS_TECHNIQUE))
        metrics = getattr(self, "postprocess_metrics", {}).get(technique) or (self.post_metrics if technique == getattr(self, "selected_postprocess_technique", None) else {})
        if not metrics:
            return "Immagine corrente: nessun risultato disponibile, esegui prima l'analisi."
        return _metric_line_from_row("Immagine corrente", {
            "detections": metrics.get("detections", 0),
            "TP": metrics.get("TP", 0),
            "FP": metrics.get("FP", 0),
            "IGNORED": metrics.get("IGNORED", metrics.get("ignored", 0)),
            "FN": metrics.get("FN", 0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "F1": metrics.get("F1", 0.0),
            "F2": metrics.get("F2", 0.0),
            "threshold": metrics.get("ml_min_score", ML_MIN_SCORE),
        })

    def _evaluate_metric_line_for_technique(self, technique):
        technique = _canonical_technique_name(technique)
        self.evaluate_best_rows_by_technique = _load_evaluate_best_rows_by_technique()
        row = self.evaluate_best_rows_by_technique.get(technique)
        if row:
            metric_name = row.get("_metric_key", "F1")
            return _metric_line_from_row(f"Test set evaluate · best {metric_name} NMS", row)
        best = _load_best_postprocess_params() or _best_postprocess_params or {}
        if best and _canonical_technique_name(best.get("technique")) == technique:
            return _metric_line_from_row("Test set evaluate · NMS", best)
        return "Test set evaluate NMS non disponibile. Esegui evaluate_test_postprocess_class3_nms_only.py e verifica curve_points_nms_only.csv."

    def select_postprocess_config(self, technique, open_window=False):
        """Aggiorna viewer, overlay e metriche NMS senza aprire finestre."""
        technique = _canonical_technique_name(technique)
        if not self.postprocess_results:
            if open_window:
                messagebox.showinfo(
                    "NMS non disponibile",
                    "Carica e analizza una radiografia prima di aprire i risultati NMS."
                )
            return
        # Ricalcolo da zero NMS usando le ROI grezze
        # dell'immagine corrente e la soglia operativa fissa. Non uso
        # metriche cached/stale: i valori TP/FP/FN mostrati sono sempre quelli
        # delle box attualmente disegnate.
        technique_threshold = float((self.postprocess_thresholds or {}).get(technique, _threshold_for_technique(technique, ML_MIN_SCORE)))
        dets_raw = postprocess_rois_by_technique(
            self.rois,
            technique=technique,
            min_ml_score=technique_threshold,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
        )
        if dets_raw is None:
            dets_raw = self.postprocess_results.get(technique, [])
        self.selected_postprocess_technique = technique
        self.nms_rois, evaluated_gt_boxes, metrics = _with_fresh_detection_labels(dets_raw, self.gt_boxes)
        # Copia lo stato matched delle GT valutate dentro self.gt_boxes, così anche gli FN sono corretti sul viewer.
        for idx, gt in enumerate(self.gt_boxes):
            try:
                gt["matched"] = bool(evaluated_gt_boxes[idx].get("matched", False)) if idx < len(evaluated_gt_boxes) else False
            except Exception:
                pass
        stored = dict(metrics)
        stored["selected_technique"] = technique
        stored["technique"] = technique
        stored["technique_label"] = TECHNIQUE_LABELS.get(technique, technique)
        stored["ml_min_score"] = float(technique_threshold)
        stored["threshold"] = float(technique_threshold)
        self.post_metrics = stored
        self.postprocess_results[technique] = [dict(d) for d in self.nms_rois]
        self.postprocess_metrics[technique] = dict(stored)
        self.postprocess_thresholds[technique] = float(technique_threshold)
        self.active_tab = "ml"
        self._sync_view_selector()
        self._set_feature_panel_visible()
        self._sync_config_selector()
        tp = int(self.post_metrics.get("TP", 0))
        fp = int(self.post_metrics.get("FP", 0))
        fn = int(self.post_metrics.get("FN", 0))
        self.hdr_sub.configure(
            text=(
                f"NMS attivo: "
                f"{TECHNIQUE_LABELS.get(technique, technique)} · ML>={float(self.post_metrics.get('ml_min_score', ML_MIN_SCORE)):.3f} · TP {tp}  FP {fp}  FN {fn} · criterio TP: {TP_IOU_RULE_TEXT}."
            )
        )
        self.redraw()
        if open_window:
            self.show_configuration_window(technique)

    def show_configuration_window(self, technique):
        """Apre una nuova interfaccia con i risultati NMS."""
        technique = _canonical_technique_name(technique)
        idx = _technique_index_label(technique)
        label = TECHNIQUE_LABELS.get(technique, technique)
        win = ctk.CTkToplevel(self)
        win.withdraw()
        win.title(f"NMS · {label}")
        win.configure(fg_color=BG_MAIN)
        win.minsize(760, 520)
        win.transient(self)
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(2, weight=1)

        titlebar = ctk.CTkFrame(win, height=82, fg_color="#FFFFFF", corner_radius=0)
        titlebar.grid(row=0, column=0, sticky="ew")
        titlebar.grid_columnconfigure(0, weight=1)
        titlebar.grid_propagate(False)
        ctk.CTkLabel(
            titlebar, text=f"NMS", text_color=TEXT_MUTED,
            font=("Segoe UI", 10, "bold"), anchor="w"
        ).grid(row=0, column=0, padx=(26, 10), pady=(14, 0), sticky="w")
        ctk.CTkLabel(
            titlebar, text=label, text_color=TEXT_MAIN,
            font=("Segoe UI", 20, "bold"), anchor="w"
        ).grid(row=1, column=0, padx=(26, 10), pady=(0, 14), sticky="w")
        ctk.CTkButton(
            titlebar, text="Chiudi", width=82, height=34, corner_radius=12,
            fg_color="#EEF3F7", hover_color="#E3EAF0", text_color=TEXT_SUB,
            font=("Segoe UI", 11, "bold"), command=win.destroy
        ).grid(row=0, column=1, rowspan=2, padx=(0, 24), pady=22)

        params_card = ctk.CTkFrame(
            win, fg_color="#FFFFFF", corner_radius=20,
            border_width=1, border_color=GLASS_BORDER
        )
        params_card.grid(row=1, column=0, sticky="ew", padx=22, pady=(18, 10))
        params_card.grid_columnconfigure(0, weight=1)
        best_data = _load_best_postprocess_params() or _best_postprocess_params or {}
        best_tech = _canonical_technique_name(best_data.get("technique", DEFAULT_POSTPROCESS_TECHNIQUE)) if best_data else DEFAULT_POSTPROCESS_TECHNIQUE
        best_note = "TEST SET NMS" if technique == best_tech else f"Test set evaluate NMS: {TECHNIQUE_LABELS.get(best_tech, best_tech)}"
        params_text = (
            f"{best_note}\n"
            f"Parametri operativi: ML>={float(ML_MIN_SCORE):.3f} · "
            f"NMS IoU>={float(NMS_IOU_THRESHOLD):.3f} · "
            f"TP {TP_IOU_RULE_TEXT}\n"
            f"{_best_postprocess_params_summary()}"
        )
        ctk.CTkLabel(
            params_card, text=params_text, text_color=TEXT_SUB,
            font=("Segoe UI", 11), justify="left", anchor="w"
        ).grid(row=0, column=0, padx=20, pady=16, sticky="ew")

        body = ctk.CTkFrame(
            win, fg_color="#FFFFFF", corner_radius=20,
            border_width=1, border_color=GLASS_BORDER
        )
        body.grid(row=2, column=0, sticky="nsew", padx=22, pady=(0, 22))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(2, weight=1)

        image_line = self._current_image_metric_line(technique)
        eval_line = self._evaluate_metric_line_for_technique(technique)
        lines = [
            image_line,
            eval_line,
            "",
            "Legenda: DET=rilevamenti disegnati · TP=vera frattura con IoU>0.50 · FP=falso positivo · FN=frattura mancata.",
            "Scorciatoia: 2=NMS. Il viewer principale mostra solo il risultato NMS."
        ]
        ctk.CTkLabel(
            body, text="RISULTATI", text_color=TEXT_MAIN,
            font=("Segoe UI", 14, "bold"), anchor="w"
        ).grid(row=0, column=0, padx=20, pady=(18, 6), sticky="ew")
        ctk.CTkLabel(
            body, text="\n".join(lines), text_color=TEXT_SUB,
            font=("Segoe UI", 12), justify="left", anchor="nw", wraplength=860
        ).grid(row=1, column=0, padx=20, pady=(0, 18), sticky="ew")

        comparison = [
            f"NMS\n   {self._current_image_metric_line(NMS_TECHNIQUE)}\n   {self._evaluate_metric_line_for_technique(NMS_TECHNIQUE)}"
        ]
        text_box = ctk.CTkTextbox(
            body, fg_color="#F8FAFC", text_color=TEXT_MAIN,
            border_width=1, border_color=GLASS_BORDER, corner_radius=14,
            font=("Consolas", 10), wrap="word"
        )
        text_box.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        text_box.insert("1.0", "\n\n".join(comparison))
        text_box.configure(state="disabled")

        self._center_dialog(win, 960, 650)
        win.attributes("-topmost", True)
        win.deiconify()
        win.lift()
        win.focus_force()
        win.after(260, lambda: win.winfo_exists() and win.attributes("-topmost", False))

    def switch_tab(self, tab="ml"):
        """Alterna il viewer unico tra ROI candidate da CSV e risultato ML/NMS."""
        self.active_tab = "cpp" if tab == "cpp" else "ml"
        self._sync_view_selector()
        self._set_feature_panel_visible()
        if self.img_path and self.rois:
            label = "zone candidate da CSV" if self.active_tab == "cpp" else f"risultato {TECHNIQUE_LABELS.get(getattr(self, 'selected_postprocess_technique', DEFAULT_POSTPROCESS_TECHNIQUE), 'post-processing ML')}"
            tp_note = "" if self.active_tab == "cpp" else f" · TP {TP_IOU_RULE_TEXT}"
            self.hdr_sub.configure(text=f"Vista attiva: {label}{tp_note} · usa I/M per alternare le viste e ‹ › per navigare.")
        self.redraw()

    def run_analysis(self):
        if not self.img_path or self.is_running:
            return
        self.is_running = True
        self._analysis_token += 1
        token = self._analysis_token
        analysis_path = Path(self.img_path)
        self._update_browser_controls()
        self.btn_run.configure(text="Analisi NMS in corso…", state="disabled")
        self.btn_new.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.hdr_sub.configure(text=f"Analisi ML/NMS da CSV per {analysis_path.name}…")
        self.pipe_led.set_state(True)
        self.pipe_lbl.configure(text="Analisi NMS da CSV", fg="#BAE6FD")
        self.state_chip.configure(text="ANALYZING", fg_color="#FEF3C7", text_color=ACCENT_ORANGE)
        self.progress_bar.grid()
        self.progress_bar.start()
        threading.Thread(target=self._worker, args=(token, analysis_path), daemon=True).start()

    def _worker(self, token, analysis_path):
        """Non tocca Tkinter: deposita solo il risultato in una coda thread-safe."""
        try:
            data = run_pipeline_inference(analysis_path)
            self._analysis_events.put((token, analysis_path, data, None))
        except Exception as error:
            self._analysis_events.put((token, analysis_path, None, error))

    def _poll_analysis_events(self):
        try:
            while True:
                token, analysis_path, data, error = self._analysis_events.get_nowait()
                if token == self._analysis_token:
                    self._analysis_done(data, error, analysis_path)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(70, self._poll_analysis_events)

    def _analysis_done(self, data, error, analysis_path=None):
        self.is_running = False
        self.btn_run.configure(text="Rianalizza ora", state="normal")
        self.btn_new.configure(state="normal")
        self.progress_bar.stop()
        self.progress_bar.grid_remove()

        if error:
            self.rois = []
            self.gt_boxes = []
            self.nms_rois = []
            self.post_metrics = {}
            self.postprocess_results = {}
            self.postprocess_metrics = {}
            self.postprocess_thresholds = {}
            self.feature_columns = []
            self.feature_roi_index = 0
            self.feature_panel.grid_remove()
            self.hdr_title.configure(text=self.img_path.name if self.img_path else "Errore")
            self.hdr_sub.configure(text=f"Analisi non completata: {error}")
            self.pipe_led.set_state(True)
            self.pipe_lbl.configure(text="Errore analisi", fg="#FCA5A5")
            self.state_chip.configure(text="ERROR", fg_color="#FEE2E2", text_color=ACCENT_RED)
            messagebox.showerror("Errore analisi", f"Analisi non completata:\n{error}")
        else:
            self.rois = data.get("rois", []) if isinstance(data, dict) else (data or [])
            self.gt_boxes = data.get("gt_boxes", []) if isinstance(data, dict) else []
            self.nms_rois = data.get("nms_rois", []) if isinstance(data, dict) else []
            self.post_metrics = data.get("post_metrics", {}) if isinstance(data, dict) else {}
            self.postprocess_results = data.get("postprocess_results", {}) if isinstance(data, dict) else {}
            self.postprocess_metrics = data.get("postprocess_metrics", {}) if isinstance(data, dict) else {}
            self.postprocess_thresholds = data.get("postprocess_thresholds", {}) if isinstance(data, dict) else {}
            self.evaluate_best_rows_by_technique = _load_evaluate_best_rows_by_technique()
            self.selected_postprocess_technique = data.get("selected_postprocess_technique", DEFAULT_POSTPROCESS_TECHNIQUE) if isinstance(data, dict) else DEFAULT_POSTPROCESS_TECHNIQUE
            self.feature_columns = data.get("feature_columns", []) if isinstance(data, dict) else []
            self.feature_roi_index = 0

            # FIX: quando si passa alla prossima immagine con le frecce, la GUI
            # riceve le GT pulite dal worker ma non aveva ancora copiato dentro
            # self.gt_boxes lo stato matched del risultato NMS. Per questo le
            # GT apparivano rosse all'inizio e diventavano viola solo dopo click
            # su NMS. Qui ricalcoliamo subito detection + metriche + matched
            # per il risultato NMS, PRIMA del primo redraw.
            try:
                selected_technique = _canonical_technique_name(self.selected_postprocess_technique)
                technique_threshold = float((self.postprocess_thresholds or {}).get(
                    selected_technique,
                    _threshold_for_technique(selected_technique, ML_MIN_SCORE)
                ))
                dets_raw = postprocess_rois_by_technique(
                    self.rois,
                    technique=selected_technique,
                    min_ml_score=technique_threshold,
                    nms_iou_threshold=NMS_IOU_THRESHOLD,
                )
                self.nms_rois, evaluated_gt_boxes, fresh_metrics = _with_fresh_detection_labels(dets_raw, self.gt_boxes)
                for idx, gt in enumerate(self.gt_boxes):
                    gt["matched"] = bool(evaluated_gt_boxes[idx].get("matched", False)) if idx < len(evaluated_gt_boxes) else False
                fresh_metrics["selected_technique"] = selected_technique
                fresh_metrics["technique"] = selected_technique
                fresh_metrics["technique_label"] = TECHNIQUE_LABELS.get(selected_technique, selected_technique)
                fresh_metrics["ml_min_score"] = float(technique_threshold)
                fresh_metrics["threshold"] = float(technique_threshold)
                self.post_metrics = dict(fresh_metrics)
                self.postprocess_results[selected_technique] = [dict(d) for d in self.nms_rois]
                self.postprocess_metrics[selected_technique] = dict(fresh_metrics)
                self.postprocess_thresholds[selected_technique] = float(technique_threshold)
                self.selected_postprocess_technique = selected_technique
            except Exception as sync_exc:
                print(f"[WARN] Sync GT iniziale non riuscita: {sync_exc}")

            self._set_feature_panel_visible()
            n_nms = len(self.nms_rois)
            prec = float(self.post_metrics.get("precision", 0.0)) * 100.0
            rec = float(self.post_metrics.get("recall", 0.0)) * 100.0
            tp = int(self.post_metrics.get("TP", 0))
            fp = int(self.post_metrics.get("FP", 0))
            fn = int(self.post_metrics.get("FN", 0))
            self.btn_save.configure(state="normal")
            self.hdr_title.configure(text=self.img_path.name)
            active_desc = "zone candidate da CSV" if self.active_tab == "cpp" else f"risultato {TECHNIQUE_LABELS.get(getattr(self, 'selected_postprocess_technique', DEFAULT_POSTPROCESS_TECHNIQUE), 'post-processing ML')}"
            self.hdr_sub.configure(text=f"Vista attiva: {active_desc} · TP {tp}  FP {fp}  FN {fn} · criterio TP: {TP_IOU_RULE_TEXT}.")
            self._sync_view_selector()
            self._sync_config_selector()
            self.pipe_led.set_state(True)
            self.pipe_lbl.configure(text=f"Completata · {n_nms} rilevamenti", fg="#A7F3D0")
            self.state_chip.configure(text="REVIEW READY", fg_color="#DCFCE7", text_color=ACCENT_GREEN)
            self.redraw()
            self._trigger_flash()

        queued_index = self.pending_image_index
        self.pending_image_index = None
        self._update_browser_controls()
        if queued_index is not None and self.image_files:
            queued_index = max(0, min(queued_index, len(self.image_files) - 1))
            target = self.image_files[queued_index]
            if target != self.img_path:
                self.image_index = queued_index
                self.load_image(target, auto_analyze=True)

    def _trigger_flash(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        flash_id = self.canvas.create_rectangle(0, 0, w, h, fill="#FFFFFF", outline="")
        self.after(100, lambda: self.canvas.delete(flash_id))

    def _sync_current_ml_evaluation_state(self):
        """Sincronizza detection mostrate, metriche e colore GT/FN.

        Evita il caso incoerente in cui una GT resta rossa come FN mentre
        una ROI visualizzata ha IoU >= 0.50: prima di disegnare o mostrare
        tooltip, il risultato NMS viene rivalutato sulle box effettivamente
        mostrate e lo stato matched delle GT viene aggiornato.
        """
        if self.active_tab != "ml" or not self.gt_boxes:
            return
        if not self.nms_rois:
            return
        try:
            evaluated_dets, evaluated_gt_boxes, metrics = _with_fresh_detection_labels(self.nms_rois, self.gt_boxes)

            # Copia label e IoU valutate nelle stesse ROI che la GUI sta mostrando.
            for idx, det in enumerate(evaluated_dets):
                if idx >= len(self.nms_rois):
                    break
                for key in ("post_label", "best_gt_iou", "_best_gt", "_tp_gt"):
                    if key in det:
                        self.nms_rois[idx][key] = det.get(key)

            # Copia matched nelle GT originali: rosso solo se davvero FN per NMS.
            for idx, gt in enumerate(self.gt_boxes):
                gt["matched"] = bool(evaluated_gt_boxes[idx].get("matched", False)) if idx < len(evaluated_gt_boxes) else False

            # Mantieni coerenti i numeri mostrati in alto con NMS.
            if isinstance(self.post_metrics, dict):
                keep = {
                    "technique": self.selected_postprocess_technique,
                    "selected_technique": self.selected_postprocess_technique,
                    "technique_label": TECHNIQUE_LABELS.get(self.selected_postprocess_technique, self.selected_postprocess_technique),
                    "ml_min_score": self.post_metrics.get("ml_min_score", self.post_metrics.get("threshold", ML_MIN_SCORE)),
                    "threshold": self.post_metrics.get("threshold", self.post_metrics.get("ml_min_score", ML_MIN_SCORE)),
                }
                self.post_metrics.update(metrics)
                self.post_metrics.update(keep)
        except Exception as exc:
            print(f"[WARN] Sync valutazione viewer non riuscita: {exc}")

    def save_results(self):
        if not self.img_path or not self.rois: return
        try:
            if self.active_tab == "cpp":
                rois_to_save = self.rois
                saved = save_annotated_image(self.img_path, rois_to_save, "cpp", gt_boxes=self.gt_boxes)
            else:
                rois_to_save = self.nms_rois if self.nms_rois else self.rois
                saved = save_annotated_image(self.img_path, rois_to_save, "ml", gt_boxes=self.gt_boxes)
            messagebox.showinfo("Salvataggio", f"Risultato salvato:\n{saved.name}")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nel salvataggio:\n{e}")

    def _active_ml_threshold_for_ui(self):
        """Soglia ML effettiva mostrata in GUI: sempre ML >= 0.50."""
        return 0.50

    # ── Mouse tooltip ─────────────────────────────────────────────────────────
    def on_mouse_move(self, event):
        if not self.rois or not self.base_image:
            self.tooltip.hide()
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.base_image.size
        scale = min(w / iw, h / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox, oy = (w - dw) // 2, (h - dh) // 2

        display_rois = self.nms_rois if self.active_tab == "ml" and self.nms_rois else self.rois

        # In vista ML il tooltip usa la stessa valutazione del viewer.
        # Non basta che una ROI grezza abbia IoU >= 0.50: deve essere nella
        # detection finale NMS ed essere il TP vincente per quella ground truth.
        if self.active_tab == "ml":
            self._sync_current_ml_evaluation_state()
            display_rois = self.nms_rois if self.nms_rois else self.rois

        hovered_any = False
        for roi in display_rois:
            if self.active_tab == "ml" and int(roi.get("ml_pred", 0)) == 0:
                has_gt = "label" in roi and not pd.isna(roi.get("label"))
                gt = int(roi.get("label")) if has_gt else None
                if not (has_gt and gt == 1): continue

            rx = ox + float(roi["x"]) * scale
            ry = oy + float(roi["y"]) * scale
            rw = float(roi["width"]) * scale
            rh = float(roi["height"]) * scale

            if rx <= event.x <= rx+rw and ry <= event.y <= ry+rh:
                ml_score = float(roi.get("ml_score", 0))
                prob    = ml_score * 100
                roi_id  = roi.get("roi_id", 0)
                method  = roi.get("method", "N/D")
                has_gt  = "label" in roi and not pd.isna(roi.get("label"))
                gt      = int(roi.get("label")) if has_gt else None
                pred    = int(roi.get("ml_pred", 0))
                best_iou = _best_gt_iou_for_display(roi, self.gt_boxes if self.active_tab == "ml" else None)
                iou_txt = _format_iou_for_ui(best_iou)
                active_thr = self._active_ml_threshold_for_ui() if self.active_tab == "ml" else 0.50
                in_active_threshold = ml_score >= active_thr
                outcome = "CANDIDATO"
                if self.active_tab == "ml":
                    post_label = roi.get("post_label")
                    if post_label == "TP":
                        outcome = f"VERA FRATTURA (TP: IoU {iou_txt} >= {TP_IOU_THRESHOLD:.2f})"
                    elif post_label == "FP":
                        outcome = f"FALSO POSITIVO (FP: IoU {iou_txt}, serve {TP_IOU_RULE_TEXT})"
                    elif post_label == "IGNORE":
                        outcome = f"IGNORATA (duplicato; IoU {iou_txt}, GT gia' assegnata)"
                    elif not in_active_threshold:
                        # Caso importante: la ROI puo' avere IoU buona con la GT, ma non entra
                        # nel risultato NMS perche' la soglia operativa fissa e' 0.50.
                        if best_iou is None:
                            outcome = f"ESCLUSA da NMS: ML {ml_score:.3f} < soglia {active_thr:.3f}"
                        else:
                            outcome = f"ESCLUSA da NMS: ML {ml_score:.3f} < soglia {active_thr:.3f}; IoU GT {iou_txt} non valutata come TP"
                    elif pred == 1:
                        if best_iou is None:
                            outcome = f"ROI sopra soglia ML>=0.50 ma IoU non valutata (serve {TP_IOU_RULE_TEXT})"
                        elif _is_tp_iou_match(best_iou):
                            # Se non c'e' post_label ma la ROI supera soglia e IoU, e' una incoerenza
                            # di sincronizzazione: lo segnaliamo chiaramente invece di parlare di duplicato.
                            outcome = f"ROI sopra soglia ML>=0.50 e IoU {iou_txt} >= {TP_IOU_THRESHOLD:.2f}, ma non presente come TP finale: rianalizza NMS"
                        else:
                            outcome = f"ROI sopra soglia ML>=0.50 ma NON TP (IoU {iou_txt} < {TP_IOU_THRESHOLD:.2f})"
                    elif has_gt and gt == 1:
                        outcome = "FRATTURA PERSA (FN)"
                    else:
                        outcome = "ML DISCARD"
                extra = ""
                if self.active_tab == "ml":
                    extra += f"\nBest IoU GT: {iou_txt}\nRegola TP: {TP_IOU_RULE_TEXT}\nSoglia ML NMS: > {active_thr:.3f}"
                if self.active_tab == "ml" and roi.get("nms_count"):
                    extra += f"\nNMS di: {roi.get('nms_count')} ROI\nROI sorgenti: {roi.get('source_roi_ids', '')}"
                text = f"ROI {roi_id} ({method})\nProbabilità ML: {prob:.1f}%\nStato: {outcome}\nDim: {int(roi['width'])}x{int(roi['height'])}{extra}"
                self.tooltip.show(text, event.x, event.y)
                hovered_any = True
                break

        if not hovered_any:
            self.tooltip.hide()

    # ── Redraw ────────────────────────────────────────────────────────────────
    def redraw(self):
        if not self.base_image:
            self._draw_empty_placeholder()
            return
        # Prima del disegno riallinea GT e detection NMS:
        # rosso = FN reale, viola = GT coperta da almeno un TP.
        self._sync_current_ml_evaluation_state()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 50 or h < 50: return

        self.canvas.delete("all")

        iw, ih = self.base_image.size
        scale   = min(w / iw, h / ih)
        dw, dh  = max(10, int(iw * scale)), max(10, int(ih * scale))
        ox, oy  = (w - dw) // 2, (h - dh) // 2

        # Resize cache
        if (not hasattr(self, "_last_size") or
            self._last_size != (w, h) or
            not hasattr(self, "_cached_path") or
            self._cached_path != self.img_path):
            self._last_size   = (w, h)
            self._cached_path = self.img_path
            resized           = self.base_image.resize((dw, dh), Image.Resampling.LANCZOS)
            self.tk_image     = ImageTk.PhotoImage(resized)

        # Grid and calm background lights behind image only
        self._draw_tech_grid(w, h)
        self._draw_ambient_lights(w, h)
        self.canvas.create_rectangle(ox - 10, oy - 10, ox + dw + 10, oy + dh + 10, fill="#090F1C", outline="")
        self.canvas.create_rectangle(ox - 5, oy - 5, ox + dw + 5, oy + dh + 5, outline="#1D2A3F", width=1)
        self.canvas.create_image(w//2, h//2, image=self.tk_image, anchor="center")
        self.canvas.create_rectangle(ox, oy, ox + dw, oy + dh, outline="#5EE7FF", width=1)
        view_status = "CSV FEATURE" if self.active_tab == "cpp" and self.rois else ("ML REVIEW" if self.rois else "IMAGE READY")
        self._draw_viewer_hud(w, h, view_status)

        # Pulsing HUD corner brackets
        bl = int(18 + 7 * math.sin(self.pulse * math.pi / 30))
        clr = VIEWER_ACCENT
        t = 2
        for bx, by, dx, dy in [(12,12,1,1),(w-12,12,-1,1),(12,h-12,1,-1),(w-12,h-12,-1,-1)]:
            self.canvas.create_line(bx, by, bx + dx*bl, by,       fill=clr, width=t)
            self.canvas.create_line(bx, by, bx,          by + dy*bl, fill=clr, width=t)

        # ROI bounding boxes
        if self.rois:
            # Always draw ground-truth boxes as an underlay so appaiono sia nella
            # vista Image Processing sia nella vista ML. Viola = GT; nella vista
            # post-nms una GT non abbinata da nessuna detection e' un FN -> rossa.
            for roi in self.gt_boxes:
                rx = ox + float(roi["x"]) * scale
                ry = oy + float(roi["y"]) * scale
                rw = float(roi["width"])  * scale
                rh = float(roi["height"]) * scale
                is_fn = self.active_tab == "ml" and bool(self.post_metrics) and not roi.get("matched", False)
                gt_color = ACCENT_RED if is_fn else ACCENT_PURPLE
                self.canvas.create_rectangle(rx, ry, rx+rw, ry+rh, outline=gt_color, width=2)

            display_rois = self.nms_rois if self.active_tab == "ml" and self.nms_rois else self.rois

            for roi_idx, roi in enumerate(display_rois):
                is_ml = self.active_tab == "ml"

                # Le detection post-nms portano post_label (TP/FP/IGNORE) a
                # prescindere dal nome del metodo (NMS storico o NMS attuale).
                if is_ml and roi.get("post_label") in ("TP", "FP", "IGNORE"):
                    pl = roi.get("post_label")
                    if pl == "IGNORE":
                        continue   # detection su una GT gia' assegnata: ignorata, non disegnata
                    color = ACCENT_GREEN if pl == "TP" else ACCENT_ORANGE
                else:
                    has_gt = "label" in roi and not pd.isna(roi.get("label"))
                    try:
                        gt = int(roi.get("label")) if has_gt else None
                    except Exception:
                        gt = None
                    if has_gt and gt == 1:
                        rx = ox + float(roi["x"]) * scale
                        ry = oy + float(roi["y"]) * scale
                        rw = float(roi["width"])  * scale
                        rh = float(roi["height"]) * scale
                        self.canvas.create_rectangle(rx, ry, rx+rw, ry+rh, outline=ACCENT_PURPLE, width=2)

                    pred  = int(roi.get("ml_pred", 0))
                    if is_ml:
                        best_iou = _best_gt_iou_for_display(roi, self.gt_boxes)
                        if pred == 1:
                            # Non basta label==1: in vista ML il verde e' riservato a TP con IoU>0.50.
                            color = ACCENT_GREEN if _is_tp_iou_match(best_iou) else ACCENT_ORANGE
                        elif has_gt and gt == 1:
                            color = ACCENT_RED
                        else:
                            continue
                    else:
                        color = CANDIDATE_CLR

                rx = ox + float(roi["x"]) * scale
                ry = oy + float(roi["y"]) * scale
                rw = float(roi["width"])  * scale
                rh = float(roi["height"]) * scale
                selected = (not is_ml and roi_idx == self.feature_roi_index)
                if selected:
                    self.canvas.create_rectangle(rx-3, ry-3, rx+rw+3, ry+rh+3, outline="#FFFFFF", width=1)
                self.canvas.create_rectangle(rx, ry, rx+rw, ry+rh, outline=color, width=4 if selected else (3 if is_ml else 2))
                roi_id = roi.get("roi_id", "")
                label = str(roi_id)
                if is_ml and roi.get("nms_count"):
                    label = f"{roi_id} x{roi.get('nms_count')}"
                self.canvas.create_text(rx+4, ry+8, text=label, fill=color,
                                        font=("Menlo", 8, "bold"), anchor="w")

            if self.active_tab == "cpp":
                txt = f"CSV FEATURE  ·  ROI={len(self.rois)}  ·  CLICCA UN BOX PER LE FEATURE"
                self.canvas.create_rectangle(24, h-58, 440, h-26, fill="#0B1728", outline="#334155", width=1)
                self.canvas.create_text(38, h-42, text=txt, fill="#B8F3F8", font=("Segoe UI", 10, "bold"), anchor="w")
            elif self.active_tab == "ml" and self.post_metrics:
                txt = (
                    f"{TECHNIQUE_LABELS.get(getattr(self, 'selected_postprocess_technique', DEFAULT_POSTPROCESS_TECHNIQUE), 'POST').upper()}  TP@{TP_IOU_RULE_TEXT}={self.post_metrics.get('TP', 0)}  "
                    f"FP={self.post_metrics.get('FP', 0)}  "
                    f"FN={self.post_metrics.get('FN', 0)}  "
                    f"P={float(self.post_metrics.get('precision', 0.0))*100:.1f}%  "
                    f"R={float(self.post_metrics.get('recall', 0.0))*100:.1f}%"
                )
                self.canvas.create_rectangle(24, h-58, 610, h-26, fill="#0B1728", outline="#334155", width=1)
                self.canvas.create_text(38, h-42, text=txt, fill="#E0F2FE", font=("SF Pro Text", 10, "bold"), anchor="w")

    # ── Animation loop ────────────────────────────────────────────────────────
    def _animate(self):
        self.pulse  = (self.pulse + 1) % 60
        self.phase += 0.02
        factor = 0.5 + 0.5 * math.sin(self.pulse * math.pi / 30)

        try:
            self._sidebar_accent.configure(bg="#DADAE0")
        except Exception:
            pass

        self.logo_led.set_color(ACCENT_GREEN)

        try:
            if self.btn_run.cget("state") == "normal" and not self.hovering_run:
                self.btn_run.configure(fg_color=ACCENT_BLUE, border_color="#4DA3FF")
        except Exception:
            pass

        if not self.img_path and not self.hovering_dz:
            self.dz_icon.configure(fg=ACCENT_BLUE)
            self.dz_main.configure(fg=TEXT_MAIN)

        if not self.hovering_viewer:
            self.viewer_shell.configure(border_color=GLASS_BORDER)

        if not self.img_path:
            self._draw_empty_placeholder()

        self.after(250, self._animate)


# ── Avvio GUI ─────────────────────────────────────────────────────────────────
def _fallback_cli():
    print("[*] Fallback CLI disabilitato: questa versione non esegue run_pipeline.py.")
    print(f"[*] Usa CSV feature gia' presenti in: {CSV_FEATURE_DIR}")
    print(f"[*] Oppure CSV ufficiali in sola lettura: {IPA_CSV_FEATURE_DIR}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        app = FractureApp()
        app.mainloop()
    except Exception:
        import traceback
        err = traceback.format_exc()
        PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = PIPELINE_LOG_DIR / "gui_error.log"
        log_path = _safe_write_text_file(log_path, err, encoding="utf-8")
        print("[!] Errore apertura GUI. Dettagli salvati in gui_error.log")
        print(err)
        sys.exit(1)
