#!/usr/bin/env python3
# Build marker: EVALUATE_NMS_ONLY_BUILD_2026_06_24
# Build marker: EVALUATE_APFORMULA_IOUFIX_BUILD_2026_06_19
# Build marker: EVALUATE_PR_AP_CLEAN_PLOT_BUILD_2026_06_20
# Build marker: EVALUATE_APFORMULA_BUILD_2026_06_19
# Build marker: EVALUATE_FPOVERLAPIGNORE_BUILD_2026_06_19
# Build marker: EVALUATE_FPOVERLAPIGNORE_NAMEFIX_BUILD_2026_06_19
# Build marker: FIXED_MLGE50_IOUGE50_IGNOREGE20_BUILD_2026_06_19
# FIXED_BUILD_CHATGPT_2026_06_19_TP_GE_IGNORE_GT10_BEST_OPERATING_POINT_TPGE_IGNORE_GT010
"""
Valutazione TEST set per fratture classe YOLO 3 solo con post-processing NMS.

Genera:
- curve Precision-Recall e FROC per NMS;
- Average Precision come area sotto la curva Precision-Recall, secondo detection file ordinato per confidence score;
- CSV con tutti i punti delle curve;
- JSON best_postprocess_params.json letto da pipeline_gui.py;
- riepilogo testuale per la GUI.

Regola TP (modalita' NMS-only):
    una detection combacia con una GT se IoU >= 0.50; se piu' detection
    combaciano con la stessa GT, la TP e' quella con ML score piu' alto
    (NON quella con IoU migliore). Una detection non-TP che si sovrappone
    alla stessa GROUND TRUTH gia' trovata (IoU con la GT >= ignore_iou,
    default 0.30) e che ha IoU con la GT inferiore a quella della TP viene
    IGNORATA (duplicato: non conta come FP). Ogni altra detection e' FP.

Uso tipico dalla root del progetto:
    python evaluate_test_postprocess.py --labels "/percorso/alle/labels" --no-show-plots
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin

# Necessaria per caricare modelli pickle creati con questa classe.
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, indices):
        self.indices = indices
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.indices]

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ── Radice del progetto (rilevata automaticamente) ──────────────────────────
# Questo file si trova in <progetto>/ML, quindi la radice del progetto e' due
# livelli sopra. Cosi' funziona su qualsiasi computer e sistema operativo,
# anche spostando la cartella, senza dover modificare i percorsi a mano.
WINDOWS_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_IMAGE_DIR = WINDOWS_PROJECT_ROOT / "img_fracture"
PROJECT_ML_DIR = WINDOWS_PROJECT_ROOT / "ML"
TRAINING_RESULTS_DIR = PROJECT_ML_DIR / "risultati_addestramento"
TRAINING_SPLIT_IMAGES_DIR = TRAINING_RESULTS_DIR / "split_images"
EVALUATION_RESULTS_DIR = PROJECT_ML_DIR / "risultati_valutazione_postprocessing"
EVALUATION_CSV_DIR = EVALUATION_RESULTS_DIR / "csv"
EVALUATION_CURVES_DIR = EVALUATION_RESULTS_DIR / "curve"
PIPELINE_RESULTS_DIR = PROJECT_ML_DIR / "risultati_pipeline"

DEFAULT_LABEL_DIR = PROJECT_IMAGE_DIR
DEFAULT_MANIFEST = TRAINING_SPLIT_IMAGES_DIR / "test_manifest.csv"
DEFAULT_IMAGES_DIR = TRAINING_SPLIT_IMAGES_DIR / "test"
# Set di validation (per lo sweep NMS automatico): scelto dal training.
DEFAULT_VAL_MANIFEST = TRAINING_SPLIT_IMAGES_DIR / "validation_manifest.csv"
DEFAULT_VAL_IMAGES_DIR = TRAINING_SPLIT_IMAGES_DIR / "validation"
DEFAULT_OUT_CSV = EVALUATION_CSV_DIR / "postprocess_test_metrics_class3.csv"
DEFAULT_CURVES_DIR = EVALUATION_CURVES_DIR

# Modalita' NMS-only: preprocessing e merge non vengono piu' valutati.
NMS_TECHNIQUE = "nms"
TECHNIQUES = [NMS_TECHNIQUE]
TECHNIQUE_LABELS = {
    NMS_TECHNIQUE: "NMS",
}
TECHNIQUE_TIE_PREFERENCE = {
    NMS_TECHNIQUE: 1,
}


def _ensure_numpy_matplotlib():
    try:
        import numpy as np  # type: ignore
    except ImportError:
        import subprocess
        print("[*] Installazione numpy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        import numpy as np  # type: ignore
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        import subprocess
        print("[*] Installazione matplotlib...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt  # type: ignore
    return np, plt


np, plt = _ensure_numpy_matplotlib()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _resolve_project_path(value: str | Path, project_root: Path, kind: str = "path") -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = project_root / p
    p = p.resolve()
    # Compatibilita' con vecchi default errati ml/ml/...
    if not p.exists() and "ml" in p.parts:
        parts = list(p.parts)
        collapsed: List[str] = []
        for part in parts:
            if part == "ml" and collapsed and collapsed[-1] == "ml":
                continue
            collapsed.append(part)
        try:
            alt = Path(*collapsed)
            if alt.exists() or kind == "output":
                p = alt.resolve()
        except Exception:
            pass
    return p


def _load_pipeline_module(project_root: Path, module_path_arg: str | None = None):
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    roots: List[Path] = []
    for r in [project_root, script_dir, cwd, cwd / "ipa_project", project_root / "ipa_project", script_dir.parent, script_dir.parent / "ipa_project"]:
        try:
            rr = r.resolve()
        except Exception:
            rr = r
        if rr not in roots:
            roots.append(rr)
    names = ["pipeline_ML_percorsi_v5.py", "pipeline_ML_ordinata_v5.py", "pipeline_ML_ordinata_v4.py", "pipeline_ML_ordinata_v3.py", "pipeline_gui.py", "pipeline_gui_postprocess_class3.py", "pipeline_gui_postprocess.py", "pipeline_gui(4).py"]
    candidates: List[Path] = []
    if module_path_arg:
        mp = Path(module_path_arg)
        if mp.is_absolute():
            candidates.append(mp)
        else:
            candidates.extend([cwd / mp, script_dir / mp, project_root / mp])
    for root in roots:
        for name in names:
            candidates.append(root / name)
    seen: set[str] = set()
    unique: List[Path] = []
    for c in candidates:
        key = str(c.resolve()).lower() if c.exists() else str(c).lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    for p in unique:
        if p.exists() and p.is_file():
            spec = importlib.util.spec_from_file_location("pipeline_gui_eval_module", str(p))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["pipeline_gui_eval_module"] = mod
            spec.loader.exec_module(mod)
            return mod, p.resolve()
    searched = "\n".join(f" - {p}" for p in unique)
    raise FileNotFoundError("Non trovo la pipeline/interfaccia. Passa il percorso con --pipeline-file.\n" + searched)


def _read_manifest_images(manifest_path: Path, project_root: Path) -> List[Path]:
    images: List[Path] = []
    seen: set[str] = set()
    if not manifest_path.exists():
        return images
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            found = str(row.get("found", "")).lower() in {"true", "1", "yes", "y"}
            if not found:
                continue
            path_value = row.get("copied_path") or row.get("source_path") or ""
            if not path_value:
                continue
            p = Path(path_value)
            if not p.is_absolute():
                p = project_root / p
            if not p.exists() or not p.is_file():
                src = row.get("source_path") or ""
                p = Path(src)
                if not p.is_absolute():
                    p = project_root / p
            if not p.exists() or not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            key = row.get("image") or p.name
            if key in seen:
                continue
            seen.add(key)
            images.append(p.resolve())
    return images


def _collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Cartella immagini non trovata: {images_dir}")
    imgs = sorted(p.resolve() for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        raise FileNotFoundError(f"Nessuna immagine trovata in: {images_dir}")
    return imgs


def _label_search_dirs(labels_arg: Optional[str], image_path: Path) -> List[Path]:
    candidates: List[Path] = []
    if labels_arg:
        base = Path(labels_arg)
        if not base.is_absolute():
            base = Path.cwd() / base
        candidates.extend([base, base / "labels", base / "label"])
    candidates.extend([PROJECT_IMAGE_DIR, PROJECT_IMAGE_DIR / "labels", PROJECT_IMAGE_DIR / "label"])
    env = os.getenv("IPA_GT_LABELS_DIR")
    if env:
        candidates.extend([Path(env), Path(env) / "labels", Path(env) / "label"])
    candidates.extend([
        image_path.parent,
        image_path.parent / "labels",
        image_path.parent / "label",
        image_path.parent.parent / "labels",
        image_path.parent.parent / "label",
    ])
    out: List[Path] = []
    seen: set[str] = set()
    for c in candidates:
        if c.exists() and c.is_dir():
            key = str(c.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(c.resolve())
    return out


def _possible_label_stems(image_stem: str) -> List[str]:
    stems = [image_stem]
    cur = image_stem
    while re.search(r"_\d+$", cur):
        cur = re.sub(r"_\d+$", "", cur)
        if cur not in stems:
            stems.append(cur)
    return stems


def _find_label_file(image_path: Path, labels_arg: Optional[str]) -> Optional[Path]:
    for root in _label_search_dirs(labels_arg, image_path):
        for stem in _possible_label_stems(image_path.stem):
            for c in [root / f"{stem}.txt", root / f"{stem}{image_path.suffix}.txt"]:
                if c.exists() and c.is_file():
                    return c.resolve()
    return None


def _load_image_size(pipeline_mod: Any, image_path: Path) -> Tuple[int, int]:
    if hasattr(pipeline_mod, "load_16bit_image_as_rgb"):
        img = pipeline_mod.load_16bit_image_as_rgb(image_path)
        return img.size
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size


def _load_yolo_class_boxes(label_file: Path, image_size: Tuple[int, int], gt_class: int) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    img_w, img_h = image_size
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
            if cls_id != gt_class:
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
    return boxes


def _load_gt_boxes_for_image(pipeline_mod: Any, image_path: Path, labels_arg: Optional[str], gt_class: int) -> Tuple[List[Dict[str, Any]], str]:
    label_file = _find_label_file(image_path, labels_arg)
    if label_file is None:
        return [], ""
    return _load_yolo_class_boxes(label_file, _load_image_size(pipeline_mod, image_path), gt_class), str(label_file)


# ──────────────────────────────────────────────────────────────────────────────
# Geometria / post-processing fallback
# ──────────────────────────────────────────────────────────────────────────────

def _box_xyxy(b: Dict[str, Any]) -> Tuple[float, float, float, float]:
    x = _safe_float(b.get("x", 0.0))
    y = _safe_float(b.get("y", 0.0))
    w = _safe_float(b.get("width", 0.0))
    h = _safe_float(b.get("height", 0.0))
    return x, y, x + w, y + h


def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _box_xyxy(a)
    bx1, by1, bx2, by2 = _box_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _safe_score(roi: Dict[str, Any]) -> float:
    return max(0.0, min(1.0, _safe_float(roi.get("ml_score", roi.get("score", 0.0)), 0.0)))


def _filtered_candidates(rois: Iterable[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rois or []:
        score = _safe_score(r)
        if score >= threshold:
            rr = dict(r)
            rr["ml_score"] = score
            rr["score"] = score
            rr["ml_pred"] = 1
            out.append(rr)
    return out


def _make_detection(seed: Dict[str, Any], group: List[Dict[str, Any]], idx: int, technique: str, threshold: float, union_box: bool = False) -> Dict[str, Any]:
    if union_box and group:
        x1 = min(_box_xyxy(r)[0] for r in group)
        y1 = min(_box_xyxy(r)[1] for r in group)
        x2 = max(_box_xyxy(r)[2] for r in group)
        y2 = max(_box_xyxy(r)[3] for r in group)
        x, y, w, h = x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)
    else:
        x, y, x2, y2 = _box_xyxy(seed)
        w, h = max(1.0, x2 - x), max(1.0, y2 - y)
    score = max((_safe_score(r) for r in group), default=_safe_score(seed))
    return {
        "roi_id": f"{technique.upper().replace('_', '')[:3]}{idx}",
        "method": TECHNIQUE_LABELS.get(technique, technique),
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "score": float(score),
        "ml_score": float(score),
        "ml_pred": 1,
        "merged_count": len(group),
        "seed_roi_id": str(seed.get("roi_id", "")),
        "source_roi_ids": ",".join(str(r.get("roi_id", "")) for r in group if str(r.get("roi_id", ""))),
        "technique": technique,
        "technique_label": TECHNIQUE_LABELS.get(technique, technique),
        "merge_min_ml_score": float(threshold),
    }



def _fallback_nms(rois: List[Dict[str, Any]], threshold: float, nms_iou: float) -> List[Dict[str, Any]]:
    remaining = sorted(_filtered_candidates(rois, threshold), key=_safe_score, reverse=True)
    out: List[Dict[str, Any]] = []
    while remaining:
        best = remaining.pop(0)
        suppressed: List[Dict[str, Any]] = []
        survivors: List[Dict[str, Any]] = []
        for cand in remaining:
            if _iou(best, cand) >= nms_iou:
                suppressed.append(cand)
            else:
                survivors.append(cand)
        remaining = survivors
        out.append(_make_detection(best, [best] + suppressed, len(out) + 1, NMS_TECHNIQUE, threshold, union_box=False))
        out[-1]["nms_iou_threshold"] = float(nms_iou)
    return out



def _postprocess(pipeline_mod: Any, rois: List[Dict[str, Any]], technique: str, threshold: float, nms_iou: float, merge_iou: float) -> List[Dict[str, Any]]:
    technique = _canonical_technique(technique)
    # Modalita' NMS-only: anche se arriva un alias storico, viene ricondotto a NMS.
    if hasattr(pipeline_mod, "postprocess_rois_by_technique"):
        try:
            return list(pipeline_mod.postprocess_rois_by_technique(
                rois,
                technique=NMS_TECHNIQUE,
                min_ml_score=float(threshold),
                nms_iou_threshold=float(nms_iou),
                merge_iou_threshold=float(merge_iou),  # parametro lasciato solo per compatibilita' firma
            ))
        except TypeError:
            # Fallback per vecchie versioni della GUI.
            pass
    return _fallback_nms(rois, threshold, nms_iou)


def _canonical_technique(name: Any) -> str:
    # NMS-only: qualsiasi nome/alias storico viene forzato a NMS.
    return NMS_TECHNIQUE


# ──────────────────────────────────────────────────────────────────────────────
# Valutazione coerente: TP se IoU >= soglia
# ──────────────────────────────────────────────────────────────────────────────



# IGNORE duplicati: una detection non-TP che si sovrappone alla GROUND TRUTH gia'
# trovata (IoU con la GT >= ignore_iou) e con IoU sulla GT inferiore a quella della
# TP viene marcata IGNORE (non conta come FP). E' la regola "duplicato sulla stessa
# frattura gia' trovata": fra due ROI che insistono sulla stessa GT si ignora quella
# con IoU piu' basso sulla GT.


def _evaluate_local(detections: List[Dict[str, Any]], gt_boxes: List[Dict[str, Any]], tp_iou: float, ignore_iou: float) -> Dict[str, Any]:
    gt = [dict(g) for g in (gt_boxes or [])]
    dets = [dict(d) for d in (detections or [])]
    for g in gt:
        g["matched"] = False
    for det in dets:
        best_iou = 0.0
        best_gi = None
        for gi, g in enumerate(gt):
            v = _iou(det, g)
            if v > best_iou:
                best_iou = v
                best_gi = gi
        det["best_gt_iou"] = float(best_iou)
        det["_best_gt"] = best_gi
        det["_tp_gt"] = best_gi if best_gi is not None and best_iou >= tp_iou else None

    # Tra le detection che combaciano con la stessa GT (IoU >= tp_iou) la TP e'
    # quella con ML score piu' alto, NON quella con IoU migliore.
    tp = 0
    tp_by_gi: Dict[int, Dict[str, Any]] = {}
    for gi, g in enumerate(gt):
        contenders = [d for d in dets if d.get("_tp_gt") == gi]
        if not contenders:
            continue
        winner = max(
            contenders,
            key=lambda d: (
                _safe_float(d.get("ml_score", d.get("score", 0.0))),
                _safe_float(d.get("best_gt_iou", 0.0)),
            ),
        )
        winner["post_label"] = "TP"
        g["matched"] = True
        tp += 1
        tp_by_gi[gi] = winner

    # IGNORE: una detection non-TP che si sovrappone alla GROUND TRUTH gia' trovata
    # (IoU con la GT >= ignore_iou) con IoU sulla GT inferiore (o uguale) a quella
    # della TP viene IGNORATA (duplicato sulla stessa frattura), quindi NON conta
    # come FP. Ogni altra detection non-TP resta un FP.
    for det in dets:
        if det.get("post_label") == "TP":
            continue
        gi = det.get("_best_gt")
        det_gt_iou = _safe_float(det.get("best_gt_iou", 0.0))
        tp_det = tp_by_gi.get(gi) if gi is not None else None
        if (tp_det is not None and det_gt_iou >= ignore_iou
                and det_gt_iou <= _safe_float(tp_det.get("best_gt_iou", 0.0))):
            det["post_label"] = "IGNORE"
            det["ignore_reason"] = "overlap_ge_ignore_iou_with_found_gt"
            det["duplicate_kept_roi_id"] = str(tp_det.get("roi_id", ""))
        else:
            det["post_label"] = "FP"
            det["ignore_reason"] = ""
            det["duplicate_kept_roi_id"] = ""

    fp = sum(1 for d in dets if d.get("post_label") == "FP")
    ignored = sum(1 for d in dets if d.get("post_label") == "IGNORE")
    fn = sum(1 for g in gt if not g.get("matched", False))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    beta2 = 4.0
    f2 = ((1.0 + beta2) * precision * recall / (beta2 * precision + recall)) if (beta2 * precision + recall) else 0.0
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "IGNORED": int(ignored),
        "GT": int(len(gt)),
        "detections": int(len(dets)),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "F1": float(f1),
        "F2": float(f2),
        "tp_iou": float(tp_iou),
        "tp_rule": f"IoU >= {tp_iou:.2f}",
        "ignore_iou": float(ignore_iou),
    }


def _evaluate(pipeline_mod: Any, detections: List[Dict[str, Any]], gt_boxes: List[Dict[str, Any]], tp_iou: float, ignore_iou: float) -> Dict[str, Any]:
    """Valutazione interna fissa e coerente: TP se IoU >= 0.50."""
    return _evaluate_local(detections, gt_boxes, tp_iou, ignore_iou)


# ──────────────────────────────────────────────────────────────────────────────
# Curve
# ──────────────────────────────────────────────────────────────────────────────

def _threshold_grid(all_scores: List[float], max_points: int = 600) -> np.ndarray:
    scores = np.asarray([_safe_float(s, np.nan) for s in all_scores], dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([1.000001, 0.5, -0.000001], dtype=float)
    uniq = np.unique(np.clip(scores, 0.0, 1.0))
    if uniq.size > max_points:
        uniq = np.unique(np.quantile(uniq, np.linspace(0.0, 1.0, max_points)))
    # Alta -> bassa: a ogni soglia ricreo davvero il post-processing.
    # Includo sempre 0.50 per avere il punto operativo fisso ML >= 50%.
    return np.unique(np.r_[min(1.0, uniq.max()) + 1e-9, uniq[::-1], 0.50, max(0.0, uniq.min()) - 1e-9])[::-1]


def _curve_points_for_technique(
    pipeline_mod: Any,
    image_records: List[Dict[str, Any]],
    thresholds: np.ndarray,
    technique: str,
    nms_iou: float,
    merge_iou: float,
    tp_iou: float,
    ignore_iou: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_images = max(1, len(image_records))
    total_gt = sum(len(rec.get("gt_boxes", [])) for rec in image_records)
    for thr in thresholds:
        total = {"TP": 0, "FP": 0, "FN": 0, "IGNORED": 0, "detections": 0}
        for rec in image_records:
            dets = _postprocess(pipeline_mod, rec.get("rois", []), technique, float(thr), nms_iou, merge_iou)
            m = _evaluate(pipeline_mod, dets, rec.get("gt_boxes", []), tp_iou, ignore_iou)
            for k in total:
                total[k] += int(m.get(k, 0))
        precision = total["TP"] / (total["TP"] + total["FP"]) if (total["TP"] + total["FP"]) else 1.0
        recall = total["TP"] / total_gt if total_gt else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        beta2 = 4.0
        f2 = ((1.0 + beta2) * precision * recall / (beta2 * precision + recall)) if (beta2 * precision + recall) else 0.0
        rows.append({
            "technique": technique,
            "technique_label": TECHNIQUE_LABELS.get(technique, technique),
            "threshold": float(thr),
            "TP": int(total["TP"]),
            "FP": int(total["FP"]),
            "FN": int(total["FN"]),
            "IGNORED": int(total["IGNORED"]),
            "GT": int(total_gt),
            "detections": int(total["detections"]),
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "F1": float(f1),
            "F2": float(f2),
            "fp_per_image": float(total["FP"] / n_images),
        })
    return rows


def _finite_xy(x_values: Iterable[float], y_values: Iterable[float]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for x, y in zip(x_values, y_values):
        xf = _safe_float(x, np.nan)
        yf = _safe_float(y, np.nan)
        if np.isfinite(xf) and np.isfinite(yf):
            out.append((float(xf), float(yf)))
    return out


def _trapz(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(1, min(len(xs), len(ys))):
        area += (xs[i] - xs[i - 1]) * ((ys[i] + ys[i - 1]) / 2.0)
    return float(abs(area))


def _aggregate_max_y_by_x(x_values: Iterable[float], y_values: Iterable[float]) -> Tuple[List[float], List[float]]:
    buckets: Dict[float, float] = {}
    for x, y in _finite_xy(x_values, y_values):
        key = round(float(x), 12)
        buckets[key] = max(float(y), buckets.get(key, float("-inf")))
    if not buckets:
        return [], []
    xs = sorted(buckets.keys())
    ys = [buckets[x] for x in xs]
    return xs, ys


def _pr_curve(rows: List[Dict[str, Any]]) -> Tuple[List[float], List[float], float]:
    recall, precision = _aggregate_max_y_by_x([r["recall"] for r in rows], [r["precision"] for r in rows])
    if not recall:
        return [], [], 0.0
    # inviluppo PR standard: precisione monotona non crescente con recall crescente.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    if recall[0] > 0:
        recall = [0.0] + recall
        precision = [precision[0]] + precision
    # IMPORTANTE: non estendo artificialmente la curva fino a recall=1.0.
    # Se una tecnica di post-processing non raggiunge recall 1.0, l'area PR
    # deve restare piu' piccola: e' una penalizzazione corretta delle FN residue.
    return recall, precision, _trapz(recall, precision)


# ──────────────────────────────────────────────────────────────────────────────
# Average Precision secondo formula detection-file:
#   1) assegna un confidence score a ogni finding;
#   2) match con GT e rimuove/ignora doppie detection;
#   3) crea coppie (label, confidence score);
#   4) ordina per score decrescente e calcola TP/FP cumulativi;
#   5) AP = area trapezoidale sotto la curva precision-recall.
# ──────────────────────────────────────────────────────────────────────────────

def _ap_match_image_detections(
    detections: List[Dict[str, Any]],
    gt_boxes: List[Dict[str, Any]],
    image_name: str,
    technique: str,
    tp_iou: float,
    duplicate_iou: float,
) -> List[Dict[str, Any]]:
    """Etichetta le detection di una singola immagine per AP.

    Diversamente dal punto operativo a soglia fissa, qui si costruisce il file
    detection richiesto dalla formula AP: coppie label/confidence. Le doppie
    detection vengono escluse dalla lista TP/FP marcandole IGNORE.
    """
    dets = [dict(d) for d in (detections or [])]
    gt = [dict(g) for g in (gt_boxes or [])]
    for i, det in enumerate(dets):
        det["_ap_index"] = i
        det["_ap_score"] = _safe_score(det)
        det["_ap_label"] = None
        det["_ap_reason"] = ""
        best_iou = 0.0
        best_gi = None
        for gi, g in enumerate(gt):
            v = _iou(det, g)
            if v > best_iou:
                best_iou = v
                best_gi = gi
        det["best_gt_iou"] = float(best_iou)
        det["_best_gt"] = best_gi

    # Step 2 delle formule: per ogni GT i finding con IoU >= soglia sono candidati.
    # Vince (TP) quello con ML/confidence score maggiore. Una detection non-TP che
    # si sovrappone alla GROUND TRUTH gia' trovata (IoU con la GT >= duplicate_iou) e
    # con IoU sulla GT inferiore (o uguale) a quella della TP viene marcata IGNORE ed
    # esclusa dalla lista TP/FP dell'AP (duplicato). Ogni altra detection non-TP e' un FP.
    tp_by_gi: Dict[int, Dict[str, Any]] = {}
    for gi, _g in enumerate(gt):
        contenders = [d for d in dets if _iou(d, gt[gi]) >= float(tp_iou)]
        if not contenders:
            continue
        winner = max(contenders, key=lambda d: (_safe_float(d.get("_ap_score", 0.0)), _safe_float(d.get("best_gt_iou", 0.0))))
        if winner.get("_ap_label") is None:
            winner["_ap_label"] = "TP"
            winner["_ap_reason"] = "matched_gt_highest_confidence"
            tp_by_gi[gi] = winner

    for d in dets:
        if d.get("_ap_label") == "TP":
            continue
        gi = d.get("_best_gt")
        d_gt_iou = _safe_float(d.get("best_gt_iou", 0.0))
        tp_det = tp_by_gi.get(gi) if gi is not None else None
        if (tp_det is not None and d_gt_iou >= float(duplicate_iou)
                and d_gt_iou <= _safe_float(tp_det.get("best_gt_iou", 0.0))):
            d["_ap_label"] = "IGNORE"
            d["_ap_reason"] = "overlap_ge_ignore_iou_with_found_gt"
        else:
            d["_ap_label"] = "FP"
            d["_ap_reason"] = "no_gt_match_or_lower_confidence_on_same_gt"

    rows: List[Dict[str, Any]] = []
    for det in dets:
        label = str(det.get("_ap_label") or "FP")
        rows.append({
            "technique": technique,
            "technique_label": TECHNIQUE_LABELS.get(technique, technique),
            "image": image_name,
            "label": label,
            "confidence_score": float(_safe_float(det.get("_ap_score", det.get("ml_score", det.get("score", 0.0))))),
            "best_gt_iou": float(_safe_float(det.get("best_gt_iou", 0.0))),
            "matched_gt_index": "" if det.get("_best_gt") is None else int(det.get("_best_gt")),
            "reason": str(det.get("_ap_reason", "")),
            "roi_id": str(det.get("roi_id", "")),
            "seed_roi_id": str(det.get("seed_roi_id", "")),
            "source_roi_ids": str(det.get("source_roi_ids", "")),
            "x": float(_safe_float(det.get("x", 0.0))),
            "y": float(_safe_float(det.get("y", 0.0))),
            "width": float(_safe_float(det.get("width", 0.0))),
            "height": float(_safe_float(det.get("height", 0.0))),
        })
    return rows


def _precision_recall_envelope_from_points(
    points: List[Dict[str, Any]],
    total_gt: int,
) -> Tuple[List[float], List[float], float]:
    """Restituisce la curva PR interpolata standard e la AP.

    Formula usata per object detection:
      1) detection ordinate per confidence score decrescente;
      2) TP/FP cumulativi;
      3) precision envelope monotona: p_interp(r)=max_{r'>=r} p(r');
      4) AP = somma degli incrementi di recall pesati dalla precision interpolata.

    Questo evita curve visivamente rumorose dovute a tanti piccoli FP consecutivi
    e rende il grafico Precision-Recall leggibile e difendibile in relazione.
    """
    if not points or int(total_gt) <= 0:
        return [0.0, 1.0], [0.0, 0.0], 0.0

    recalls = [0.0] + [max(0.0, min(1.0, _safe_float(p.get("recall", 0.0)))) for p in points] + [1.0]
    precisions = [1.0] + [max(0.0, min(1.0, _safe_float(p.get("precision", 0.0)))) for p in points] + [0.0]

    # Precision envelope: la precisione a un certo recall e' la migliore
    # ottenibile da quel recall in poi. E' la forma standard per calcolare AP.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        if recalls[i] > recalls[i - 1]:
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    # Comprimo i punti duplicati sullo stesso valore di recall, tenendo la
    # precisione interpolata massima. Il grafico resta pulito, senza scatter.
    buckets: Dict[float, float] = {}
    for r, p in zip(recalls, precisions):
        key = round(float(r), 12)
        buckets[key] = max(float(p), buckets.get(key, 0.0))
    xs = sorted(buckets.keys())
    ys = [buckets[x] for x in xs]

    # Garantisce il drop finale a precision=0 se la tecnica non raggiunge recall=1.
    if xs and xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(0.0)
    elif xs and xs[-1] == 1.0 and max(_safe_float(p.get("recall", 0.0)) for p in points) < 1.0:
        ys[-1] = 0.0

    return xs, ys, float(max(0.0, min(1.0, ap)))


def _average_precision_from_detection_file(
    detection_rows: List[Dict[str, Any]],
    total_gt: int,
) -> Tuple[List[Dict[str, Any]], float]:
    """Calcola PR e AP ordinando le detection per confidence score decrescente."""
    usable = [r for r in detection_rows if str(r.get("label", "")) in {"TP", "FP"}]
    usable.sort(key=lambda r: float(r.get("confidence_score", 0.0)), reverse=True)

    tp_cum = 0
    fp_cum = 0
    points: List[Dict[str, Any]] = []
    for rank, row in enumerate(usable, start=1):
        if row["label"] == "TP":
            tp_cum += 1
        else:
            fp_cum += 1
        precision = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) else 1.0
        recall = tp_cum / max(1, int(total_gt)) if total_gt else 0.0
        point = dict(row)
        point.update({
            "rank": int(rank),
            "cum_TP": int(tp_cum),
            "cum_FP": int(fp_cum),
            "precision": float(precision),
            "recall": float(recall),
        })
        points.append(point)

    _, _, ap = _precision_recall_envelope_from_points(points, total_gt)
    return points, float(ap)


def _ap_curve_for_technique(
    pipeline_mod: Any,
    image_records: List[Dict[str, Any]],
    technique: str,
    nms_iou: float,
    merge_iou: float,
    tp_iou: float,
    duplicate_iou: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """Costruisce detection file e punti PR/AP usando tutte le finding non sogliate.

    Per la AP si rimuove il thresholding sul confidence score: threshold=0.0.
    Il confronto operativo del progetto resta invece ML >= 0.50.
    """
    detection_rows: List[Dict[str, Any]] = []
    total_gt = sum(len(rec.get("gt_boxes", [])) for rec in image_records)
    for rec in image_records:
        dets = _postprocess(pipeline_mod, rec.get("rois", []), technique, 0.0, nms_iou, merge_iou)
        detection_rows.extend(_ap_match_image_detections(
            dets,
            rec.get("gt_boxes", []),
            str(rec.get("image", "")),
            technique,
            tp_iou,
            duplicate_iou,
        ))
    pr_points, ap = _average_precision_from_detection_file(detection_rows, total_gt)
    return detection_rows, pr_points, ap


# ──────────────────────────────────────────────────────────────────────────────
# Tema grafico chiaro ("white mode") condiviso dalle curve (FROC, Precision-Recall):
# sfondo bianco, accenti leggibili, griglia leggera, testo scuro.
# ──────────────────────────────────────────────────────────────────────────────
_FUT_BG     = "#FFFFFF"   # sfondo figura (bianco)
_FUT_PANEL  = "#FFFFFF"   # sfondo dell'area del grafico
_FUT_GRID   = "#D7E0EA"   # griglia / bordi tenui
_FUT_TEXT   = "#14243A"   # testo principale (blu notte)
_FUT_MUTED  = "#5E7186"   # testo secondario
_FUT_CYAN   = "#0E9BB8"   # accento primario (ciano/teal, leggibile su bianco)
_FUT_GREEN  = "#0E9F6E"   # accento secondario (verde smeraldo)
_FUT_VIOLET = "#7C5CFF"   # accento terziario (viola)


def _style_futuristic(fig, ax, title=None, xlabel=None, ylabel=None, grid_axis="both"):
    """Applica il tema chiaro a una figura/asse matplotlib."""
    fig.patch.set_facecolor(_FUT_BG)
    ax.set_facecolor(_FUT_PANEL)
    for side, spine in ax.spines.items():
        spine.set_color(_FUT_GRID)
        if side in ("top", "right"):
            spine.set_alpha(0.35)
            spine.set_linewidth(0.6)
        else:
            spine.set_linewidth(1.1)
    ax.tick_params(colors=_FUT_MUTED, labelsize=10, length=4)
    ax.grid(False)
    if grid_axis in ("both", "x"):
        ax.xaxis.grid(True, which="major", color=_FUT_GRID, alpha=0.40, linewidth=0.8)
    if grid_axis in ("both", "y"):
        ax.yaxis.grid(True, which="major", color=_FUT_GRID, alpha=0.40, linewidth=0.8)
        ax.yaxis.grid(True, which="minor", color=_FUT_GRID, alpha=0.16, linewidth=0.5)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=_FUT_TEXT, fontsize=12.5, fontweight="bold", pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=_FUT_MUTED, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=_FUT_MUTED, fontsize=11)


def _style_legend(ax, **kwargs):
    """Legenda coerente col tema chiaro."""
    leg = ax.legend(facecolor=_FUT_PANEL, edgecolor=_FUT_GRID, framealpha=0.92, **kwargs)
    if leg is not None:
        leg.get_frame().set_linewidth(0.8)
        for txt in leg.get_texts():
            txt.set_color(_FUT_TEXT)
    return leg


def _plot_average_precision_curves(
    ap_points_by_technique: Dict[str, List[Dict[str, Any]]],
    ap_by_technique: Dict[str, float],
    curves_dir: Path,
    show_plots: bool,
    output_name: str = "precision_recall_nms_only.png",
) -> Path:
    """Disegna una Precision-Recall pulita e standard per object detection.

    La curva viene costruita sul detection file ordinato per confidence score.
    Non uso smoothing artificiale: uso la precision envelope interpolata standard,
    cioe' la stessa curva usata per calcolare l'Average Precision.
    """
    labels_short = {
        NMS_TECHNIQUE: "NMS",
    }
    fig = plt.figure(figsize=(9.4, 7.4))
    _draw_pr(plt.gca(), ap_points_by_technique, ap_by_technique)

    ap_line = "   |   ".join(
        f"AP {labels_short.get(t, t)} = {float(ap_by_technique.get(t, 0.0)):.4f}"
        for t in TECHNIQUES
    )
    plt.figtext(
        0.5, 0.018,
        ap_line + "   ·   AP = area sotto la curva Precision-Recall",
        ha="center", va="bottom", fontsize=9.5, color=_FUT_TEXT,
        bbox={"boxstyle": "round,pad=0.42", "facecolor": _FUT_PANEL, "edgecolor": _FUT_CYAN, "alpha": 0.9},
    )
    plt.tight_layout(rect=(0, 0.075, 1, 1))
    out = curves_dir / output_name
    plt.savefig(out, dpi=220, facecolor=_FUT_BG)

    # NB: non viene piu' generato l'alias precision_recall_comparison.png.
    # Quei vecchi grafici "comparison" potevano contenere prove Pre-processing/Merge:
    # in modalita' NMS-only esiste solo precision_recall_nms_only.png.
    if show_plots:
        plt.show(block=False)
    else:
        plt.close()
    return out


# Punti FP/img fissi usati per leggere la sensibilita' interpolata (es. a 1 FP/img).
# Restano entro 0-5 FP/immagine: oltre 5 FP/img la curva non viene piu' valutata.
FROC_FPPI_POINTS: Tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
FROC_DB_FLOOR = -40.0


def _ratio_to_db(value: float, floor_db: float = FROC_DB_FLOOR) -> float:
    value = max(0.0, min(1.0, _safe_float(value, 0.0)))
    if value <= 0.0:
        return float(floor_db)
    return max(float(floor_db), float(10.0 * np.log10(value)))


def _froc_curve(rows: List[Dict[str, Any]]) -> Tuple[List[float], List[float], Dict[float, float]]:
    """Curva FROC (FP/img vs sensibilita').

    Ritorna (fppi, recall, sensibilita_per_punto), dove sensibilita_per_punto e' la
    sensibilita' interpolata ai punti FP/img fissi FROC_FPPI_POINTS (usata ad es. per
    leggere la sensibilita' a 1 FP/img).
    """
    fppi, recall = _aggregate_max_y_by_x([r["fp_per_image"] for r in rows], [r["recall"] for r in rows])
    if not fppi:
        return [], [], {}
    # inviluppo monotono: sensibilita' migliore ottenibile entro quel budget FP/image.
    for i in range(1, len(recall)):
        recall[i] = max(recall[i], recall[i - 1])
    if fppi[0] > 0:
        fppi = [0.0] + fppi
        recall = [0.0] + recall
    xs = np.asarray(fppi, dtype=float)
    ys = np.asarray(recall, dtype=float)
    pts = np.asarray(FROC_FPPI_POINTS, dtype=float)
    # A sinistra del primo punto reale la sensibilita' e' 0; oltre l'ultimo punto
    # osservato resta sull'ultimo valore (piu' FP non possono ridurre la sensibilita').
    sens = np.interp(pts, xs, ys, left=0.0, right=float(ys[-1]) if ys.size else 0.0)
    sens_at = {float(p): float(s) for p, s in zip(FROC_FPPI_POINTS, sens)}
    return fppi, recall, sens_at


# NB: la curva ROC e' stata rimossa (non viene piu' generata ne' mostrata).
# Restano solo FROC e Precision-Recall.


def _draw_froc(ax, curve_rows_by_technique):
    """Disegna la FROC su `ax` con l'asse Y (sensibilita') in SCALA LOGARITMICA e
    asse X (FP/img) lineare 0-5. Stessa resa per il PNG di evaluate e per il grafico
    interattivo della GUI. Ritorna la sensibilita' a 1 FP/img per tecnica."""
    from matplotlib import ticker as _mticker
    colors = {NMS_TECHNIQUE: _FUT_CYAN}
    labels = TECHNIQUE_LABELS
    fp_axis_max = 5.0
    at1_by_tech = {}
    y_min = 1.0
    for tech, rows in curve_rows_by_technique.items():
        x, y, sens_at = _froc_curve(rows)
        at1_by_tech[tech] = float(sens_at.get(1.0, 0.0))
        if not x:
            continue
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        # La scala log non ammette sensibilita' = 0: tengo i punti > 0 entro 0-5 FP/img.
        mask = (xa <= fp_axis_max) & (ya > 0)
        if not mask.any():
            continue
        y_min = min(y_min, float(ya[mask].min()))
        col = colors.get(tech, _FUT_CYAN)
        ax.fill_between(xa[mask], ya[mask], 1e-4, color=col, alpha=0.14, zorder=1)
        ax.plot(xa[mask], ya[mask], color=col, linewidth=2.6,
                label=f"{labels.get(tech, tech)}", zorder=3, solid_capstyle="round")
    ax.set_yscale("log")
    # Limite inferiore adattivo, appena sotto la sensibilita' minima osservata.
    y_lower = max(0.01, y_min * 0.8)
    ax.set_ylim(y_lower, 1.05)
    ax.set_xlim(0.0, fp_axis_max)
    # Tacche Y leggibili in percentuale (5,10,20,30,50,70,100%) entro il range.
    y_ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0) if t >= y_lower * 0.99]
    ax.yaxis.set_major_locator(_mticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(_mticker.FuncFormatter(lambda v, _pos: f"{v * 100:.0f}%"))
    ax.yaxis.set_minor_formatter(_mticker.NullFormatter())
    _style_futuristic(ax.figure, ax,
                      title="FROC · sensibilità in scala logaritmica",
                      xlabel="Falsi positivi per immagine (FP/img, 0-5)",
                      ylabel="Sensibilità / Recall (scala log)")
    _style_legend(ax, loc="lower right", fontsize=10)
    return at1_by_tech


def _draw_pr(ax, ap_points_by_technique, ap_by_technique):
    """Disegna la curva Precision-Recall su `ax`. Ritorna True se c'e' almeno una curva."""
    colors = {NMS_TECHNIQUE: _FUT_CYAN}
    labels_short = {NMS_TECHNIQUE: "NMS"}
    any_curve = False
    for tech in TECHNIQUES:
        pts = ap_points_by_technique.get(tech, [])
        if not pts:
            continue
        max_tp = max((int(_safe_float(p.get("cum_TP", 0))) for p in pts), default=0)
        max_recall = max((_safe_float(p.get("recall", 0.0)) for p in pts), default=0.0)
        total_gt = int(round(max_tp / max_recall)) if max_recall > 0 else max_tp
        recall, precision, ap_value = _precision_recall_envelope_from_points(pts, total_gt)
        ap_value = float(ap_by_technique.get(tech, ap_value))
        any_curve = True
        col = colors.get(tech, _FUT_CYAN)
        ax.fill_between(recall, precision, 0, color=col, alpha=0.14, zorder=1)
        ax.plot(recall, precision, color=col, linewidth=2.6,
                label=f"{labels_short.get(tech, tech)} · AP={ap_value:.4f}", zorder=3, solid_capstyle="round")
    _style_futuristic(ax.figure, ax,
                      title="Curva Precision–Recall · soglia ML variabile lungo la curva",
                      xlabel="Recall / Sensibilità", ylabel="Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    if any_curve:
        _style_legend(ax, loc="lower left")
    else:
        ax.text(0.5, 0.5, "Nessun punto PR disponibile", ha="center", va="center", fontsize=12, color=_FUT_MUTED)
    return any_curve


def _plot_curves(curve_rows_by_technique: Dict[str, List[Dict[str, Any]]], curves_dir: Path, show_plots: bool) -> Dict[str, Dict[str, float]]:
    curves_dir.mkdir(parents=True, exist_ok=True)
    colors = {NMS_TECHNIQUE: _FUT_CYAN}
    labels = TECHNIQUE_LABELS
    areas: Dict[str, Dict[str, float]] = {"PR": {}, "FROC": {}}

    # La curva Precision-Recall ufficiale e' quella di _plot_average_precision_curves
    # (AP sul detection file), disegnata subito dopo in main(). Qui NON la ridisegno,
    # cosi' esce una sola curva PR e non due (sweep + AP).

    # FROC (asse Y in dB) — stessa resa usata dal grafico interattivo della GUI.
    fig = plt.figure(figsize=(9.2, 6.6))
    areas["FROC_AT_1FPPI"] = _draw_froc(plt.gca(), curve_rows_by_technique)
    plt.tight_layout()
    froc_png = curves_dir / "froc_nms_only.png"
    plt.savefig(froc_png, dpi=200, facecolor=_FUT_BG)
    if show_plots:
        plt.show(block=False)
    else:
        plt.close()

    # La curva ROC e' stata rimossa: restano solo FROC e Precision-Recall.
    return areas


def _plot_froc_comparison(curve_rows_by_nms, summary_by_nms, curves_dir, show_plots, best_nms=None):
    """Disegna in UNA sola figura le FROC di piu' soglie NMS (confronto su validation).
    Ogni curva e' etichettata con il proprio F1 e CPM; il vincitore e' evidenziato (★)."""
    from matplotlib import ticker as _mticker
    palette = [_FUT_CYAN, _FUT_GREEN, _FUT_VIOLET, "#FF8A5B", "#E05780"]
    fig = plt.figure(figsize=(9.4, 6.8))
    ax = plt.gca()
    y_min = 1.0
    for i, nms in enumerate(sorted(curve_rows_by_nms.keys())):
        x, y, sens_at = _froc_curve(curve_rows_by_nms[nms])
        if not x:
            continue
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        mask = (xa <= 5.0) & (ya > 0)
        if not mask.any():
            continue
        y_min = min(y_min, float(ya[mask].min()))
        col = palette[i % len(palette)]
        cpm = float(np.mean([sens_at.get(float(p), 0.0) for p in FROC_FPPI_POINTS]))
        f1 = _safe_float(summary_by_nms.get(nms, {}).get("F1", 0.0))
        is_best = best_nms is not None and abs(float(nms) - float(best_nms)) < 1e-9
        ax.plot(xa[mask], ya[mask], color=col, linewidth=3.0 if is_best else 2.2,
                label=f"NMS {nms:.2f} · F1={f1:.3f} · CPM={cpm:.3f}" + ("  ★ scelto" if is_best else ""),
                zorder=4 if is_best else 3, solid_capstyle="round")
    ax.set_yscale("log")
    y_lower = max(0.01, y_min * 0.8)
    ax.set_ylim(y_lower, 1.05)
    ax.set_xlim(0.0, 5.0)
    y_ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0) if t >= y_lower * 0.99]
    ax.yaxis.set_major_locator(_mticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(_mticker.FuncFormatter(lambda v, _pos: f"{v * 100:.0f}%"))
    ax.yaxis.set_minor_formatter(_mticker.NullFormatter())
    _style_futuristic(fig, ax,
                      title="FROC validation · confronto soglie NMS (vincitore per F1)",
                      xlabel="Falsi positivi per immagine (FP/img, 0-5)",
                      ylabel="Sensibilità / Recall (scala log)")
    _style_legend(ax, loc="lower right", fontsize=9.5)
    plt.tight_layout()
    out = curves_dir / "froc_validation_nms_comparison.png"
    plt.savefig(out, dpi=200, facecolor=_FUT_BG)
    if show_plots:
        plt.show(block=False)
    else:
        plt.close()
    return out


def _select_best_nms(summary_by_nms, recall_tol=0.02):
    """Sceglie la soglia NMS con F1 piu' alta, MA solo tra quelle che non perdono
    recall oltre `recall_tol` rispetto al recall massimo osservato (salvaguardia
    clinica: non sacrifichiamo fratture per un guadagno di precision marginale).
    A parita' di F1 preferisce il recall piu' alto.
    """
    if not summary_by_nms:
        return None
    max_recall = max(_safe_float(s.get("recall", 0.0)) for s in summary_by_nms.values())
    eligible = {nms: s for nms, s in summary_by_nms.items()
                if _safe_float(s.get("recall", 0.0)) >= max_recall - recall_tol}
    if not eligible:
        eligible = summary_by_nms
    return max(eligible.items(),
               key=lambda kv: (_safe_float(kv[1].get("F1", 0.0)), _safe_float(kv[1].get("recall", 0.0))))[0]


# NB: il grafico P/R/F1 (_plot_operating_metrics_comparison) e' stato rimosso.
# Precision, Recall e F1 alla soglia operativa si leggono ora direttamente nei
# chip delle metriche della GUI, accanto a TP/FP/FN. I dati restano comunque nel
# CSV operating_metrics_nms_only.csv per consultazione/tesi.




def _timestamped_sibling(path: Path, reason: str = "locked") -> Path:
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_reason = re.sub(r"[^A-Za-z0-9_]+", "_", str(reason or "locked")).strip("_") or "locked"
    return path.with_name(f"{path.stem}_{safe_reason}_{stamp}{path.suffix}")


def _safe_write_text_file(path: Path, text: str, encoding: str = "utf-8") -> Path:
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


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> Path:
    """Scrive un CSV; se e' aperto/bloccato, crea una copia timestampata."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    target = out_csv
    if not rows:
        return _safe_write_text_file(target, "", encoding="utf-8")
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    try:
        f = target.open("w", encoding="utf-8", newline="")
    except PermissionError:
        target = _timestamped_sibling(out_csv, "locked")
        print(f"[WARN] CSV bloccato o aperto: {out_csv}")
        print(f"[WARN] Salvo una nuova copia in: {target}")
        f = target.open("w", encoding="utf-8", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    return target


def _save_summary(path: Path, data: Dict[str, Any]) -> Path:
    lines = []
    for k, v in data.items():
        lines.append(f"{k}: {v}\n")
    return _safe_write_text_file(path, "".join(lines), encoding="utf-8")


def _best_row(curve_rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """Sceglie il miglior PUNTO OPERATIVO, non la tecnica con area piu' grande.

    In modalita' NMS-only viene fatto lo sweep reale della soglia ML. Poi si prende
    il singolo punto NMS con metrica massima. Il criterio ammesso e' solo F1 o F2:
    NON si sceglie il punto operativo su sola Precision o solo Recall (sarebbe
    banalmente massimizzabile). L'Average Precision resta la metrica di sintesi
    delle curve.
    Recall e F1/F2 sono calcolate sul totale delle GT del test set: quindi una
    tecnica che non arriva a recall=1.0 viene penalizzata automaticamente.
    Le aree PR/FROC sono salvate e disegnate, ma non decidono il best JSON.
    """
    metric = metric.upper()
    if metric not in {"F1", "F2"}:
        metric = "F1"

    def key(r: Dict[str, Any]) -> Tuple[float, float, float, int, int, int, int]:
        tech = str(r.get("technique", ""))
        primary = _safe_float(r.get(metric, 0.0))
        return (
            primary,
            _safe_float(r.get("F1", 0.0)),
            _safe_float(r.get("precision", 0.0)),
            -int(r.get("FP", 0)),
            -int(r.get("IGNORED", 0)),
            -int(r.get("detections", 0)),
            TECHNIQUE_TIE_PREFERENCE.get(tech, 0),
        )

    return max(curve_rows, key=key)


def _aggregate_fixed(curve_rows_by_technique: Dict[str, List[Dict[str, Any]]], operating_threshold: float) -> Dict[str, Dict[str, Any]]:
    fixed: Dict[str, Dict[str, Any]] = {}
    for tech, rows in curve_rows_by_technique.items():
        # Prendo il punto piu' vicino alla soglia operativa richiesta.
        if not rows:
            continue
        fixed[tech] = min(rows, key=lambda r: abs(float(r.get("threshold", 0.0)) - operating_threshold))
    return fixed


def _collect_image_records(pipeline_mod, images, labels_dir, gt_class):
    """Esegue l'inferenza della pipeline su una lista di immagini e raccoglie
    (image_records, all_scores). Usata per lo sweep NMS sulla validation."""
    records: List[Dict[str, Any]] = []
    scores: List[float] = []
    for img_path in images:
        try:
            gt_boxes, label_src = _load_gt_boxes_for_image(pipeline_mod, img_path, str(labels_dir), gt_class)
            data = pipeline_mod.run_pipeline_inference(img_path)
            if not isinstance(data, dict):
                continue
            rois = data.get("rois", []) or []
            for r in rois:
                scores.append(_safe_score(r))
            records.append({"image": img_path.name, "path": str(img_path),
                            "label_file": label_src, "rois": rois, "gt_boxes": gt_boxes})
        except Exception as exc:
            print(f"    [WARN validation] {img_path.name}: {exc}")
    return records, scores


def _auto_validation_nms_sweep(pipeline_mod, args, project_root, curves_dir, labels_dir, sweep_vals):
    """Sweep NMS AUTOMATICO sul set di VALIDATION (auto-trovato).

    Per ogni soglia NMS calcola la FROC + metriche, sceglie il vincitore per F1 (con
    salvaguardia sul recall), salva la figura di confronto e i punti per la GUI.
    Ritorna (nms_selected, summary). Se la validation non c'e', ritorna (None, {}).
    """
    try:
        val_manifest = _resolve_project_path(args.validation_manifest, project_root, kind="file")
    except Exception:
        val_manifest = None
    try:
        val_images_dir = _resolve_project_path(args.validation_images, project_root, kind="dir")
    except Exception:
        val_images_dir = None
    images = _read_manifest_images(val_manifest, project_root) if val_manifest else []
    if not images and val_images_dir:
        images = _collect_images(val_images_dir)
    if not images:
        print(f"[WARN] Sweep NMS automatico saltato: set di validation non trovato ({args.validation_manifest}).")
        return None, {}
    print(f"\n[*] Sweep NMS AUTOMATICO su VALIDATION: {len(images)} immagini · soglie {sweep_vals}.")
    records, scores = _collect_image_records(pipeline_mod, images, labels_dir, args.gt_class)
    if not records or not scores:
        print("[WARN] Sweep NMS saltato: nessuna ROI/score sulla validation.")
        return None, {}
    thresholds = _threshold_grid(scores, max_points=args.max_curve_points)
    comp_rows_by_nms: Dict[float, List[Dict[str, Any]]] = {}
    summary: Dict[float, Dict[str, float]] = {}
    for nms in sweep_vals:
        rows = _curve_points_for_technique(
            pipeline_mod=pipeline_mod, image_records=records, thresholds=thresholds,
            technique=NMS_TECHNIQUE, nms_iou=float(nms), merge_iou=float(args.merge_iou),
            tp_iou=float(args.match_iou), ignore_iou=float(args.ignore_iou),
        )
        comp_rows_by_nms[nms] = rows
        fx = _aggregate_fixed({NMS_TECHNIQUE: rows}, float(args.operating_threshold)).get(NMS_TECHNIQUE, {})
        summary[nms] = {
            "F1": _safe_float(fx.get("F1", 0.0)), "F2": _safe_float(fx.get("F2", 0.0)),
            "precision": _safe_float(fx.get("precision", 0.0)), "recall": _safe_float(fx.get("recall", 0.0)),
        }
        s = summary[nms]
        print(f"    NMS {nms:.2f}: F1={s['F1']:.4f}  P={s['precision']*100:6.2f}%  R={s['recall']*100:6.2f}%  F2={s['F2']:.4f}")
    winner = _select_best_nms(summary)
    print(f"[*] NMS vincente (validation, F1 con salvaguardia recall): {winner:.2f}")
    _plot_froc_comparison(comp_rows_by_nms, summary, curves_dir,
                          show_plots=not args.no_show_plots, best_nms=winner)
    sweep_points_rows: List[Dict[str, Any]] = []
    for nms, rows in comp_rows_by_nms.items():
        f1_nms = _safe_float(summary.get(nms, {}).get("F1", 0.0))
        is_win = winner is not None and abs(float(nms) - float(winner)) < 1e-9
        for r in rows:
            rr = dict(r)
            rr["nms"] = float(nms)
            rr["nms_f1"] = float(f1_nms)
            rr["nms_selected"] = int(is_win)
            sweep_points_rows.append(rr)
    _write_csv(sweep_points_rows, curves_dir / "froc_validation_sweep_points.csv")
    return winner, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta post-processing classe 3 usando solo NMS e genera curve con sweep soglia ML.")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES_DIR), help="Cartella immagini di test")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Manifest test generato dal training")
    parser.add_argument("--labels", default=str(DEFAULT_LABEL_DIR), help="Cartella label YOLO")
    parser.add_argument("--pipeline-file", default=None, help="Path esplicito a pipeline_gui.py")
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV), help="CSV dettaglio fixed-threshold per immagine")
    parser.add_argument("--curves-dir", default=str(DEFAULT_CURVES_DIR), help="Cartella output curve")
    parser.add_argument("--gt-class", type=int, default=3, help="Classe YOLO frattura")
    parser.add_argument("--match-iou", type=float, default=0.50, help="TP se IoU >= match_iou")
    parser.add_argument("--ignore-iou", type=float, default=0.30, help="IGNORE duplicati: una detection non-TP che si sovrappone alla GT gia' trovata (IoU con la GT >= ignore_iou) e con IoU sulla GT inferiore a quella della TP non conta come FP")
    parser.add_argument("--nms-iou", type=float, default=0.50, help="IoU NMS")
    parser.add_argument("--nms-sweep", default="0.30,0.40,0.50", help="Soglie NMS confrontate AUTOMATICAMENTE sulla validation (default '0.30,0.40,0.50'): sceglie il vincitore per F1 (salvaguardia recall) e lo usa per la valutazione del test. Metti '' per disattivare.")
    parser.add_argument("--validation-manifest", default=str(DEFAULT_VAL_MANIFEST), help="Manifest del set di validation per lo sweep NMS automatico.")
    parser.add_argument("--validation-images", default=str(DEFAULT_VAL_IMAGES_DIR), help="Cartella immagini di validation per lo sweep NMS automatico.")
    parser.add_argument("--merge-iou", type=float, default=0.50, help=argparse.SUPPRESS)  # compatibilita' CLI: non usato in modalita' NMS-only
    parser.add_argument("--operating-threshold", type=float, default=0.50, help="Soglia ML operativa fissa. Default: 0.50; si considerano ROI con ML >= soglia")
    parser.add_argument("--best-metric", choices=["f1", "f2"], default="f1", help="Criterio del punto operativo: solo F1 o F2 (no precision/recall da soli; l'AP resta la metrica di sintesi delle curve)")
    parser.add_argument("--max-curve-points", type=int, default=600, help="Numero massimo punti soglia")
    parser.add_argument("--no-show-plots", action="store_true", help="Salva i grafici senza aprire finestre")
    parser.add_argument("--open-folder", action="store_true", help="Apre la cartella output alla fine")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    images_dir = _resolve_project_path(args.images, project_root, kind="dir")
    manifest_path = _resolve_project_path(args.manifest, project_root, kind="file")
    labels_dir = _resolve_project_path(args.labels, project_root, kind="dir")
    out_path = _resolve_project_path(args.out, project_root, kind="output")
    curves_dir = _resolve_project_path(args.curves_dir, project_root, kind="output")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    os.environ["IPA_GT_LABELS_DIR"] = str(labels_dir.resolve())
    print(f"[INFO] Uso label da: {labels_dir.resolve()}")
    pipeline_mod, pipeline_path = _load_pipeline_module(project_root, args.pipeline_file)
    print(f"[INFO] Pipeline caricata da: {pipeline_path}")
    if not hasattr(pipeline_mod, "run_pipeline_inference"):
        raise AttributeError(f"Il file {pipeline_path} non contiene run_pipeline_inference().")

    # ── Sweep NMS AUTOMATICO sulla VALIDATION (sempre attivo, salvo --nms-sweep "") ──
    # Sceglie l'NMS migliore per F1 sulla validation e lo usa per la valutazione del
    # test qui sotto. Cosi' un singolo lancio fa: sweep validation → vincitore → test.
    nms_selected = None
    nms_sweep_summary: Dict[float, Dict[str, float]] = {}
    sweep_vals = []
    for v in str(getattr(args, "nms_sweep", "") or "").replace(";", ",").split(","):
        v = v.strip()
        if v:
            try:
                sweep_vals.append(round(float(v), 4))
            except ValueError:
                pass
    if sweep_vals:
        try:
            nms_selected, nms_sweep_summary = _auto_validation_nms_sweep(
                pipeline_mod, args, project_root, curves_dir, labels_dir, sweep_vals)
            if nms_selected is not None:
                args.nms_iou = float(nms_selected)
                print(f"[*] La valutazione del test userà l'NMS vincente: {nms_selected:.2f}\n")
        except Exception as exc:
            print(f"[WARN] Sweep NMS automatico non riuscito: {exc}")

    images = _read_manifest_images(manifest_path, project_root)
    if images:
        print(f"[INFO] Immagini test lette dal manifest: {len(images)} ({manifest_path})")
    else:
        images = _collect_images(images_dir)
        print(f"[WARN] Manifest non usato/trovato. Immagini lette dalla cartella: {len(images)} ({images_dir})")

    image_records: List[Dict[str, Any]] = []
    all_scores: List[float] = []
    per_image_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    missing_label_count = 0

    for idx, img_path in enumerate(images, start=1):
        print(f"\n[{idx}/{len(images)}] {img_path.name}")
        try:
            gt_boxes, label_src = _load_gt_boxes_for_image(pipeline_mod, img_path, str(labels_dir), args.gt_class)
            if label_src:
                print(f"    [GT] {Path(label_src).name} | box classe {args.gt_class}: {len(gt_boxes)}")
            else:
                missing_label_count += 1
                print(f"    [GT WARN] Nessun file label trovato.")

            data = pipeline_mod.run_pipeline_inference(img_path)
            if not isinstance(data, dict):
                raise RuntimeError("run_pipeline_inference non ha restituito un dict.")
            rois = data.get("rois", []) or []
            for r in rois:
                all_scores.append(_safe_score(r))
            image_records.append({"image": img_path.name, "path": str(img_path), "label_file": label_src, "rois": rois, "gt_boxes": gt_boxes})

            # Riga di dettaglio alla soglia operativa per ogni tecnica.
            for tech in TECHNIQUES:
                dets = _postprocess(pipeline_mod, rois, tech, args.operating_threshold, args.nms_iou, args.merge_iou)
                m = _evaluate(pipeline_mod, dets, gt_boxes, args.match_iou, args.ignore_iou)
                per_image_rows.append({
                    "image": img_path.name,
                    "path": str(img_path),
                    "label_file": label_src,
                    "technique": tech,
                    "technique_label": TECHNIQUE_LABELS[tech],
                    "threshold": args.operating_threshold,
                    "TP": int(m.get("TP", 0)),
                    "FP": int(m.get("FP", 0)),
                    "FN": int(m.get("FN", 0)),
                    "IGNORED": int(m.get("IGNORED", 0)),
                    "GT_class3": int(m.get("GT", len(gt_boxes))),
                    "detections": int(m.get("detections", len(dets))),
                    "precision": _safe_float(m.get("precision", 0.0)),
                    "recall": _safe_float(m.get("recall", 0.0)),
                    "F1": _safe_float(m.get("F1", 0.0)),
                    "F2": _safe_float(m.get("F2", 0.0)),
                    "error": "",
                })
            print(f"    ROI={len(rois)} GT={len(gt_boxes)} score range: " + (f"{min(_safe_score(r) for r in rois):.3f}-{max(_safe_score(r) for r in rois):.3f}" if rois else "vuoto"))
        except Exception as exc:
            failures.append(f"{img_path.name}: {exc}")
            per_image_rows.append({"image": img_path.name, "path": str(img_path), "technique": "", "error": str(exc)})
            print(f"    [ERRORE] {exc}")

    out_path = _write_csv(per_image_rows, out_path)

    total_gt = sum(len(rec.get("gt_boxes", [])) for rec in image_records)
    print("\n" + "=" * 78)
    print("RISULTATI GLOBALI NMS SUL TEST SET")
    print("=" * 78)
    print(f"Immagini elaborate:       {len(image_records)} / {len(images)}")
    print(f"Label file mancanti:      {missing_label_count}")
    print(f"Ground truth classe {args.gt_class}:    {total_gt}")
    print(f"Regola TP:                IoU >= {args.match_iou:.2f}")
    print(f"CSV dettaglio:            {out_path}")

    if not image_records or not all_scores:
        print("[WARN] Curve non generate: nessuna immagine valida oppure nessuno score ML disponibile.")
        return 2 if failures else 0

    thresholds = _threshold_grid(all_scores, max_points=args.max_curve_points)
    print(f"\n[*] Calcolo curve con sweep REALE delle soglie ML: {len(thresholds)} soglie.")
    # (Lo sweep NMS e' gia' stato eseguito sulla validation a inizio main: l'NMS
    #  vincente e' in args.nms_iou e usato qui sotto per la valutazione del test.)

    curve_rows_by_technique: Dict[str, List[Dict[str, Any]]] = {}
    all_curve_rows: List[Dict[str, Any]] = []
    for tech in TECHNIQUES:
        print(f"    - {TECHNIQUE_LABELS[tech]}")
        rows = _curve_points_for_technique(
            pipeline_mod=pipeline_mod,
            image_records=image_records,
            thresholds=thresholds,
            technique=tech,
            nms_iou=float(args.nms_iou),
            merge_iou=float(args.merge_iou),
            tp_iou=float(args.match_iou),
            ignore_iou=float(args.ignore_iou),
        )
        curve_rows_by_technique[tech] = rows
        all_curve_rows.extend(rows)

    curve_points_csv = curves_dir / "curve_points_nms_only.csv"
    curve_points_csv = _write_csv(all_curve_rows, curve_points_csv)
    areas = _plot_curves(curve_rows_by_technique, curves_dir, show_plots=not args.no_show_plots)

    # Average Precision secondo le formule del detection file:
    # uso tutte le finding non sogliate, le ordino per confidence score e calcolo
    # TP/FP cumulativi. Questo NON cambia il confronto operativo ML >= 0.50.
    ap_detection_rows: List[Dict[str, Any]] = []
    ap_curve_rows: List[Dict[str, Any]] = []
    ap_points_by_technique: Dict[str, List[Dict[str, Any]]] = {}
    ap_by_technique: Dict[str, float] = {}
    for tech in TECHNIQUES:
        det_rows, pr_rows, ap_value = _ap_curve_for_technique(
            pipeline_mod=pipeline_mod,
            image_records=image_records,
            technique=tech,
            nms_iou=float(args.nms_iou),
            merge_iou=float(args.merge_iou),
            tp_iou=float(args.match_iou),
            duplicate_iou=float(args.ignore_iou),
        )
        ap_detection_rows.extend(det_rows)
        ap_curve_rows.extend(pr_rows)
        ap_points_by_technique[tech] = pr_rows
        ap_by_technique[tech] = float(ap_value)
    ap_detection_csv = curves_dir / "ap_detection_file_nms_only.csv"
    ap_points_csv = curves_dir / "ap_precision_recall_points_nms_only.csv"
    ap_detection_csv = _write_csv(ap_detection_rows, ap_detection_csv)
    ap_points_csv = _write_csv(ap_curve_rows, ap_points_csv)

    # AP e' l'area sotto la curva Precision-Recall: non salvo un grafico AP separato.
    # Sovrascrivo/aggiorno il PNG PR con la curva usata per AP, cosi' la GUI mostra
    # un solo grafico PR con il valore AP in legenda.
    pr_png = _plot_average_precision_curves(
        ap_points_by_technique,
        ap_by_technique,
        curves_dir,
        show_plots=not args.no_show_plots,
        output_name="precision_recall_nms_only.png",
    )
    areas["PR"] = {tech: float(ap_by_technique.get(tech, 0.0)) for tech in TECHNIQUES}

    # Confronto fisso a soglia 0.5 o soglia richiesta.
    fixed = _aggregate_fixed(curve_rows_by_technique, float(args.operating_threshold))
    operating_metrics_csv = curves_dir / "operating_metrics_nms_only.csv"
    operating_metrics_csv = _write_csv([fixed[t] for t in TECHNIQUES if t in fixed], operating_metrics_csv)
    # Il grafico P/R/F1 e' stato rimosso: l'F1 si legge ora direttamente nei chip
    # delle metriche della GUI, accanto a TP/FP/FN/Precision/Recall.
    metrics_png = ""
    print("\n" + "=" * 78)
    print(f"NMS A SOGLIA OPERATIVA {args.operating_threshold:.3f}")
    print("=" * 78)
    for tech in TECHNIQUES:
        r = fixed.get(tech, {})
        print(
            f"{TECHNIQUE_LABELS[tech]:34s} "
            f"DET={int(r.get('detections', 0)):4d} TP={int(r.get('TP', 0)):3d} FP={int(r.get('FP', 0)):3d} "
            f"IGN={int(r.get('IGNORED', 0)):3d} FN={int(r.get('FN', 0)):3d} "
            f"P={_safe_float(r.get('precision', 0))*100:6.2f}% R={_safe_float(r.get('recall', 0))*100:6.2f}% "
            f"F1={_safe_float(r.get('F1', 0)):.4f} F2={_safe_float(r.get('F2', 0)):.4f}"
        )

    print("\n" + "=" * 78)
    print("AVERAGE PRECISION SECONDO DETECTION FILE ORDINATO PER CONFIDENCE")
    print("=" * 78)
    for tech in TECHNIQUES:
        print(f"{TECHNIQUE_LABELS[tech]:34s} AP={ap_by_technique.get(tech, 0.0):.4f}")

    # Se questa run NON ha fatto lo sweep (es. valutazione finale sul test con il
    # vincitore), conservo l'NMS vincente gia' scelto su validation, cosi' la GUI
    # continua a mostrarlo.
    if nms_selected is None:
        try:
            prev = json.loads((curves_dir / "best_postprocess_params.json").read_text(encoding="utf-8"))
            if isinstance(prev, dict) and prev.get("nms_selected") is not None:
                nms_selected = float(prev["nms_selected"])
        except Exception:
            pass

    # Scelta migliore SOLO alla soglia operativa fissa ML >= 0.50.
    # Le curve continuano a essere salvate come analisi, ma NON decidono la soglia GUI.
    fixed_rows = [fixed[t] for t in TECHNIQUES if t in fixed]
    best = _best_row(fixed_rows, args.best_metric)
    best_tech = str(best["technique"])
    best_json = {
        "technique": best_tech,
        "technique_label": TECHNIQUE_LABELS.get(best_tech, best_tech),
        "ml_threshold": float(args.operating_threshold),
        "ml_rule": f"ML >= {float(args.operating_threshold):.2f}",
        "nms_iou": float(args.nms_iou),
        # NMS scelto dallo sweep su validation (per F1, con salvaguardia recall), se attivo.
        "nms_selected": (float(nms_selected) if nms_selected is not None else None),
        "nms_sweep": {f"{k:.2f}": v for k, v in nms_sweep_summary.items()},
        "tp_iou": float(args.match_iou),
        "tp_rule": f"IoU >= {float(args.match_iou):.2f}",
        "ignore_iou": float(args.ignore_iou),
        "selection_metric": args.best_metric.upper(),
        "TP": int(best.get("TP", 0)),
        "FP": int(best.get("FP", 0)),
        "FN": int(best.get("FN", 0)),
        "IGNORED": int(best.get("IGNORED", 0)),
        "GT": int(best.get("GT", 0)),
        "detections": int(best.get("detections", 0)),
        "precision": float(best.get("precision", 0.0)),
        "recall": float(best.get("recall", 0.0)),
        "F1": float(best.get("F1", 0.0)),
        "F2": float(best.get("F2", 0.0)),
        "fp_per_image": float(best.get("fp_per_image", 0.0)),
        "operating_metrics_png": str(metrics_png),
        "operating_metrics_csv": str(operating_metrics_csv),
        "area_pr_threshold_sweep": float(areas.get("PR", {}).get(best_tech, 0.0)),
        "average_precision_detection_file": float(ap_by_technique.get(best_tech, 0.0)),
        "ap_by_technique": {t: float(ap_by_technique.get(t, 0.0)) for t in TECHNIQUES},
        "ap_formula": "AP = sum over recall increments of interpolated precision envelope; detections sorted by confidence score",
        "ap_detection_file_csv": str(ap_detection_csv),
        "ap_precision_recall_points_csv": str(ap_points_csv),
        # AP non ha un secondo grafico: usa lo stesso PNG della curva Precision-Recall.
        "ap_precision_recall_png": str(pr_png),
        "froc_sensitivity_at_1fppi": float(areas.get("FROC_AT_1FPPI", {}).get(best_tech, 0.0)),
        "froc_fppi_points": list(FROC_FPPI_POINTS),
        "froc_formula": "FROC: sensibilita' (asse Y in scala logaritmica) vs falsi positivi per immagine (asse X lineare 0-5)",
        "curve_points_csv": str(curve_points_csv),
        "selection_mode": "nms_only_fixed_ml_gt_0_50",
        "selection_note": "Modalita' NMS-only: la soglia operativa resta ML >= 0.50; Pre-processing e Merge non vengono valutati.",
        "created_by": Path(__file__).name,
    }
    best_json_path = curves_dir / "best_postprocess_params.json"
    best_json_path = _safe_write_text_file(
        best_json_path,
        json.dumps(best_json, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Summary letto dalla GUI.
    summary_path = curves_dir / "global_metrics_nms_only.txt"
    _save_summary(summary_path, {
        "BEST_TECHNIQUE": best_json["technique"],
        "BEST_TECHNIQUE_LABEL": best_json["technique_label"],
        "BEST_ML_THRESHOLD": f"{best_json['ml_threshold']:.6f}",
        "SELECTION_METRIC": best_json["selection_metric"],
        "TP_RULE": best_json["tp_rule"],
        "TP": best_json["TP"],
        "FP": best_json["FP"],
        "FN": best_json["FN"],
        "IGNORED": best_json["IGNORED"],
        "GT": best_json["GT"],
        "DETECTIONS": best_json["detections"],
        "PRECISION_PERCENT": f"{best_json['precision'] * 100:.2f}",
        "RECALL_PERCENT": f"{best_json['recall'] * 100:.2f}",
        "F1": f"{best_json['F1']:.6f}",
        "F2": f"{best_json['F2']:.6f}",
        "NMS_IOU": f"{args.nms_iou:.3f}",
        "IGNORE_IOU": f"{args.ignore_iou:.3f}",
        "METRICS_PNG": str(metrics_png),
        "OPERATING_METRICS_CSV": str(operating_metrics_csv),
        "PR_PNG": str(curves_dir / "precision_recall_nms_only.png"),
        "FROC_PNG": str(curves_dir / "froc_nms_only.png"),
        "AP_PNG": str(pr_png),
        **{f"AVERAGE_PRECISION_DETECTION_FILE_{t}": f"{ap_by_technique.get(t, 0.0):.6f}" for t in TECHNIQUES},
        "AP_FORMULA": "interpolated_precision_envelope",
        "CURVE_POINTS_CSV": str(curve_points_csv),
        "AP_DETECTION_FILE_CSV": str(ap_detection_csv),
        "AP_PR_POINTS_CSV": str(ap_points_csv),
        "BEST_JSON": str(best_json_path),
    })

    # Summary tecnico curve.
    _save_summary(curves_dir / "summary_nms_only.txt", {
        "curve_points_evaluated": len(thresholds),
        "best_metric": args.best_metric.upper(),
        "selection_mode": best_json["selection_mode"],
        "best_technique": best_json["technique"],
        "best_threshold": f"{best_json['ml_threshold']:.6f}",
        "best_precision": f"{best_json['precision']:.6f}",
        "best_recall": f"{best_json['recall']:.6f}",
        "best_F1": f"{best_json['F1']:.6f}",
        "best_F2": f"{best_json['F2']:.6f}",
        "operating_metrics_png": str(metrics_png),
        "operating_metrics_csv": str(operating_metrics_csv),
        **{f"OPERATING_PRECISION_{t}": f"{_safe_float(fixed.get(t, {}).get('precision', 0.0)):.6f}" for t in TECHNIQUES},
        **{f"OPERATING_RECALL_{t}": f"{_safe_float(fixed.get(t, {}).get('recall', 0.0)):.6f}" for t in TECHNIQUES},
        **{f"OPERATING_F1_{t}": f"{_safe_float(fixed.get(t, {}).get('F1', 0.0)):.6f}" for t in TECHNIQUES},
        **{f"AREA_PR_THRESHOLD_SWEEP_{t}": f"{areas.get('PR', {}).get(t, 0.0):.6f}" for t in TECHNIQUES},
        **{f"AVERAGE_PRECISION_DETECTION_FILE_{t}": f"{ap_by_technique.get(t, 0.0):.6f}" for t in TECHNIQUES},
        **{f"FROC_SENS_AT_1FPPI_{t}": f"{areas.get('FROC_AT_1FPPI', {}).get(t, 0.0):.6f}" for t in TECHNIQUES},
    })

    print("\n" + "=" * 78)
    print("PARAMETRI NMS PER LA GUI")
    print("=" * 78)
    print(f"Tecnica scelta:            {best_json['technique']} · {best_json['technique_label']}")
    print(f"Soglia ML operativa:       ML >= {best_json['ml_threshold']:.6f}")
    print(f"Criterio riepilogo:        massimo {best_json['selection_metric']} alla soglia fissa ML >= 0.50")
    print(f"TP={best_json['TP']} FP={best_json['FP']} IGN={best_json['IGNORED']} FN={best_json['FN']} GT={best_json['GT']}")
    print(f"Precision={best_json['precision']*100:.2f}% Recall={best_json['recall']*100:.2f}% F1={best_json['F1']:.4f} F2={best_json['F2']:.4f}")
    print(f"JSON salvato:              {best_json_path}")
    print(f"Curve salvate in:          {curves_dir}")
    print(f"Curva PR/AP per GUI:       {pr_png}")
    print(f"CSV punti curve:           {curve_points_csv}")

    if args.open_folder:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(curves_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(curves_dir)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(curves_dir)])
        except Exception as exc:
            print(f"[WARN] Non riesco ad aprire la cartella dei grafici: {exc}")

    if not args.no_show_plots:
        print("\n[INFO] Chiudi le finestre dei grafici per terminare lo script.")
        plt.show()

    if failures:
        print("\nImmagini fallite:")
        for item in failures[:30]:
            print(f" - {item}")
        if len(failures) > 30:
            print(f" ... altre {len(failures) - 30}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
