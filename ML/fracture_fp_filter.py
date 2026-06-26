#!/usr/bin/env python3
# Build marker: FRACTURE_ML_CSV_FEATURE_TUNING_ENSEMBLE_BUILD_2026_06_23
"""
Filtro ML per eliminare i falsi positivi dalle ROI candidate prodotte dal C++.

Versione multi-metodologia per la relazione, adattata al progetto fratture:
  - confronto tra i 5 CSV feature prodotti dal C++:
    fratture_mancate_FN, GLCM, GLCM+LBP+HOG, HOG, LBP;
  - confronto supervisionato tra i modelli richiesti:
    SVM-RBF, SVM lineare, KNN, Random Forest;
  - tuning automatico degli iper-parametri su validation set;
  - ensemble supervisionati: soft voting e stacking tra i modelli base migliori;
  - confronto opzionale tra normalizzazioni: none, StandardScaler, MinMaxScaler, RobustScaler;
  - PCA opzionale per SVM/KNN;
  - undersampling SOLO sul training set;
  - split train/validation/test a livello di immagine, con controllo anti-leakage;
  - salvataggio del modello finale in fracture_fp_model.pkl compatibile con pipeline/GUI.

Il modello finale resta supervisionato: viene scelto sul validation set e poi valutato
sul test set. Gli esperimenti aggiuntivi servono per documentare il confronto nella relazione.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    from sklearn.svm import SVC
except ImportError as _e:
    raise ImportError("scikit-learn non trovato. Installa con: pip install scikit-learn") from _e


# Regola clinica richiesta per definire una ROI positiva nel dataset ML:
# una ROI e' positiva solo quando la sovrapposizione con la ground truth
# e' maggiore o uguale al 50%. Con IoU == 0.50 la ROI e' TP.
TP_IOU_THRESHOLD: float = 0.50
TP_IOU_RULE: str = ">="

METADATA_COLUMNS: List[str] = [
    "image", "roi_id", "method", "x", "y",
    "width", "height", "score", "roi_file", "best_iou",
    "label_original", "iou_tp_threshold", "iou_tp_rule", "iou_tp_match", "label",
]

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"
)

MODEL_NAME = "BEST_OF_MULTI_METHOD_SUPERVISED_UNDERSAMPLING"
MAX_THRESHOLD_POINTS = 300  # limite rapido per sweep soglie: evita simulazioni troppo lunghe

PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────────────
# Percorsi principali del progetto di Daniela.
# Gli output NON vengono salvati nella cartella degli script .py, ma in una
# cartella dedicata e ordinata: ML/risultati_addestramento.
# ──────────────────────────────────────────────────────────────────────────────
# Radice del progetto: una cartella sopra ML/, rilevata automaticamente dalla
# posizione di questo file. Funziona su qualsiasi computer e sistema operativo.
WINDOWS_PROJECT_ROOT = PROJECT_ROOT.parent
DEFAULT_IMAGE_DIR = WINDOWS_PROJECT_ROOT / "img_fracture"
DEFAULT_LABEL_DIR = DEFAULT_IMAGE_DIR / "labels"

OUTPUT_ROOT = WINDOWS_PROJECT_ROOT / "ML" / "risultati_addestramento"
OUTPUT_MODEL_DIR = OUTPUT_ROOT / "modello"
OUTPUT_REPORT_DIR = OUTPUT_ROOT / "report"
OUTPUT_CSV_DIR = OUTPUT_ROOT / "csv"
OUTPUT_SPLIT_IMAGES_DIR = OUTPUT_ROOT / "split_images"

DEFAULT_MODEL_PATH = OUTPUT_MODEL_DIR / "fracture_fp_model.pkl"
DEFAULT_REPORT_PATH = OUTPUT_REPORT_DIR / "fracture_fp_report.json"
DEFAULT_THRESHOLD_GRID_CSV = OUTPUT_CSV_DIR / "ml_threshold_grid.csv"
DEFAULT_TEST_PREDICTIONS_CSV = OUTPUT_CSV_DIR / "fracture_fp_test_predictions.csv"
DEFAULT_PREDICTIONS_CSV = OUTPUT_CSV_DIR / "fracture_fp_predictions.csv"
DEFAULT_KEPT_CANDIDATES_CSV = OUTPUT_CSV_DIR / "fracture_fp_kept_candidates.csv"
DEFAULT_MODEL_COMPARISON_CSV = OUTPUT_CSV_DIR / "model_comparison.csv"
DEFAULT_EXPERIMENT_COMPARISON_CSV = OUTPUT_CSV_DIR / "experiment_comparison.csv"
DEFAULT_FEATURESET_COMPARISON_CSV = OUTPUT_CSV_DIR / "feature_set_comparison.csv"
DEFAULT_UNSUPERVISED_COMPARISON_CSV = OUTPUT_CSV_DIR / "unsupervised_comparison.csv"


def ensure_output_folders() -> None:
    """Crea tutte le cartelle di salvataggio ordinate."""
    for folder in [OUTPUT_ROOT, OUTPUT_MODEL_DIR, OUTPUT_REPORT_DIR, OUTPUT_CSV_DIR, OUTPUT_SPLIT_IMAGES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)




def _timestamped_sibling(path: Path, reason: str = "locked") -> Path:
    """Crea un nome alternativo quando Windows blocca un file aperto."""
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_reason = re.sub(r"[^A-Za-z0-9_]+", "_", str(reason or "locked")).strip("_") or "locked"
    return path.with_name(f"{path.stem}_{safe_reason}_{stamp}{path.suffix}")


def _safe_to_csv(df: pd.DataFrame, path: Any, **kwargs) -> Path:
    """Salva un CSV senza fermarsi se il file e' aperto in Excel/VS Code.

    Se Windows restituisce PermissionError, il file originale non viene toccato
    e viene creata una copia timestampata nella stessa cartella. In questo modo
    puoi tenere aperto/modificare il CSV mentre lo script continua a funzionare.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        alt = _timestamped_sibling(path, "locked")
        print(f"[WARN] File CSV bloccato o aperto: {path}")
        print(f"[WARN] Salvo una nuova copia modificabile in: {alt}")
        df.to_csv(alt, **kwargs)
        return alt


def _safe_write_text_file(path: Any, text: str, encoding: str = "utf-8") -> Path:
    """Scrive testo/JSON con fallback se il file e' aperto o bloccato."""
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


def _safe_pickle_dump(obj: Any, path: Any) -> Path:
    """Salva un pickle; se il modello e' bloccato, crea una copia timestampata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("wb") as f:
            pickle.dump(obj, f)
        return path
    except PermissionError:
        alt = _timestamped_sibling(path, "locked")
        print(f"[WARN] Modello bloccato o aperto: {path}")
        print(f"[WARN] Salvo una nuova copia in: {alt}")
        with alt.open("wb") as f:
            pickle.dump(obj, f)
        return alt

def _is_windows_absolute_path(raw: str) -> bool:
    """Riconosce percorsi Windows tipo C:/... anche quando lo script viene validato su Linux."""
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(raw or "")))


def _resolve_output_file(value: Any, default_path: Path) -> Path:
    """Risolve un file di output. Se il percorso e' relativo, lo mette sotto OUTPUT_ROOT.

    Nota: Path("C:/...").is_absolute() e' False su Linux; per questo riconosco
    esplicitamente i path Windows, cosi' non vengono duplicati sotto OUTPUT_ROOT.
    """
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        path = default_path
    elif _is_windows_absolute_path(raw):
        path = Path(raw)
    else:
        path = Path(raw)
        if not path.is_absolute():
            path = OUTPUT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_output_dir(value: Any, default_path: Path) -> Path:
    """Risolve una cartella di output. Se il percorso e' relativo, la mette sotto OUTPUT_ROOT."""
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        path = default_path
    elif _is_windows_absolute_path(raw):
        path = Path(raw)
    else:
        path = Path(raw)
        if not path.is_absolute():
            path = OUTPUT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Percorsi CSV feature prodotti dal C++.
# Permette di usare nomi brevi/simili: glcm_lbp_hog, texture, hog, lbp, glcm, fn.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CSV_FEATURE_DIR = (
    WINDOWS_PROJECT_ROOT / "IPA" / "risultati_rilevamento_fratture" / "CSV_feature"
)

FEATURE_CSV_FILES = {
    "fn": DEFAULT_CSV_FEATURE_DIR / "fratture_mancate_FN.csv",
    "glcm": DEFAULT_CSV_FEATURE_DIR / "roi_feature_glcm_labeled.csv",
    "glcm_lbp_hog": DEFAULT_CSV_FEATURE_DIR / "roi_feature_glcm_lbp_hog_labeled.csv",
    "hog": DEFAULT_CSV_FEATURE_DIR / "roi_feature_hog_labeled.csv",
    "lbp": DEFAULT_CSV_FEATURE_DIR / "roi_feature_lbp_labeled.csv",
}

# Confronto feature richiesto: ogni CSV viene trattato come un diverso dataset di feature.
# Questo corrisponde ai file prodotti dal C++ nella cartella CSV_feature.
DEFAULT_COMPARE_CSV_FEATURE_KEYS = ["fn", "glcm", "glcm_lbp_hog", "hog", "lbp"]
CSV_FEATURE_LABELS = {
    "fn": "Fratture mancate FN",
    "glcm": "GLCM",
    "glcm_lbp_hog": "GLCM+LBP+HOG",
    "hog": "HOG",
    "lbp": "LBP",
}
CSV_FEATURE_DEFAULT_INTERNAL_SET = {
    "fn": "texture_score",
    "glcm": "glcm",
    "glcm_lbp_hog": "texture",
    "hog": "hog",
    "lbp": "lbp",
}

FEATURE_CSV_ALIASES = {
    "fratture_mancate": "fn",
    "fratture_mancate_fn": "fn",
    "missed": "fn",
    "missed_fractures": "fn",
    "roi_feature_glcm_labeled": "glcm",
    "roi_glcm": "glcm",
    "feature_glcm": "glcm",
    "roi_feature_glcm_lbp_hog_labeled": "glcm_lbp_hog",
    "roi_lbp_glcm_hog_features_labeled": "glcm_lbp_hog",
    "roi_feature_lbp_glcm_hog_labeled": "glcm_lbp_hog",
    "lbp_glcm_hog": "glcm_lbp_hog",
    "glcm_hog_lbp": "glcm_lbp_hog",
    "texture": "glcm_lbp_hog",
    "texture_score": "glcm_lbp_hog",
    "features": "glcm_lbp_hog",
    "roi_feature_hog_labeled": "hog",
    "roi_hog": "hog",
    "feature_hog": "hog",
    "roi_feature_lbp_labeled": "lbp",
    "roi_lbp": "lbp",
    "feature_lbp": "lbp",
}

def _normalize_feature_alias(value: str) -> str:
    text = str(value or "").strip().strip('"').strip("'").replace("\\", "/")
    name = Path(text).name
    stem = Path(name).stem if name else text
    key = stem.lower().replace("-", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key)
    return FEATURE_CSV_ALIASES.get(key, key)

def resolve_feature_csv_path(value: Any) -> Path:
    """Risolve alias o nomi simili nei CSV feature ufficiali del C++."""
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        return FEATURE_CSV_FILES["glcm_lbp_hog"]

    direct = Path(raw)
    if direct.exists():
        return direct.resolve()

    key = _normalize_feature_alias(raw)
    if key in FEATURE_CSV_FILES:
        return FEATURE_CSV_FILES[key]

    # Se viene passato un nome file simile, cerca nella cartella CSV_feature.
    stem = Path(raw.replace("\\", "/")).stem.lower()
    candidates = []
    if DEFAULT_CSV_FEATURE_DIR.exists():
        for csv_path in DEFAULT_CSV_FEATURE_DIR.glob("*.csv"):
            s = csv_path.stem.lower()
            if stem and (stem in s or s in stem):
                candidates.append(csv_path)
        if candidates:
            return sorted(candidates, key=lambda p: len(p.name))[0]

    return direct



def _make_run_seed(seed_value: Optional[int]) -> int:
    """
    Se seed_value è None, crea un seed nuovo a ogni esecuzione.
    Se invece passi --seed 7, lo split diventa ripetibile.
    """
    if seed_value is None:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return int(seed_value)


def _svuota_cartella(path: Path) -> None:
    """Cancella tutto il contenuto di una cartella, ma lascia la cartella esistente."""
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _clear_split_output_dirs(out_root: Path) -> None:
    """
    Prima di ogni training svuota le cartelle degli split e i manifest precedenti.
    Così non restano immagini vecchie da esecuzioni precedenti.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "train_undersampled", "validation", "test"]:
        _svuota_cartella(out_root / split_name)

    for manifest in out_root.glob("*_manifest.csv"):
        manifest.unlink()



# ──────────────────────────────────────────────────────────────────────────────
# Ricerca/copia immagini usate negli split
# ──────────────────────────────────────────────────────────────────────────────

def _candidate_image_dirs(csv_path: Path, extra_image_dirs: Optional[List[Path]] = None) -> List[Path]:
    cwd = Path.cwd()
    csv_parent = csv_path.parent.resolve()
    project_root = Path(__file__).parent.resolve()

    extra_image_dirs = extra_image_dirs or []

    candidates = [
        *extra_image_dirs,
        DEFAULT_IMAGE_DIR,
        cwd,
        project_root,
        csv_parent,
        csv_parent.parent,
        project_root / "images",
        project_root / "images" / "image",
        project_root / "images" / "images",
        project_root / "image",
        project_root / "dataset",
        project_root / "dataset" / "images",
        cwd / "images",
        cwd / "images" / "image",
        cwd / "images" / "images",
        cwd / "dataset",
        cwd / "dataset" / "images",
    ]

    out: List[Path] = []
    seen = set()
    for c in candidates:
        try:
            rc = c.resolve()
        except Exception:
            rc = c
        if rc not in seen and c.exists() and c.is_dir():
            seen.add(rc)
            out.append(c)
    return out



def _normalize_image_stem(stem: str) -> str:
    """
    Rimuove suffissi finali tipo _2, _3, _10.
    Esempio:
    nome_2 -> nome
    """
    return re.sub(r"_\d+$", "", stem)


def _find_image_file(image_value: str, csv_path: Path, extra_image_dirs: Optional[List[Path]] = None) -> Optional[Path]:
    """
    Cerca l'immagine partendo dalla colonna 'image' del CSV.
    Cerca prima nelle cartelle passate da CLI, poi nei default locali del progetto.
    Gestisce anche il caso in cui nel CSV ci sia nome_2.png ma nella cartella ci sia nome.png.
    """
    if image_value is None or str(image_value).strip() == "":
        return None

    raw = str(image_value).strip().replace("\\", "/")
    p = Path(raw)

    extra_image_dirs = extra_image_dirs or []

    direct_candidates = [
        p,
        *(root / p for root in extra_image_dirs),
        DEFAULT_IMAGE_DIR / p,
        Path.cwd() / p,
        csv_path.parent / p,
        Path(__file__).parent / p,
    ]
    for cand in direct_candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    name = p.name
    stem = p.stem
    stem_norm = _normalize_image_stem(stem)

    names_to_try = set()
    names_to_try.add(name)

    for ext in IMAGE_EXTENSIONS:
        names_to_try.add(f"{stem}{ext}")
        names_to_try.add(f"{stem}{ext.upper()}")

    if stem_norm != stem:
        for ext in IMAGE_EXTENSIONS:
            names_to_try.add(f"{stem_norm}{ext}")
            names_to_try.add(f"{stem_norm}{ext.upper()}")

    search_roots = _candidate_image_dirs(csv_path, extra_image_dirs)

    for root in search_roots:
        for nm in names_to_try:
            cand = root / nm
            if cand.exists() and cand.is_file():
                return cand.resolve()

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for nm in names_to_try:
            for cand in root.rglob(nm):
                if cand.exists() and cand.is_file():
                    return cand.resolve()

    return None



def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcola l'identità del contenuto per riconoscere copie/rinomine della stessa immagine."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_image_group(
        image_value: str,
        csv_path: Path,
        extra_image_dirs: Optional[List[Path]] = None,
) -> str:
    """Identificatore usato per lo split: la stessa immagine non può attraversare gli split.

    Se il file è trovabile usa SHA-256 del contenuto: anche copie o file rinominati
    vengono raggruppati insieme. Se il file non è trovabile usa un nome normalizzato
    conservativo, eliminando suffissi finali tipo ``_2``.
    """
    if image_value is None or str(image_value).strip() == "":
        raise ValueError("Trovata una ROI senza nome immagine valido nella colonna di gruppo.")

    src = _find_image_file(str(image_value), csv_path, extra_image_dirs)
    if src is not None:
        return f"sha256:{_sha256_file(src)}"

    p = Path(str(image_value).strip().replace("\\", "/"))
    normalized_stem = _normalize_image_stem(p.stem).lower()
    return f"name:{normalized_stem}{p.suffix.lower()}"


def _assert_no_image_leakage(
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        group_col: str = "_image_group",
) -> None:
    """Interrompe l'esecuzione se una stessa immagine compare in più split."""
    sets = {
        "train": set(df.iloc[train_idx][group_col].astype(str)),
        "validation": set(df.iloc[val_idx][group_col].astype(str)),
        "test": set(df.iloc[test_idx][group_col].astype(str)),
    }
    overlaps = {
        "train/validation": sets["train"] & sets["validation"],
        "train/test": sets["train"] & sets["test"],
        "validation/test": sets["validation"] & sets["test"],
    }
    contaminated = {name: sorted(values) for name, values in overlaps.items() if values}
    if contaminated:
        raise RuntimeError(
            "DATA LEAKAGE RILEVATO: la stessa immagine fisica compare in split differenti: "
            + json.dumps(contaminated, indent=2)
        )
    print("[OK] Controllo leakage superato: nessuna immagine condivisa tra train, validation e test.")


def _safe_copy_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.exists():
        try:
            if src.resolve() == dst.resolve():
                return dst
        except Exception:
            pass

        stem, suffix = src.stem, src.suffix
        i = 1
        while dst.exists():
            dst = dst_dir / f"{stem}_{i}{suffix}"
            i += 1

    shutil.copy2(src, dst)
    return dst


def _find_label_file_for_image(src_img: Path, label_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Cerca la label corrispondente nella cartella labels associata alle immagini.
    La label deve avere lo stesso stem dell'immagine e suffisso .txt.
    Se la cartella label non esiste, il training continua comunque.
    """
    base_label_dir = Path(label_dir) if label_dir else DEFAULT_LABEL_DIR
    if not base_label_dir.exists() or not base_label_dir.is_dir():
        return None

    label_path = base_label_dir / f"{src_img.stem}.txt"
    if label_path.exists() and label_path.is_file():
        return label_path.resolve()

    stem_norm = _normalize_image_stem(src_img.stem)
    if stem_norm != src_img.stem:
        label_path = base_label_dir / f"{stem_norm}.txt"
        if label_path.exists() and label_path.is_file():
            return label_path.resolve()

    return None


def _copy_label_for_image(src_img: Path, split_dir: Path, label_dir: Optional[Path] = None) -> Tuple[bool, str, str]:
    """
    Copia la label dell'immagine dentro split_dir/labels.
    Ritorna: found, source_label, copied_label
    """
    label_src = _find_label_file_for_image(src_img, label_dir)
    if label_src is None:
        return False, "", ""

    label_dst_dir = split_dir / "labels"
    label_dst = _safe_copy_file(label_src, label_dst_dir)
    return True, str(label_src), str(label_dst)


def _save_split_images(
    df: pd.DataFrame,
    split_indices: Dict[str, np.ndarray],
    csv_path: Path,
    group_col: str,
    out_root: Path,
    image_dirs: Optional[List[Path]] = None,
    label_dir: Optional[Path] = None,
) -> Dict[str, dict]:
    result: Dict[str, dict] = {}

    if group_col not in df.columns:
        print(f"[WARN] Impossibile salvare immagini split: colonna '{group_col}' non presente.")
        return result

    # IMPORTANTE: a ogni nuova esecuzione pulisce train/validation/test
    # prima di copiare le nuove immagini.
    _clear_split_output_dirs(out_root)

    search_dirs = _candidate_image_dirs(csv_path, image_dirs)
    print("[*] Cartelle cercate per immagini:")
    for d in search_dirs:
        print(f"    - {d}")

    for split_name, idx_arr in split_indices.items():
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        values = (
            df.iloc[idx_arr][group_col]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        manifest_rows = []
        copied = 0
        missing = 0

        for image_value in values:
            src = _find_image_file(image_value, csv_path, image_dirs)
            if src is None:
                missing += 1
                manifest_rows.append({
                    "split": split_name,
                    "image": image_value,
                    "found": False,
                    "source_path": "",
                    "copied_path": "",
                    "label_found": False,
                    "label_source_path": "",
                    "label_copied_path": "",
                })
                continue

            dst = _safe_copy_file(src, split_dir)
            label_found, label_source, label_copied = _copy_label_for_image(src, split_dir, label_dir)

            copied += 1
            manifest_rows.append({
                "split": split_name,
                "image": image_value,
                "found": True,
                "source_path": str(src),
                "copied_path": str(dst),
                "label_found": label_found,
                "label_source_path": label_source,
                "label_copied_path": label_copied,
            })

            if not label_found:
                expected_label_dir = label_dir or DEFAULT_LABEL_DIR
                print(f"[WARN] Label mancante per {src.name}: {expected_label_dir / (src.stem + '.txt')}")

        manifest_path = out_root / f"{split_name}_manifest.csv"
        manifest_path = _safe_to_csv(pd.DataFrame(manifest_rows), manifest_path, index=False)

        result[split_name] = {
            "dir": str(split_dir),
            "manifest": str(manifest_path),
            "unique_images": int(len(values)),
            "copied": int(copied),
            "missing": int(missing),
        }

        print(
            f"[*] Immagini {split_name}: copiate {copied}/{len(values)} in {split_dir}"
            + (f" | mancanti: {missing}" if missing else "")
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    derived_features: List[str],
    method_levels: List[str],
) -> Tuple[np.ndarray, List[str]]:
    arrays: List[np.ndarray] = []
    names: List[str] = []

    for col in feature_cols:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            v = np.zeros(len(df), dtype=np.float64)
        arrays.append(v[:, None])
        names.append(col)

    def _num(c: str) -> np.ndarray:
        return pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    if "score" in derived_features:
        arrays.append(_num("score")[:, None]); names.append("score")
    if "width" in derived_features:
        arrays.append(_num("width")[:, None]); names.append("width")
    if "height" in derived_features:
        arrays.append(_num("height")[:, None]); names.append("height")
    if "area" in derived_features:
        arrays.append((_num("width") * _num("height"))[:, None]); names.append("area")
    if "aspect" in derived_features:
        arrays.append((_num("width") / np.maximum(_num("height"), 1e-12))[:, None]); names.append("aspect")
    if "method_one_hot" in derived_features:
        methods = df.get("method", pd.Series([""] * len(df))).astype(str).to_numpy()
        for level in method_levels:
            arrays.append((methods == level).astype(np.float64)[:, None])
            names.append(f"method={level}")

    if not arrays:
        raise ValueError("Nessuna feature selezionata.")

    X = np.hstack(arrays).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


# ──────────────────────────────────────────────────────────────────────────────
# Etichettatura ROI tramite IoU con la Ground Truth
# ──────────────────────────────────────────────────────────────────────────────

def _apply_iou_tp_rule(df: pd.DataFrame, iou_threshold: float = TP_IOU_THRESHOLD) -> pd.DataFrame:
    """Prepara la label ML delle ROI.

    Caso preferito: se il CSV contiene ``best_iou`` o una colonna IoU equivalente,
    la label viene ricalcolata con la regola clinica ``IoU >= 0.50``.

    Caso compatibile: alcuni CSV feature prodotti in precedenza contengono gia'
    una colonna ``label`` ma non salvano ``best_iou``. In quel caso il training
    NON deve bloccarsi: viene usata la label esistente, conservando nel report
    che la sorgente della label e' il CSV. Questo permette di confrontare anche
    ``fratture_mancate_FN.csv`` e i CSV feature piu' vecchi.
    """
    out = df.copy()

    # Accetta anche nomi alternativi usati in versioni diverse degli script.
    iou_candidates = [
        "best_iou", "best_gt_iou", "gt_iou", "max_iou", "iou", "IoU",
        "bestIoU", "best_iou_gt", "iou_gt",
    ]
    iou_col = next((c for c in iou_candidates if c in out.columns), None)

    if iou_col is not None:
        if "label" in out.columns and "label_original" not in out.columns:
            out["label_original"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
        best_iou = pd.to_numeric(out[iou_col], errors="coerce").fillna(0.0).astype(float)
        out["best_iou"] = best_iou
        out["iou_tp_threshold"] = float(iou_threshold)
        out["iou_tp_rule"] = f"best_iou >= {float(iou_threshold):.2f}"
        out["iou_tp_match"] = (best_iou >= float(iou_threshold)).astype(int)
        out["label_source"] = f"{iou_col}_threshold"
        out["label"] = out["iou_tp_match"].astype(int)
        return out

    if "label" in out.columns:
        label_num = pd.to_numeric(out["label"], errors="coerce")
        valid = label_num.dropna().astype(int)
        valid = valid[valid.isin([0, 1])]
        if len(valid) == len(out):
            out["label_original"] = valid.to_numpy(dtype=int)
            out["label"] = valid.to_numpy(dtype=int)
            out["best_iou"] = np.nan
            out["iou_tp_threshold"] = float(iou_threshold)
            out["iou_tp_rule"] = "label gia' presente nel CSV; best_iou assente"
            out["iou_tp_match"] = out["label"].astype(int)
            out["label_source"] = "csv_label_fallback_no_best_iou"
            print(
                "[WARN] CSV senza colonna best_iou: uso la colonna label gia' presente "
                "come ground truth ROI-level. Per una relazione piu' rigorosa, rigenera "
                "il CSV dal C++ includendo best_iou."
            )
            return out

    raise ValueError(
        "Il CSV non contiene best_iou ne' una label binaria valida. "
        "Per addestrare serve una ground truth ROI-level: aggiungi best_iou oppure label 0/1."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metriche / soglia
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float, beta: float) -> dict:
    pred = (y_score >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    spec = tn / (tn + fp) if tn + fp > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    fnr = fn / (fn + tp) if fn + tp > 0 else 0.0
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    b2 = beta * beta
    fb = (1 + b2) * prec * rec / (b2 * prec + rec) if b2 * prec + rec > 0 else 0.0

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())

    try:
        roc = float(roc_auc_score(y_true, y_score)) if pos > 0 and neg > 0 else 0.0
    except Exception:
        roc = 0.0

    try:
        ap = float(average_precision_score(y_true, y_score)) if pos > 0 else 0.0
    except Exception:
        ap = 0.0

    bk = f"F{beta:g}"
    return {
        "threshold": float(threshold),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall_sensitivity_TPR": round(rec, 6),
        "specificity_TNR": round(spec, 6),
        "FPR": round(fpr, 6),
        "FNR": round(fnr, 6),
        "F1": round(f1, 6),
        bk: round(fb, 6),
        "balanced_accuracy": round(0.5 * (rec + spec), 6),
        "ROC_AUC": round(roc, 6),
        "PR_AP": round(ap, 6),
    }


def _find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float,
    min_recall: Optional[float],
    min_precision: Optional[float] = None,
) -> Tuple[float, dict]:
    thresholds = np.unique(y_score)
    if len(thresholds) > MAX_THRESHOLD_POINTS:
        thresholds = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, int(MAX_THRESHOLD_POINTS))))
    if len(thresholds) == 0:
        return 0.5, _compute_metrics(y_true, y_score, 0.5, beta)

    thresholds = np.r_[thresholds.max() + 1e-12, thresholds[::-1], thresholds.min() - 1e-12]
    beta_key = f"F{beta:g}"

    best_thr = 0.5
    best_tuple = None
    best_metrics = None

    for thr in thresholds:
        met = _compute_metrics(y_true, y_score, float(thr), beta)
        pr = met["precision"]
        re = met["recall_sensitivity_TPR"]
        fb = met[beta_key]
        fpr = met["FPR"]

        if min_recall is not None and re < min_recall:
            continue
        if min_precision is not None and pr < min_precision:
            continue

        score_tuple = (fb, re, pr, -fpr) if beta >= 1.0 else (fb, pr, re, -fpr)
        if best_tuple is None or score_tuple > best_tuple:
            best_tuple = score_tuple
            best_thr = float(thr)
            best_metrics = met

    if best_metrics is None:
        for thr in thresholds:
            met = _compute_metrics(y_true, y_score, float(thr), beta)
            pr = met["precision"]
            re = met["recall_sensitivity_TPR"]
            fb = met[beta_key]
            fpr = met["FPR"]
            recall_gap = max(0.0, (min_recall or 0.0) - re)
            precision_gap = max(0.0, (min_precision or 0.0) - pr)
            score_tuple = (-(recall_gap + precision_gap), fb, re, pr, -fpr)
            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best_thr = float(thr)
                best_metrics = met

    return best_thr, best_metrics or _compute_metrics(y_true, y_score, best_thr, beta)


def _operating_points(y_true: np.ndarray, y_score: np.ndarray, beta: float) -> dict:
    thresholds = np.unique(y_score)
    if len(thresholds) > MAX_THRESHOLD_POINTS:
        thresholds = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, int(MAX_THRESHOLD_POINTS))))
    if len(thresholds) == 0:
        thresholds = np.array([0.5])
    thresholds = np.r_[thresholds.max() + 1e-12, thresholds[::-1], thresholds.min() - 1e-12]

    by_recall = {}
    for target in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        feasible = []
        for thr in thresholds:
            met = _compute_metrics(y_true, y_score, float(thr), beta)
            if met["recall_sensitivity_TPR"] >= target:
                feasible.append(met)
        if feasible:
            by_recall[f"recall>={target:.2f}"] = max(feasible, key=lambda m: (m["precision"], m["F1"], -m["FPR"]))

    by_precision = {}
    for target in (0.30, 0.40, 0.50, 0.60, 0.70):
        feasible = []
        for thr in thresholds:
            met = _compute_metrics(y_true, y_score, float(thr), beta)
            if met["precision"] >= target:
                feasible.append(met)
        if feasible:
            by_precision[f"precision>={target:.2f}"] = max(feasible, key=lambda m: (m["recall_sensitivity_TPR"], m["F1"], -m["FPR"]))

    return {"best_precision_at_recall": by_recall, "best_recall_at_precision": by_precision}


def _threshold_sweep_rows(y_true: np.ndarray, y_score: np.ndarray, beta: float, split_name: str, max_points: int = MAX_THRESHOLD_POINTS) -> List[dict]:
    """Salva tutti i punti operativi ML ROI-level utili per scegliere la soglia.

    Questa e' una grid search ROI-level: non cambia il modello, cambia solo la
    soglia con cui il punteggio ML viene trasformato in keep/reject.
    """
    thresholds = np.unique(y_score)
    if len(thresholds) > max_points:
        thresholds = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, int(max_points))))
    if len(thresholds) == 0:
        thresholds = np.array([0.5])
    thresholds = np.r_[thresholds.max() + 1e-12, thresholds[::-1], thresholds.min() - 1e-12]

    rows: List[dict] = []
    for thr in thresholds:
        met = _compute_metrics(y_true, y_score, float(thr), beta)
        cm = met["confusion_matrix"]
        rows.append({
            "split": split_name,
            "threshold": float(thr),
            "TP": int(cm["TP"]),
            "FP": int(cm["FP"]),
            "TN": int(cm["TN"]),
            "FN": int(cm["FN"]),
            "accuracy": float(met["accuracy"]),
            "precision": float(met["precision"]),
            "recall": float(met["recall_sensitivity_TPR"]),
            "specificity": float(met["specificity_TNR"]),
            "FPR": float(met["FPR"]),
            "F1": float(met["F1"]),
            f"F{beta:g}": float(met[f"F{beta:g}"]),
            "ROC_AUC": float(met["ROC_AUC"]),
            "PR_AP": float(met["PR_AP"]),
        })
    return rows


def _get_scores(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    d = pipeline.decision_function(X)
    return (d - d.min()) / max(d.max() - d.min(), 1e-12)



# ──────────────────────────────────────────────────────────────────────────────
# Metodologie confrontate: feature set, preprocessing, supervised e unsupervised
# ──────────────────────────────────────────────────────────────────────────────

AVAILABLE_FEATURE_SETS = ["glcm", "lbp", "hog", "glcm_lbp", "texture", "texture_score"]
DEFAULT_FULL_FEATURE_SETS = ["glcm", "lbp", "hog", "glcm_lbp", "texture", "texture_score"]
BASE_SUPERVISED_CLASSIFIERS = ["SVM_RBF", "SVM_LINEAR", "KNN", "RANDOM_FOREST"]
ENSEMBLE_SUPERVISED_CLASSIFIERS = ["SOFT_VOTING", "STACKING"]
REQUESTED_SUPERVISED_CLASSIFIERS = BASE_SUPERVISED_CLASSIFIERS + ENSEMBLE_SUPERVISED_CLASSIFIERS
DEFAULT_CORE_CLASSIFIERS = BASE_SUPERVISED_CLASSIFIERS.copy()
DEFAULT_EXTENDED_CLASSIFIERS = REQUESTED_SUPERVISED_CLASSIFIERS.copy()
MODEL_NAMES = DEFAULT_EXTENDED_CLASSIFIERS.copy()


def _csv_list(value: Any, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    raw = str(value).strip()
    if not raw:
        return list(default)
    if raw.lower() in {"all", "tutti", "full"}:
        return list(default)
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return out or list(default)


def _normalization_list(value: Any, full: bool = False) -> List[str]:
    default = ["standard"] if not full else ["none", "standard", "minmax"]
    out = _csv_list(value, default)
    valid = {"none", "standard", "minmax", "robust"}
    cleaned = []
    for item in out:
        key = item.lower().replace("scaler", "").replace("_", "").replace("-", "")
        if key in {"no", "none", "null"}:
            cleaned.append("none")
        elif key in {"standard", "std"}:
            cleaned.append("standard")
        elif key in {"minmax", "min"}:
            cleaned.append("minmax")
        elif key in {"robust"}:
            cleaned.append("robust")
        else:
            raise ValueError(f"Normalizzazione non valida: {item}. Valide: {sorted(valid)}")
    return list(dict.fromkeys(cleaned))


def _classifier_list(value: Any, model_set: str = "extended") -> List[str]:
    model_set = str(model_set or "extended").lower()
    if model_set == "core":
        default = DEFAULT_CORE_CLASSIFIERS
    elif model_set == "all":
        default = DEFAULT_EXTENDED_CLASSIFIERS
    else:
        default = DEFAULT_EXTENDED_CLASSIFIERS
    raw = str(value or "").strip()
    if not raw:
        return list(default)
    if raw.lower() in {"all", "tutti", "full"}:
        return list(default)
    aliases = {
        "svm": "SVM_RBF", "svm_rbf": "SVM_RBF", "rbf": "SVM_RBF",
        "svm_linear": "SVM_LINEAR", "linear_svm": "SVM_LINEAR",
        "logistic": "LOGISTIC_REGRESSION", "logreg": "LOGISTIC_REGRESSION", "lr": "LOGISTIC_REGRESSION",
        "knn": "KNN", "nearest": "KNN",
        "rf": "RANDOM_FOREST", "random_forest": "RANDOM_FOREST",
        "voting": "SOFT_VOTING", "soft_voting": "SOFT_VOTING", "ensemble": "SOFT_VOTING",
        "stacking": "STACKING", "stack": "STACKING",
        "gb": "GRADIENT_BOOSTING", "gradient_boosting": "GRADIENT_BOOSTING",
        "xgb": "XGBOOST", "xgboost": "XGBOOST",
    }
    out = []
    for item in raw.split(","):
        key = item.strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            continue
        out.append(aliases.get(key, key.upper()))
    return list(dict.fromkeys(out))


def _effective_pca_components(requested: int, n_samples: int, n_features: int):
    if requested is None or int(requested) <= 0:
        return None
    max_allowed = max(1, min(n_samples, n_features) - 1)
    return min(int(requested), max_allowed)


def _undersample_training_set(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undersampling della classe maggioritaria SOLO sul training set.

    ratio = negativi / positivi dopo undersampling.
    Con ratio=1.0 il train diventa bilanciato 1:1.
    """
    rng = np.random.default_rng(int(seed))
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Undersampling impossibile: nel train servono sia label=0 che label=1.")

    if len(neg_idx) >= len(pos_idx):
        keep_neg = min(len(neg_idx), max(1, int(round(len(pos_idx) * float(ratio)))))
        sampled_neg = rng.choice(neg_idx, size=keep_neg, replace=False)
        sampled = np.r_[pos_idx, sampled_neg]
    else:
        keep_pos = min(len(pos_idx), max(1, int(round(len(neg_idx) * float(ratio)))))
        sampled_pos = rng.choice(pos_idx, size=keep_pos, replace=False)
        sampled = np.r_[sampled_pos, neg_idx]

    rng.shuffle(sampled)
    return X[sampled], y[sampled], sampled


def _feature_spec_for_set(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str], List[str], np.ndarray, List[str]]:
    feature_set = str(feature_set or "texture_score").lower()
    if feature_set not in AVAILABLE_FEATURE_SETS:
        raise ValueError(f"Feature set non valido: {feature_set}. Validi: {AVAILABLE_FEATURE_SETS}")

    glcm_cols = [c for c in df.columns if c.startswith("glcm_")]
    lbp_cols = [c for c in df.columns if c.startswith("lbp_")]
    hog_cols = [c for c in df.columns if c.startswith("hog_")]

    if feature_set == "glcm":
        feature_cols, derived_features = glcm_cols, []
    elif feature_set == "lbp":
        feature_cols, derived_features = lbp_cols, []
    elif feature_set == "hog":
        feature_cols, derived_features = hog_cols, []
    elif feature_set == "glcm_lbp":
        feature_cols, derived_features = glcm_cols + lbp_cols, []
    elif feature_set == "texture":
        feature_cols, derived_features = glcm_cols + lbp_cols + hog_cols, []
    else:
        feature_cols = glcm_cols + lbp_cols + hog_cols
        derived_features = ["score", "width", "height", "area", "aspect", "method_one_hot"]

    method_levels: List[str] = []
    if "method_one_hot" in derived_features and "method" in df.columns:
        method_levels = sorted(str(v) for v in df["method"].dropna().unique())

    X_raw, feature_names = _extract_features(df, feature_cols, derived_features, method_levels)
    return feature_cols, derived_features, method_levels, X_raw, feature_names


def _preprocess_steps(
    normalization: str,
    pca_components_requested: int,
    n_samples: int,
    n_features: int,
    seed: int,
    use_pca: bool = True,
) -> List[Tuple[str, object]]:
    steps: List[Tuple[str, object]] = []
    normalization = str(normalization or "standard").lower()
    if normalization == "standard":
        steps.append(("scaler", StandardScaler()))
    elif normalization == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif normalization == "robust":
        steps.append(("scaler", RobustScaler()))
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Normalizzazione non valida: {normalization}")

    pca_components = _effective_pca_components(pca_components_requested, n_samples, n_features) if use_pca else None
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=int(seed))))
    return steps


def _make_pipeline(
    classifier_name: str,
    args,
    n_samples: int,
    n_features: int,
    normalization: str,
    use_pca: bool,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Pipeline]:
    name = str(classifier_name).upper()
    params = params or {}

    def _param(name_: str, default: Any = None) -> Any:
        return params.get(name_, getattr(args, name_, default))

    pca_components = int(_param("pca_components", 30))
    seed = int(args.seed)
    steps = _preprocess_steps(normalization, pca_components, n_samples, n_features, seed, use_pca=use_pca)

    gamma_value = _param("svm_gamma", "scale")
    if str(gamma_value) not in ("scale", "auto"):
        gamma_value = float(gamma_value)

    rf_max_depth_arg = int(_param("rf_max_depth", 0))
    rf_max_depth = None if rf_max_depth_arg <= 0 else rf_max_depth_arg

    if name == "SVM_RBF":
        clf = SVC(kernel="rbf", C=float(_param("svm_c", 10.0)), gamma=gamma_value,
                  class_weight=None, probability=True, cache_size=int(_param("svm_cache_mb", 1000)), random_state=seed)
    elif name == "SVM_LINEAR":
        clf = SVC(kernel="linear", C=float(_param("svm_c", 10.0)), class_weight=None,
                  probability=True, cache_size=int(_param("svm_cache_mb", 1000)), random_state=seed)
    elif name == "LOGISTIC_REGRESSION":
        clf = LogisticRegression(max_iter=int(getattr(args, "logreg_max_iter", 3000)), solver="lbfgs", n_jobs=int(getattr(args, "n_jobs", -1)))
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=int(_param("knn_neighbors", 7)), weights=str(_param("knn_weights", "distance")))
    elif name == "RANDOM_FOREST":
        clf = RandomForestClassifier(
            n_estimators=int(_param("rf_n_estimators", 300)), max_depth=rf_max_depth,
            min_samples_leaf=int(_param("rf_min_samples_leaf", 1)), class_weight=None,
            n_jobs=int(getattr(args, "n_jobs", -1)), random_state=seed)
    elif name == "GRADIENT_BOOSTING":
        clf = GradientBoostingClassifier(
            n_estimators=int(_param("gb_n_estimators", 200)),
            learning_rate=float(_param("gb_learning_rate", 0.05)),
            max_depth=int(_param("gb_max_depth", 3)), random_state=seed)
    elif name == "XGBOOST":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except Exception:
            print("[WARN] XGBoost non installato: salto il modello XGBOOST. Installa con: pip install xgboost")
            return None
        clf = XGBClassifier(
            n_estimators=int(getattr(args, "xgb_n_estimators", 250)),
            max_depth=int(getattr(args, "xgb_max_depth", 3)),
            learning_rate=float(getattr(args, "xgb_learning_rate", 0.05)),
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=seed, n_jobs=int(getattr(args, "n_jobs", -1)))
    else:
        raise ValueError(f"Classificatore non riconosciuto: {classifier_name}")

    return Pipeline([*steps, ("clf", clf)])


def build_classifiers(args, n_samples: int, n_features: int) -> Dict[str, Pipeline]:
    """Compatibilita' con le vecchie versioni: costruisce solo il set base."""
    models: Dict[str, Pipeline] = {}
    for name in DEFAULT_CORE_CLASSIFIERS:
        pipe = _make_pipeline(name, args, n_samples, n_features, "standard", use_pca=(name.startswith("SVM") or name in {"KNN", "LOGISTIC_REGRESSION"}))
        if pipe is not None:
            models[name] = pipe
    return models


def _tuning_level(args) -> str:
    level = str(getattr(args, "tuning_level", "quick") or "quick").strip().lower()
    if bool(getattr(args, "fast", False)):
        return "fast"
    if level not in {"none", "fast", "quick", "full"}:
        raise ValueError("--tuning-level deve essere uno tra: none, fast, quick, full")
    return level


def _classifier_tuning_grid(classifier_name: str, args) -> List[Dict[str, Any]]:
    """Griglia di tuning manuale sul validation set.

    Non uso il test set per scegliere gli iper-parametri. Per ogni combinazione:
    1) fit sul train undersampled;
    2) scelta soglia su validation;
    3) ranking con F-beta/recall/precision.
    """
    name = str(classifier_name).upper()
    level = _tuning_level(args)
    if level == "none":
        return [{}]
    if level == "fast":
        if name == "SVM_RBF":
            return [{"svm_c": 10.0, "svm_gamma": "scale"}, {"svm_c": 3.0, "svm_gamma": "scale"}]
        if name == "SVM_LINEAR":
            return [{"svm_c": 1.0}, {"svm_c": 10.0}]
        if name == "KNN":
            return [{"knn_neighbors": 5, "knn_weights": "distance"}, {"knn_neighbors": 9, "knn_weights": "distance"}]
        if name == "RANDOM_FOREST":
            return [{"rf_n_estimators": 200, "rf_max_depth": 0, "rf_min_samples_leaf": 1}]
        return [{}]

    if name == "SVM_RBF":
        # Grid search vera e propria su C × gamma: vengono provate TUTTE le
        # combinazioni sul validation set (criterio F2) e viene usata la migliore.
        # gamma="scale" = 1/(n_feature·var(X)) di sklearn; gli altri sono valori
        # espliciti in scala log. La C e il gamma scelti finiscono nel modello finale.
        if level == "full":
            c_values = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
            gamma_values = ["scale", 0.3, 0.1, 0.03, 0.01, 0.003]
        else:  # livello "quick" (default)
            c_values = [1.0, 3.0, 10.0, 30.0]
            gamma_values = ["scale", 0.1, 0.01]
        return [{"svm_c": c, "svm_gamma": g} for c in c_values for g in gamma_values]

    if name == "SVM_LINEAR":
        grid = [{"svm_c": 0.1}, {"svm_c": 1.0}, {"svm_c": 10.0}, {"svm_c": 30.0}]
        if level == "full":
            grid.extend([{"svm_c": 0.03}, {"svm_c": 100.0}])
        return grid

    if name == "KNN":
        grid = [
            {"knn_neighbors": 3, "knn_weights": "distance"},
            {"knn_neighbors": 5, "knn_weights": "distance"},
            {"knn_neighbors": 7, "knn_weights": "distance"},
            {"knn_neighbors": 11, "knn_weights": "distance"},
            {"knn_neighbors": 5, "knn_weights": "uniform"},
        ]
        if level == "full":
            grid.extend([
                {"knn_neighbors": 15, "knn_weights": "distance"},
                {"knn_neighbors": 21, "knn_weights": "distance"},
                {"knn_neighbors": 11, "knn_weights": "uniform"},
            ])
        return grid

    if name == "RANDOM_FOREST":
        grid = [
            {"rf_n_estimators": 200, "rf_max_depth": 0, "rf_min_samples_leaf": 1},
            {"rf_n_estimators": 300, "rf_max_depth": 0, "rf_min_samples_leaf": 2},
            {"rf_n_estimators": 300, "rf_max_depth": 12, "rf_min_samples_leaf": 1},
            {"rf_n_estimators": 400, "rf_max_depth": 20, "rf_min_samples_leaf": 2},
        ]
        if level == "full":
            grid.extend([
                {"rf_n_estimators": 500, "rf_max_depth": 0, "rf_min_samples_leaf": 1},
                {"rf_n_estimators": 500, "rf_max_depth": 16, "rf_min_samples_leaf": 3},
            ])
        return grid

    return [{}]


def _score_selection_tuple(metrics: Dict[str, Any], beta_key: str) -> Tuple[float, float, float, float]:
    return (
        float(metrics.get(beta_key, 0.0)),
        float(metrics.get("recall_sensitivity_TPR", 0.0)),
        float(metrics.get("precision", 0.0)),
        float(metrics.get("ROC_AUC", 0.0)),
    )


def _use_pca_for_classifier(classifier: str, args) -> bool:
    classifier = str(classifier).upper()
    return bool(int(getattr(args, "pca_components", 30)) > 0) and (
        classifier in {"SVM_RBF", "SVM_LINEAR", "KNN"}
        or bool(getattr(args, "pca_for_trees", False))
    )


def _train_tuned_single_classifier(
    classifier: str,
    args,
    X_tr_bal: np.ndarray,
    y_tr_bal: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    beta: float,
    min_recall: Optional[float],
    min_precision: Optional[float],
    normalization: str,
) -> Tuple[Any, float, Dict[str, Any], Dict[str, Any], int]:
    beta_key = f"F{beta:g}"
    use_pca = _use_pca_for_classifier(classifier, args)
    best = None
    grid = _classifier_tuning_grid(classifier, args)
    for params in grid:
        pipeline = _make_pipeline(
            classifier,
            args,
            X_tr_bal.shape[0],
            X_tr_bal.shape[1],
            normalization,
            use_pca=use_pca,
            params=params,
        )
        if pipeline is None:
            continue
        pipeline.fit(X_tr_bal, y_tr_bal)
        va_scores = _get_scores(pipeline, X_va)
        thr, val_metrics = _find_best_threshold(y_va, va_scores, beta, min_recall, min_precision)
        key = (_score_selection_tuple(val_metrics, beta_key), -len(str(params)))
        if best is None or key > best[0]:
            best = (key, pipeline, float(thr), val_metrics, dict(params))
    if best is None:
        raise RuntimeError(f"Tuning fallito per {classifier}")
    _, pipeline, thr, val_metrics, params = best
    return pipeline, thr, val_metrics, params, len(grid)


def _make_ensemble_estimator(
    ensemble_name: str,
    args,
    X_tr_bal: np.ndarray,
    y_tr_bal: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    beta: float,
    min_recall: Optional[float],
    min_precision: Optional[float],
    normalization: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Costruisce ensemble dai modelli base migliori dopo tuning su validation."""
    tuned_base: List[Tuple[str, Any]] = []
    base_report: Dict[str, Any] = {}
    for base_name in BASE_SUPERVISED_CLASSIFIERS:
        base_pipe, base_thr, base_val, base_params, n_grid = _train_tuned_single_classifier(
            base_name, args, X_tr_bal, y_tr_bal, X_va, y_va, beta, min_recall, min_precision, normalization
        )
        tuned_base.append((base_name.lower(), base_pipe))
        base_report[base_name] = {
            "validation_threshold": float(base_thr),
            "validation_metrics": base_val,
            "best_params": base_params,
            "grid_candidates": int(n_grid),
        }

    name = str(ensemble_name).upper()
    if name == "SOFT_VOTING":
        estimator = VotingClassifier(estimators=tuned_base, voting="soft", n_jobs=int(getattr(args, "n_jobs", -1)))
    elif name == "STACKING":
        estimator = StackingClassifier(
            estimators=tuned_base,
            final_estimator=LogisticRegression(max_iter=3000, solver="lbfgs"),
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=int(getattr(args, "n_jobs", -1)),
        )
    else:
        raise ValueError(f"Ensemble non riconosciuto: {ensemble_name}")
    return estimator, {"base_models": base_report, "ensemble_type": name}


def _train_tuned_estimator(
    classifier: str,
    args,
    X_tr_bal: np.ndarray,
    y_tr_bal: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    beta: float,
    min_recall: Optional[float],
    min_precision: Optional[float],
    normalization: str,
) -> Tuple[Any, float, Dict[str, Any], Dict[str, Any], int, bool]:
    """Ritorna estimator fit, soglia validation, metriche validation, tuning report."""
    name = str(classifier).upper()
    if name in ENSEMBLE_SUPERVISED_CLASSIFIERS:
        estimator, report = _make_ensemble_estimator(
            name, args, X_tr_bal, y_tr_bal, X_va, y_va, beta, min_recall, min_precision, normalization
        )
        estimator.fit(X_tr_bal, y_tr_bal)
        va_scores = _get_scores(estimator, X_va)
        thr, val_metrics = _find_best_threshold(y_va, va_scores, beta, min_recall, min_precision)
        n_base_candidates = sum(int(v.get("grid_candidates", 0)) for v in report.get("base_models", {}).values())
        return estimator, float(thr), val_metrics, report, n_base_candidates, False

    estimator, thr, val_metrics, params, n_grid = _train_tuned_single_classifier(
        name, args, X_tr_bal, y_tr_bal, X_va, y_va, beta, min_recall, min_precision, normalization
    )
    return estimator, float(thr), val_metrics, {"best_params": params}, n_grid, _use_pca_for_classifier(name, args)


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    try:
        if len(set(labels.tolist())) < 2 or len(labels) < 3:
            return 0.0
        max_n = min(len(labels), 2000)
        if len(labels) > max_n:
            rng = np.random.default_rng(123)
            idx = rng.choice(np.arange(len(labels)), size=max_n, replace=False)
            return float(silhouette_score(X[idx], labels[idx]))
        return float(silhouette_score(X, labels))
    except Exception:
        return 0.0


def _map_clusters_to_binary(train_clusters: np.ndarray, y_train: np.ndarray) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for cl in sorted(set(train_clusters.tolist())):
        yy = y_train[train_clusters == cl]
        if len(yy) == 0:
            mapping[int(cl)] = 0
        else:
            mapping[int(cl)] = int(np.mean(yy) >= 0.5)
    return mapping


def _binary_metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.asarray(y_pred).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    return {"accuracy": acc, "precision": precision, "recall": recall, "F1": f1, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def _run_unsupervised_experiments(
    df: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_sets: List[str],
    args,
) -> List[dict]:
    if not bool(getattr(args, "run_unsupervised", False)):
        return []
    rows: List[dict] = []
    n_clusters = int(getattr(args, "unsupervised_clusters", 2))
    pca_components = int(getattr(args, "unsupervised_pca", 30))
    for feature_set in feature_sets:
        try:
            _fc, _dfv, _mlv, X_raw, feature_names = _feature_spec_for_set(df, feature_set)
            X_tr = X_raw[train_idx]
            X_te = X_raw[test_idx]
            y_tr = y[train_idx]
            y_te = y[test_idx]
            steps = [("scaler", StandardScaler())]
            pca_n = _effective_pca_components(pca_components, len(train_idx), X_raw.shape[1])
            if pca_n is not None:
                steps.append(("pca", PCA(n_components=pca_n, random_state=int(args.seed))))
            prep = Pipeline(steps)
            X_tr_p = prep.fit_transform(X_tr)
            X_te_p = prep.transform(X_te)
            for method in ["KMEANS", "GMM"]:
                if method == "KMEANS":
                    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=int(args.seed))
                    tr_clusters = model.fit_predict(X_tr_p)
                    te_clusters = model.predict(X_te_p)
                else:
                    model = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=int(args.seed))
                    tr_clusters = model.fit_predict(X_tr_p)
                    te_clusters = model.predict(X_te_p)
                mapping = _map_clusters_to_binary(tr_clusters, y_tr)
                te_pred = np.array([mapping.get(int(c), 0) for c in te_clusters], dtype=int)
                bm = _binary_metrics_from_pred(y_te, te_pred)
                rows.append({
                    "feature_set": feature_set,
                    "method": method,
                    "n_clusters": int(n_clusters),
                    "normalization": "standard",
                    "pca_components": int(pca_n or 0),
                    "feature_count": int(X_raw.shape[1]),
                    "silhouette_train": round(_safe_silhouette(X_tr_p, tr_clusters), 6),
                    "silhouette_test": round(_safe_silhouette(X_te_p, te_clusters), 6),
                    "mapped_test_accuracy": round(float(bm["accuracy"]), 6),
                    "mapped_test_precision": round(float(bm["precision"]), 6),
                    "mapped_test_recall": round(float(bm["recall"]), 6),
                    "mapped_test_F1": round(float(bm["F1"]), 6),
                    "TP": int(bm["TP"]), "FP": int(bm["FP"]), "TN": int(bm["TN"]), "FN": int(bm["FN"]),
                    "note": "Esplorativo: non usato come modello finale GUI",
                })
        except Exception as exc:
            rows.append({"feature_set": feature_set, "method": "UNSUPERVISED_ERROR", "error": str(exc)})
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────────────────────────────────────

def _make_roi_level_splits_for_small_dataset(
        y: np.ndarray,
        args,
        reason: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback per CSV piccoli: split per singola ROI, non per immagine.

    ATTENZIONE: questo fallback serve solo quando nel CSV ci sono meno di 3 immagini
    fisiche distinte. In quel caso non e' matematicamente possibile creare train,
    validation e test indipendenti per immagine. Il confronto tra modelli funziona,
    ma la stima delle metriche e' meno rigorosa e puo' essere ottimistica.
    """
    n = int(len(y))
    unique_labels = sorted(int(v) for v in np.unique(y))
    if n < 3:
        raise ValueError(
            "Il CSV contiene meno di 3 ROI: impossibile creare train, validation e test. "
            "Usa un CSV con piu' righe o unisci piu' immagini nel CSV di addestramento."
        )
    if len(unique_labels) < 2:
        raise ValueError(
            "Il CSV contiene una sola classe dopo la regola best_iou >= 0.50. "
            "Per addestrare servono sia ROI positive sia ROI negative."
        )

    rng = np.random.default_rng(int(args.seed))
    train_idx: List[int] = []
    remaining: List[int] = []

    # Garantisce almeno un esempio per classe nel training, altrimenti
    # l'undersampling e i classificatori non possono essere addestrati.
    for label in unique_labels:
        cls_idx = np.flatnonzero(y == label)
        rng.shuffle(cls_idx)
        train_idx.append(int(cls_idx[0]))
        remaining.extend(int(i) for i in cls_idx[1:])

    remaining_arr = np.asarray(remaining, dtype=int)
    rng.shuffle(remaining_arr)

    if remaining_arr.size < 2:
        # Dataset estremamente piccolo: duplica solo gli indici di valutazione.
        # Il training resta valido, ma le metriche NON vanno considerate definitive.
        all_idx = np.arange(n, dtype=int)
        rest = [int(i) for i in all_idx if int(i) not in set(train_idx)]
        val_idx = np.asarray(rest[:1] or train_idx[:1], dtype=int)
        test_idx = np.asarray(rest[1:2] or rest[:1] or train_idx[-1:], dtype=int)
    else:
        n_test = max(1, int(round(n * float(args.test_size))))
        n_val = max(1, int(round(n * float(args.val_size))))
        while n_test + n_val > remaining_arr.size:
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        test_idx = remaining_arr[:n_test]
        val_idx = remaining_arr[n_test:n_test + n_val]
        train_idx.extend(int(i) for i in remaining_arr[n_test + n_val:])
        if val_idx.size == 0:
            val_idx = test_idx[:1]
        if test_idx.size == 0:
            test_idx = val_idx[:1]

    train_arr = np.asarray(sorted(set(train_idx)), dtype=int)
    val_arr = np.asarray(val_idx, dtype=int)
    test_arr = np.asarray(test_idx, dtype=int)

    print("[WARN] Split per IMMAGINE non possibile:", reason)
    print("[WARN] Applico fallback split per ROI per permettere il confronto tra SVM, Random Forest e Gradient Boosting.")
    print("[WARN] Le metriche possono essere ottimistiche: per la tesi/valutazione finale usa un CSV con almeno 3 immagini fisiche distinte.")
    return train_arr, val_arr, test_arr


def _make_splits(
        df: pd.DataFrame,
        y: np.ndarray,
        args,
        group_col: str = "_image_group",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split per immagine; fallback per ROI solo se il CSV e' troppo piccolo.

    La modalita' normale resta quella corretta: train/validation/test divisi per
    immagine fisica, per evitare data leakage. Se pero' nel CSV ci sono meno di
    3 immagini fisiche distinte, non si possono creare tre split indipendenti;
    in quel caso viene usato un fallback per ROI con avviso esplicito.
    """
    if group_col not in df.columns:
        raise ValueError(f"Colonna di raggruppamento per lo split non trovata: {group_col}")

    rng = np.random.default_rng(int(args.seed))
    gi = (
        df.groupby(group_col)["label"]
        .agg(["max", "sum", "count"])
        .reset_index()
        .rename(columns={"max": "has_positive"})
    )
    if len(gi) < 3:
        reason = f"trovate solo {len(gi)} immagini fisiche distinte"
        return _make_roi_level_splits_for_small_dataset(y, args, reason)

    tr_g: List[str] = []
    v_g: List[str] = []
    te_g: List[str] = []
    for gl in [0, 1]:
        grps = gi[gi["has_positive"] == gl][group_col].astype(str).to_numpy()
        rng.shuffle(grps)
        n = len(grps)
        n_te = max(1, int(round(n * args.test_size))) if n >= 3 else 0
        n_va = max(1, int(round(n * args.val_size))) if n - n_te >= 3 else 0
        te_g.extend(grps[:n_te])
        v_g.extend(grps[n_te:n_te + n_va])
        tr_g.extend(grps[n_te + n_va:])

    gv = df[group_col].astype(str)
    train_idx = np.flatnonzero(gv.isin(tr_g).to_numpy())
    val_idx = np.flatnonzero(gv.isin(v_g).to_numpy())
    test_idx = np.flatnonzero(gv.isin(te_g).to_numpy())

    if any(len(idx) == 0 for idx in (train_idx, val_idx, test_idx)):
        # Fallback consentito: rimescola le IMMAGINI, mai le ROI singole.
        # Utile quando uno strato (solo immagini positive o solo negative) e' piccolo.
        groups = gi[group_col].astype(str).to_numpy()
        rng.shuffle(groups)
        n_groups = len(groups)
        if n_groups < 3:
            reason = f"trovate solo {n_groups} immagini fisiche distinte"
            return _make_roi_level_splits_for_small_dataset(y, args, reason)
        n_te = max(1, int(round(n_groups * args.test_size)))
        n_va = max(1, int(round(n_groups * args.val_size)))
        while n_te + n_va >= n_groups:
            if n_va > 1:
                n_va -= 1
            elif n_te > 1:
                n_te -= 1
            else:
                reason = "percentuali validation/test incompatibili con il numero di immagini"
                return _make_roi_level_splits_for_small_dataset(y, args, reason)
        te_g = groups[:n_te]
        v_g = groups[n_te:n_te + n_va]
        tr_g = groups[n_te + n_va:]
        train_idx = np.flatnonzero(gv.isin(tr_g).to_numpy())
        val_idx = np.flatnonzero(gv.isin(v_g).to_numpy())
        test_idx = np.flatnonzero(gv.isin(te_g).to_numpy())
        print("[WARN] Split stratificato per immagine non possibile: applicato split casuale per immagini, senza dividere ROI.")

    _assert_no_image_leakage(df, train_idx, val_idx, test_idx, group_col)
    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────




def _resolve_csv_feature_key(value: Any) -> str:
    """Restituisce una chiave leggibile per un CSV feature, usando alias e nome file."""
    raw = str(value or "").strip().strip('"').strip("'")
    key = _normalize_feature_alias(raw)
    if key in FEATURE_CSV_FILES:
        return key
    # fallback da nome file
    stem = Path(raw.replace("\\", "/")).stem.lower()
    stem = stem.replace("-", "_").replace(" ", "_")
    stem = re.sub(r"_+", "_", stem)
    mapped = FEATURE_CSV_ALIASES.get(stem, stem)
    if mapped in FEATURE_CSV_FILES:
        return mapped
    return mapped or "csv_feature"


def _csv_feature_file_list(value: Any) -> List[Tuple[str, Path]]:
    """Lista dei CSV feature da confrontare.

    Accetta alias brevi, nomi file o path completi separati da virgola o punto e virgola.
    Default richiesto:
      fratture_mancate_FN, GLCM, GLCM+LBP+HOG, HOG, LBP.
    """
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"all", "tutti", "default"}:
        items = DEFAULT_COMPARE_CSV_FEATURE_KEYS
    else:
        items = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]
    out: List[Tuple[str, Path]] = []
    seen: set[str] = set()
    for item in items:
        key = _resolve_csv_feature_key(item)
        path = resolve_feature_csv_path(item)
        label_key = key if key in CSV_FEATURE_LABELS else Path(str(path)).stem
        unique = str(path).lower()
        if unique in seen:
            continue
        seen.add(unique)
        out.append((label_key, path))
    return out


def _feature_set_for_csv_file(csv_key: str, df: pd.DataFrame) -> str:
    """Sceglie automaticamente quali colonne usare in base al CSV feature.

    Il nome del CSV rimane il vero 'feature set' della relazione; questo valore
    serve solo internamente per selezionare le colonne corrette dal DataFrame.
    """
    key = str(csv_key or "").lower()
    default = CSV_FEATURE_DEFAULT_INTERNAL_SET.get(key)
    if default:
        return default
    has_glcm = any(c.startswith("glcm_") for c in df.columns)
    has_lbp = any(c.startswith("lbp_") for c in df.columns)
    has_hog = any(c.startswith("hog_") for c in df.columns)
    if has_glcm and has_lbp and has_hog:
        return "texture"
    if has_glcm and has_lbp:
        return "glcm_lbp"
    if has_glcm:
        return "glcm"
    if has_hog:
        return "hog"
    if has_lbp:
        return "lbp"
    return "texture_score"


def _csv_feature_label(csv_key: str, csv_path: Path) -> str:
    return CSV_FEATURE_LABELS.get(str(csv_key), Path(str(csv_path)).stem)


def _fmt_dur(seconds: float) -> str:
    """Formatta una durata come m:ss oppure h:mm:ss per i messaggi di avanzamento."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _progress_bar(done: int, total: int, width: int = 22) -> str:
    """Barra testuale di avanzamento tipo [#######---------------]."""
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_training_progress(done: int, total: int, current_key: str, start_time: float) -> None:
    """Stato percentuale dell'addestramento.

    Su terminale aggiorna SEMPRE la stessa riga (carriage return \\r), cosi' la
    percentuale "scorre" senza riempire lo schermo. Se l'output e' rediretto su
    file/log (non e' un terminale) stampa invece una riga per esperimento.
    `done` = numero di esperimenti gia' completati prima di quello in corso.
    """
    elapsed = time.time() - start_time
    pct = 100.0 * done / max(1, total)
    eta_txt = f"  ETA {_fmt_dur(elapsed / done * (total - done))}" if done > 0 else ""
    key_disp = current_key if len(current_key) <= 44 else current_key[:41] + "..."
    line = (
        f"  {pct:5.1f}% {_progress_bar(done, total, 16)} "
        f"[{done + 1}/{total}] {key_disp}{eta_txt}"
    )
    if sys.stdout.isatty():
        # \r torna a inizio riga; \x1b[K cancella eventuali residui a destra.
        sys.stdout.write("\r" + line + "\x1b[K")
        sys.stdout.flush()
    else:
        print(line, flush=True)


def _print_training_done(total: int, start_time: float) -> None:
    """Chiude la riga di avanzamento a 100% con il tempo totale impiegato."""
    line = (
        f"  100.0% {_progress_bar(total, total, 16)} "
        f"[{total}/{total}] esperimenti completati in {_fmt_dur(time.time() - start_time)}"
    )
    if sys.stdout.isatty():
        # Sovrascrive l'ultima riga di avanzamento e va a capo definitivamente.
        sys.stdout.write("\r" + line + "\x1b[K\n")
        sys.stdout.flush()
    else:
        print(line, flush=True)


def run_train_csv_feature_file_comparison(args) -> None:
    """Training principale per il confronto richiesto da Daniela.

    Diversamente dal vecchio --compare-feature-sets, qui il confronto viene fatto
    tra CSV feature differenti prodotti dal C++:
      fratture_mancate_FN, roi_feature_glcm, roi_feature_glcm_lbp_hog,
      roi_feature_hog, roi_feature_lbp.
    Ogni CSV viene valutato con gli stessi modelli supervised richiesti.
    """
    global MAX_THRESHOLD_POINTS
    MAX_THRESHOLD_POINTS = max(10, int(getattr(args, "threshold_max_points", MAX_THRESHOLD_POINTS)))
    if bool(getattr(args, "fast", False)):
        MAX_THRESHOLD_POINTS = min(MAX_THRESHOLD_POINTS, 150)
        if int(getattr(args, "pca_components", 30)) > 30:
            args.pca_components = 30
        if int(getattr(args, "svm_cache_mb", 500)) > 500:
            args.svm_cache_mb = 500

    ensure_output_folders()
    args.out_model = str(_resolve_output_file(getattr(args, "out_model", ""), DEFAULT_MODEL_PATH))
    args.report = str(_resolve_output_file(getattr(args, "report", ""), DEFAULT_REPORT_PATH))
    args.threshold_grid_csv = str(_resolve_output_file(getattr(args, "threshold_grid_csv", ""), DEFAULT_THRESHOLD_GRID_CSV))
    args.test_predictions = str(_resolve_output_file(getattr(args, "test_predictions", ""), DEFAULT_TEST_PREDICTIONS_CSV))
    args.split_images_dir = str(_resolve_output_dir(getattr(args, "split_images_dir", ""), OUTPUT_SPLIT_IMAGES_DIR))

    args.seed = _make_run_seed(getattr(args, "seed", None))
    beta = float(args.beta)
    beta_key = f"F{beta:g}"
    min_recall = None if float(args.min_recall) < 0 else float(args.min_recall)
    min_precision = None if float(args.min_precision) < 0 else float(args.min_precision)

    image_dir = Path(args.image_dir) if getattr(args, "image_dir", "") else DEFAULT_IMAGE_DIR
    label_dir = Path(args.label_dir) if getattr(args, "label_dir", "") else (image_dir / "labels")
    if not image_dir.exists():
        raise FileNotFoundError(f"Cartella immagini non trovata: {image_dir}")
    if not label_dir.exists():
        print(f"[WARN] Cartella labels non trovata: {label_dir}")
        print("       Il training continua; verranno copiati gli split senza label associate.")

    csv_feature_files = _csv_feature_file_list(getattr(args, "csv_feature_files", ""))
    if not csv_feature_files:
        raise ValueError("Nessun CSV feature selezionato per il confronto.")

    # Default richiesto: solo questi modelli supervised.
    classifiers = _classifier_list(getattr(args, "classifiers", ""), getattr(args, "model_set", "extended"))
    if not getattr(args, "classifiers", ""):
        classifiers = REQUESTED_SUPERVISED_CLASSIFIERS.copy()
    allowed_requested = set(REQUESTED_SUPERVISED_CLASSIFIERS)
    classifiers = [c for c in classifiers if c in allowed_requested]
    if not classifiers:
        raise ValueError("Nessun classificatore valido. Usa: SVM_RBF,SVM_LINEAR,KNN,RANDOM_FOREST,SOFT_VOTING,STACKING")

    full_preprocess = bool(getattr(args, "compare_preprocessing", False))
    normalizations = _normalization_list(getattr(args, "normalizations", ""), full=full_preprocess)
    if bool(getattr(args, "fast", False)):
        # In rapido tengo il confronto tra CSV ma riduco il costo dei modelli.
        normalizations = ["standard"]

    print(f"[*] Seed usato per questa esecuzione: {args.seed}")
    print("[*] Cartelle di salvataggio ordinate:")
    print(f"    Modello: {OUTPUT_MODEL_DIR}")
    print(f"    Report : {OUTPUT_REPORT_DIR}")
    print(f"    CSV    : {OUTPUT_CSV_DIR}")
    print(f"    Split  : {OUTPUT_SPLIT_IMAGES_DIR}")
    print("\n[*] Confronto richiesto tra CSV feature:")
    for k, pth in csv_feature_files:
        print(f"    - {_csv_feature_label(k, pth)}: {pth}")
    print("[*] Modelli supervised confrontati:", ", ".join(classifiers))
    print(f"[*] Tuning iper-parametri: livello={_tuning_level(args)}; il test set non viene usato per scegliere i parametri.")
    print(f"[*] Definizione TP ML: se best_iou esiste uso best_iou >= {TP_IOU_THRESHOLD:.2f}; se manca uso label 0/1 gia' presente nel CSV.")
    print(f"[*] Target soglia: beta={beta:g}, min_recall={min_recall}, min_precision={min_precision}")

    comparison_rows: List[dict] = []
    model_results: Dict[str, dict] = {}
    best_tuple = None
    best_payload: Optional[Dict[str, Any]] = None
    experiment_counter = 0
    experiment_total = len(csv_feature_files) * len(normalizations) * len(classifiers)
    _train_t0 = time.time()
    print(f"\n[*] Avvio addestramento: {experiment_total} esperimenti totali da completare.")

    for csv_key, csv_path in csv_feature_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV feature non trovato: {csv_path}")
        csv_label = _csv_feature_label(csv_key, csv_path)
        print("\n" + "=" * 86)
        print(f"CSV FEATURE SET: {csv_label}")
        print("=" * 86)
        print(f"[*] Carico {csv_path} …")

        try:
            raw_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"[WARN] Salto {csv_label}: CSV vuoto, nessuna riga/colonna da usare per il training.")
            comparison_rows.append({
                "experiment": f"{csv_key}__SKIPPED_EMPTY_CSV",
                "csv_feature_key": csv_key,
                "csv_feature_set": csv_label,
                "csv_path": str(csv_path),
                "status": "SKIPPED",
                "error": "CSV vuoto: nessuna feature e nessuna label disponibile.",
            })
            continue
        except Exception as exc:
            print(f"[WARN] Salto {csv_label}: impossibile leggere il CSV ({exc}).")
            comparison_rows.append({
                "experiment": f"{csv_key}__SKIPPED_READ_ERROR",
                "csv_feature_key": csv_key,
                "csv_feature_set": csv_label,
                "csv_path": str(csv_path),
                "status": "SKIPPED",
                "error": f"Errore lettura CSV: {exc}",
            })
            continue

        if raw_df.empty or raw_df.shape[1] == 0:
            print(f"[WARN] Salto {csv_label}: CSV vuoto o senza colonne utili.")
            comparison_rows.append({
                "experiment": f"{csv_key}__SKIPPED_EMPTY_CSV",
                "csv_feature_key": csv_key,
                "csv_feature_set": csv_label,
                "csv_path": str(csv_path),
                "status": "SKIPPED",
                "error": "CSV vuoto o senza colonne utili.",
            })
            continue

        try:
            df = _apply_iou_tp_rule(raw_df, TP_IOU_THRESHOLD)
        except ValueError as exc:
            print(f"[WARN] Salto {csv_label}: {exc}")
            comparison_rows.append({
                "experiment": f"{csv_key}__SKIPPED_NO_LABEL",
                "csv_feature_key": csv_key,
                "csv_feature_set": csv_label,
                "csv_path": str(csv_path),
                "status": "SKIPPED",
                "error": str(exc),
            })
            continue

        y = df["label"].to_numpy(dtype=int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if len(np.unique(y)) < 2:
            print(f"[WARN] Salto {csv_label}: servono sia label=0 sia label=1. Trovati pos={n_pos}, neg={n_neg}.")
            continue
        if args.group_col not in df.columns:
            raise ValueError(f"Colonna immagine non trovata nel CSV {csv_path}: {args.group_col}")

        image_dirs_for_split = [image_dir]
        df["_image_group"] = df[args.group_col].apply(
            lambda value: _canonical_image_group(value, csv_path, image_dirs_for_split)
        )
        print(f"    ROI={len(y)} | positivi={n_pos} | negativi={n_neg} | immagini={df['_image_group'].nunique()}")

        train_idx, val_idx, test_idx = _make_splits(df, y, args, group_col="_image_group")
        y_tr = y[train_idx]
        y_va = y[val_idx]
        y_te = y[test_idx]
        print(f"[*] Split {csv_label}: train={len(train_idx)} | validation={len(val_idx)} | test={len(test_idx)}")
        print(f"    Train originale → pos={int((y_tr == 1).sum())} neg={int((y_tr == 0).sum())}")
        print(f"    Val            → pos={int((y_va == 1).sum())} neg={int((y_va == 0).sum())}")
        print(f"    Test           → pos={int((y_te == 1).sum())} neg={int((y_te == 0).sum())}")

        dummy_train = np.zeros((len(y_tr), 1), dtype=np.float64)
        _dummy_bal, y_tr_bal, sampled_train_local_idx = _undersample_training_set(
            dummy_train, y_tr, seed=int(args.seed), ratio=float(args.undersample_ratio)
        )
        sampled_train_global_idx = train_idx[sampled_train_local_idx]
        print(f"    Train dopo undersampling → pos={int((y_tr_bal == 1).sum())} neg={int((y_tr_bal == 0).sum())} totale={len(y_tr_bal)}")

        internal_feature_set = _feature_set_for_csv_file(csv_key, df)
        feature_cols, derived_features, method_levels, X_raw, feature_names = _feature_spec_for_set(df, internal_feature_set)
        print(f"    Colonne feature usate internamente: {internal_feature_set} → {X_raw.shape[1]} feature")

        X_tr_bal = X_raw[sampled_train_global_idx]
        X_va = X_raw[val_idx]
        X_te = X_raw[test_idx]

        for normalization in normalizations:
            for classifier in classifiers:
                experiment_counter += 1
                use_pca = bool(int(getattr(args, "pca_components", 30)) > 0) and (
                    classifier in {"SVM_RBF", "SVM_LINEAR", "KNN"}
                    or bool(getattr(args, "pca_for_trees", False))
                )
                key = f"{csv_key}__{normalization}__{classifier}"
                _print_training_progress(experiment_counter - 1, experiment_total, key, _train_t0)
                try:
                    pipeline, thr, val_metrics, tuning_report, tuning_candidates, use_pca_effective = _train_tuned_estimator(
                        classifier=classifier,
                        args=args,
                        X_tr_bal=X_tr_bal,
                        y_tr_bal=y_tr_bal,
                        X_va=X_va,
                        y_va=y_va,
                        beta=beta,
                        min_recall=min_recall,
                        min_precision=min_precision,
                        normalization=normalization,
                    )
                    te_scores = _get_scores(pipeline, X_te)
                    test_metrics = _compute_metrics(y_te, te_scores, thr, beta)
                    row = {
                        "experiment": key,
                        "csv_feature_key": csv_key,
                        "csv_feature_set": csv_label,
                        "csv_path": str(csv_path),
                        "internal_feature_set": internal_feature_set,
                        "classifier": classifier,
                        "normalization": normalization,
                        "use_pca": bool(use_pca_effective),
                        "tuning_level": _tuning_level(args),
                        "tuning_candidates": int(tuning_candidates),
                        "tuning_best_params": json.dumps(tuning_report.get("best_params", {}), ensure_ascii=False),
                        "ensemble_base_models": json.dumps(tuning_report.get("base_models", {}), ensure_ascii=False, default=str),
                        "feature_count": int(X_raw.shape[1]),
                        f"val_{beta_key}": val_metrics[beta_key],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall_sensitivity_TPR"],
                        "val_F1": val_metrics["F1"],
                        "val_ROC_AUC": val_metrics["ROC_AUC"],
                        "val_PR_AP": val_metrics["PR_AP"],
                        "val_accuracy": val_metrics["accuracy"],
                        "test_precision": test_metrics["precision"],
                        "test_recall": test_metrics["recall_sensitivity_TPR"],
                        "test_F1": test_metrics["F1"],
                        f"test_{beta_key}": test_metrics[beta_key],
                        "test_ROC_AUC": test_metrics["ROC_AUC"],
                        "test_PR_AP": test_metrics["PR_AP"],
                        "test_accuracy": test_metrics["accuracy"],
                        "threshold": round(float(thr), 6),
                        "status": "OK",
                    }
                    comparison_rows.append(row)
                    model_results[key] = {
                        "threshold": float(thr),
                        "validation_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "csv_feature_set": csv_label,
                        "csv_path": str(csv_path),
                        "classifier": classifier,
                        "normalization": normalization,
                        "internal_feature_set": internal_feature_set,
                        "tuning_level": _tuning_level(args),
                        "tuning_candidates": int(tuning_candidates),
                        "tuning_report": tuning_report,
                    }
                    score_tuple = (
                        float(val_metrics[beta_key]),
                        float(val_metrics["recall_sensitivity_TPR"]),
                        float(val_metrics["precision"]),
                        float(val_metrics["ROC_AUC"]),
                        float(test_metrics["PR_AP"]),
                    )
                    if best_tuple is None or score_tuple > best_tuple:
                        best_tuple = score_tuple
                        best_payload = {
                            "key": key,
                            "csv_key": csv_key,
                            "csv_label": csv_label,
                            "csv_path": csv_path,
                            "df": df,
                            "y": y,
                            "train_idx": train_idx,
                            "val_idx": val_idx,
                            "test_idx": test_idx,
                            "sampled_train_global_idx": sampled_train_global_idx,
                            "pipeline": pipeline,
                            "classifier": classifier,
                            "normalization": normalization,
                            "use_pca": bool(use_pca_effective),
                            "tuning_level": _tuning_level(args),
                            "tuning_candidates": int(tuning_candidates),
                            "tuning_report": tuning_report,
                            "feature_cols": feature_cols,
                            "derived_features": derived_features,
                            "method_levels": method_levels,
                            "feature_names": feature_names,
                            "feature_count": int(X_raw.shape[1]),
                            "internal_feature_set": internal_feature_set,
                            "threshold": float(thr),
                            "val_metrics": val_metrics,
                            "test_metrics": test_metrics,
                        }
                except Exception as exc:
                    print(f"\n      [WARN] Esperimento fallito: {exc}")
                    comparison_rows.append({
                        "experiment": key,
                        "csv_feature_key": csv_key,
                        "csv_feature_set": csv_label,
                        "csv_path": str(csv_path),
                        "classifier": classifier,
                        "normalization": normalization,
                        "status": "ERROR",
                        "error": str(exc),
                    })

    _print_training_done(experiment_total, _train_t0)

    valid_rows = [r for r in comparison_rows if r.get("status") == "OK"]
    if not valid_rows or best_payload is None:
        raise RuntimeError("Nessun esperimento supervisionato valido completato.")

    valid_rows.sort(
        key=lambda r: (
            float(r.get(f"val_{beta_key}", 0.0)),
            float(r.get("val_recall", 0.0)),
            float(r.get("val_precision", 0.0)),
            float(r.get("val_ROC_AUC", 0.0)),
        ),
        reverse=True,
    )
    rank_map = {r["experiment"]: i for i, r in enumerate(valid_rows, start=1)}
    for r in comparison_rows:
        if r.get("status") == "OK":
            r["rank"] = rank_map.get(r.get("experiment"), "")
            r["selected"] = "BEST" if r.get("experiment") == best_payload["key"] else ""
        else:
            r["rank"] = ""
            r["selected"] = ""

    print("\n" + "=" * 120)
    print("CONFRONTO CSV FEATURE + MODELLI SUPERVISED SU VALIDATION E TEST")
    print("=" * 120)
    header = (
        f"{'Rank':>4}  {'CSV feature':<22}  {'Modello':<16}  {'Norm':<9}  "
        f"{'Val_'+beta_key:>9}  {'Val_P':>7}  {'Val_R':>7}  {'Val_F1':>7}  {'Val_ROC':>8}  {'Val_AP':>8}  "
        f"{'Test_P':>7}  {'Test_R':>7}  {'Test_F1':>7}  {'Test_AP':>8}  {'Thr':>8}  Sel"
    )
    print(header)
    print("-" * len(header))
    for row in valid_rows[:50]:
        label = str(row.get("csv_feature_set", ""))[:22]
        print(
            f"{int(row['rank']):>4}  {label:<22}  {row['classifier']:<16}  {row['normalization']:<9}  "
            f"{float(row[f'val_{beta_key}']):>9.4f}  {float(row['val_precision']):>7.4f}  {float(row['val_recall']):>7.4f}  "
            f"{float(row['val_F1']):>7.4f}  {float(row['val_ROC_AUC']):>8.4f}  {float(row['val_PR_AP']):>8.4f}  "
            f"{float(row['test_precision']):>7.4f}  {float(row['test_recall']):>7.4f}  {float(row['test_F1']):>7.4f}  "
            f"{float(row['test_PR_AP']):>8.4f}  {float(row['threshold']):>8.4f}  {row.get('selected', '')}"
        )

    best_key = str(best_payload["key"])
    best_pipeline: Pipeline = best_payload["pipeline"]
    sel_thr = float(best_payload["threshold"])
    val_metrics = best_payload["val_metrics"]
    test_metrics = best_payload["test_metrics"]
    cm = test_metrics["confusion_matrix"]

    print(f"\n[*] Modello finale selezionato: {best_key}")
    print(f"    CSV feature={best_payload['csv_label']} | classificatore={best_payload['classifier']} | norm={best_payload['normalization']} | PCA={best_payload['use_pca']}")
    print(f"    Validation {beta_key}={val_metrics[beta_key]:.4f} precision={val_metrics['precision']:.4f} recall={val_metrics['recall_sensitivity_TPR']:.4f} ROC={val_metrics['ROC_AUC']:.4f}")
    print(f"    Soglia selezionata: {sel_thr:.6f}")
    print("\n[*] Performance finali sul TEST SET:")
    print(f"    TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}")
    print(f"    Accuracy    = {test_metrics['accuracy']:.4f}")
    print(f"    Precision   = {test_metrics['precision']:.4f}")
    print(f"    Recall      = {test_metrics['recall_sensitivity_TPR']:.4f}")
    print(f"    Specificity = {test_metrics['specificity_TNR']:.4f}")
    print(f"    F1          = {test_metrics['F1']:.4f}")
    print(f"    {beta_key}       = {test_metrics[beta_key]:.4f}")
    print(f"    ROC AUC     = {test_metrics['ROC_AUC']:.4f}")
    print(f"    PR AP       = {test_metrics['PR_AP']:.4f}")

    # Contesto del vincitore per salvataggi compatibili.
    df = best_payload["df"]
    y = best_payload["y"]
    train_idx = best_payload["train_idx"]
    val_idx = best_payload["val_idx"]
    test_idx = best_payload["test_idx"]
    sampled_train_global_idx = best_payload["sampled_train_global_idx"]
    feature_cols = list(best_payload["feature_cols"])
    derived_features = list(best_payload["derived_features"])
    method_levels = list(best_payload["method_levels"])
    feature_names = list(best_payload["feature_names"])
    X_raw, _ = _extract_features(df, feature_cols, derived_features, method_levels)
    X_va = X_raw[val_idx]
    X_te = X_raw[test_idx]
    y_va = y[val_idx]
    y_te = y[test_idx]

    if bool(getattr(args, "skip_split_copy", False)):
        split_image_artifacts = {}
        print("[*] Copia immagini split saltata (--skip-split-copy).")
    else:
        split_image_artifacts = _save_split_images(
            df=df,
            split_indices={"train": train_idx, "train_undersampled": sampled_train_global_idx, "validation": val_idx, "test": test_idx},
            csv_path=Path(best_payload["csv_path"]),
            group_col=args.group_col,
            out_root=Path(args.split_images_dir),
            image_dirs=[image_dir],
            label_dir=label_dir,
        )

    split_metrics: Dict[str, dict] = {}
    split_info: Dict[str, dict] = {}
    for sn, idx_arr in [("train_original", train_idx), ("train_undersampled", sampled_train_global_idx), ("validation", val_idx), ("test", test_idx)]:
        XX = X_raw[idx_arr]
        yy = y[idx_arr]
        scores = _get_scores(best_pipeline, XX)
        split_metrics[sn] = _compute_metrics(yy, scores, sel_thr, beta)
        split_info[sn] = {
            "rows": int(len(idx_arr)),
            "class_counts": {
                "negative_0": int((yy == 0).sum()),
                "positive_1": int((yy == 1).sum()),
                "total": int(len(yy)),
            },
            "images": int(df.iloc[idx_arr]["_image_group"].nunique()),
            "image_groups": sorted(df.iloc[idx_arr]["_image_group"].astype(str).unique().tolist()),
        }

    best_te_scores = _get_scores(best_pipeline, X_te)
    pred_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    tpdf = df.iloc[test_idx][pred_cols].copy()
    tpdf["image_group"] = df.iloc[test_idx]["_image_group"].to_numpy()
    tpdf["ml_model"] = best_payload["classifier"]
    tpdf["ml_experiment"] = best_key
    tpdf["ml_csv_feature_set"] = best_payload["csv_label"]
    tpdf["ml_csv_path"] = str(best_payload["csv_path"])
    tpdf["ml_score"] = best_te_scores
    tpdf["ml_prediction"] = (best_te_scores >= sel_thr).astype(int)
    tpdf["ml_keep_candidate"] = tpdf["ml_prediction"].map({1: "keep", 0: "reject"})
    tp_path = Path(args.test_predictions)
    tp_path.parent.mkdir(parents=True, exist_ok=True)
    tp_path = _safe_to_csv(tpdf, tp_path, index=False)
    print(f"[*] Predizioni test salvate: {tp_path}")

    grid_rows = []
    grid_rows += _threshold_sweep_rows(y_va, _get_scores(best_pipeline, X_va), beta, "validation")
    grid_rows += _threshold_sweep_rows(y_te, best_te_scores, beta, "test")
    grid_path = Path(args.threshold_grid_csv)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path = _safe_to_csv(pd.DataFrame(grid_rows).sort_values(["split", "F1", "recall", "precision"], ascending=[True, False, False, False]), grid_path, index=False)
    print(f"[*] Grid soglie ML ROI-level salvata: {grid_path}")

    comp_path = DEFAULT_MODEL_COMPARISON_CSV
    exp_path = DEFAULT_EXPERIMENT_COMPARISON_CSV
    feature_summary_path = DEFAULT_FEATURESET_COMPARISON_CSV
    exp_path = _safe_to_csv(pd.DataFrame(comparison_rows).sort_values(["rank"], key=lambda s: pd.to_numeric(s, errors="coerce").fillna(999999)), exp_path, index=False)
    comp_path = _safe_to_csv(pd.DataFrame(comparison_rows).sort_values(["rank"], key=lambda s: pd.to_numeric(s, errors="coerce").fillna(999999)), comp_path, index=False)
    print(f"[*] Tabella comparativa completa: {exp_path}")
    print(f"[*] Tabella comparativa compatibile: {comp_path}")

    feature_summary = []
    for csv_key, csv_path in csv_feature_files:
        label = _csv_feature_label(csv_key, csv_path)
        fs_rows = [r for r in valid_rows if r.get("csv_feature_key") == csv_key and str(r.get("csv_path")) == str(csv_path)]
        if fs_rows:
            b = fs_rows[0]
            feature_summary.append({
                "csv_feature_key": csv_key,
                "csv_feature_set": label,
                "csv_path": str(csv_path),
                "best_experiment": b["experiment"],
                "best_classifier": b["classifier"],
                "best_normalization": b["normalization"],
                "feature_count": b["feature_count"],
                f"best_val_{beta_key}": b[f"val_{beta_key}"],
                "best_val_precision": b["val_precision"],
                "best_val_recall": b["val_recall"],
                "best_val_F1": b["val_F1"],
                "best_test_precision": b["test_precision"],
                "best_test_recall": b["test_recall"],
                "best_test_F1": b["test_F1"],
                "best_test_PR_AP": b["test_PR_AP"],
            })
    feature_summary_path = _safe_to_csv(pd.DataFrame(feature_summary), feature_summary_path, index=False)
    print(f"[*] Sintesi per CSV feature set: {feature_summary_path}")

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    package = {
        "version": 8,
        "style": "sklearn_csv_feature_file_comparison_tuning_ensemble",
        "feature_spec": {
            "feature_set": best_payload["internal_feature_set"],
            "csv_feature_set": best_payload["csv_label"],
            "csv_path": str(best_payload["csv_path"]),
            "columns": feature_cols,
            "derived": derived_features,
            "method_levels": method_levels,
        },
        "feature_names": feature_names,
        "model_type": "single_best",
        "selected_name": best_key,
        "selected_names": [best_key],
        "pipelines": {best_key: best_pipeline},
        "threshold": float(sel_thr),
        "thresholds_by_model": {k: float(v["threshold"]) for k, v in model_results.items()},
        "beta": float(beta),
        "min_recall": None if min_recall is None else float(min_recall),
        "min_precision": None if min_precision is None else float(min_precision),
        "selected_experiment": {
            "key": best_key,
            "classifier": best_payload["classifier"],
            "csv_feature_key": best_payload["csv_key"],
            "csv_feature_set": best_payload["csv_label"],
            "csv_path": str(best_payload["csv_path"]),
            "normalization": best_payload["normalization"],
            "use_pca": bool(best_payload["use_pca"]),
            "tuning_level": best_payload.get("tuning_level", _tuning_level(args)),
            "tuning_candidates": int(best_payload.get("tuning_candidates", 0)),
            "tuning_report": best_payload.get("tuning_report", {}),
            "feature_count": int(best_payload["feature_count"]),
        },
        "undersampling": {
            "enabled": True,
            "ratio_negative_to_positive": float(args.undersample_ratio),
            "train_original_counts": {"negative_0": int((y[train_idx] == 0).sum()), "positive_1": int((y[train_idx] == 1).sum())},
            "train_used_counts": {"negative_0": int((y[sampled_train_global_idx] == 0).sum()), "positive_1": int((y[sampled_train_global_idx] == 1).sum())},
        },
        "model_results": model_results,
        "tp_definition": {
            "source_column": "best_iou",
            "comparison": TP_IOU_RULE,
            "iou_threshold": float(TP_IOU_THRESHOLD),
            "rule": f"best_iou >= {TP_IOU_THRESHOLD:.2f}",
        },
        "image_identity": "sha256_del_contenuto; fallback_nome_normalizzato_se_file_non_trovato",
        "split_image_groups": {
            "train": sorted(df.iloc[train_idx]["_image_group"].astype(str).unique().tolist()),
            "validation": sorted(df.iloc[val_idx]["_image_group"].astype(str).unique().tolist()),
            "test": sorted(df.iloc[test_idx]["_image_group"].astype(str).unique().tolist()),
        },
    }
    out_model = _safe_pickle_dump(package, out_model)
    print(f"[*] Modello salvato: {out_model}")

    report = {
        "dataset": {
            "selected_csv": str(best_payload["csv_path"]),
            "selected_csv_feature_set": best_payload["csv_label"],
            "rows": int(len(df)),
            "images": int(df["_image_group"].nunique()),
            "image_identity": "sha256_del_contenuto; fallback_nome_normalizzato_se_file_non_trovato",
            "class_counts": {"negative_0": int((y == 0).sum()), "positive_1": int((y == 1).sum()), "total": int(len(y))},
            "tp_definition": {
                "source_column": "best_iou",
                "comparison": TP_IOU_RULE,
                "iou_threshold": float(TP_IOU_THRESHOLD),
                "rule": f"best_iou >= {TP_IOU_THRESHOLD:.2f}",
                "iou_equal_to_threshold_is_tp": True,
            },
        },
        "experiment_plan": {
            "csv_feature_files": [{"key": k, "label": _csv_feature_label(k, p), "path": str(p)} for k, p in csv_feature_files],
            "classifiers": classifiers,
            "normalizations": normalizations,
            "compare_preprocessing": full_preprocess,
            "tuning_level": _tuning_level(args),
            "base_classifiers": BASE_SUPERVISED_CLASSIFIERS,
            "ensemble_classifiers": ENSEMBLE_SUPERVISED_CLASSIFIERS,
            "run_unsupervised": False,
        },
        "features": {
            "selected_csv_feature_set": best_payload["csv_label"],
            "selected_internal_feature_set": best_payload["internal_feature_set"],
            "selected_feature_count": int(X_raw.shape[1]),
            "glcm_count": sum(n.startswith("glcm_") for n in feature_names),
            "lbp_count": sum(n.startswith("lbp_") for n in feature_names),
            "hog_count": sum(n.startswith("hog_") for n in feature_names),
        },
        "split": split_info,
        "undersampling": package["undersampling"],
        "selected_model": {
            "type": "single_best",
            "name": best_key,
            "classifier": best_payload["classifier"],
            "csv_feature_set": best_payload["csv_label"],
            "tuning_level": best_payload.get("tuning_level", _tuning_level(args)),
            "tuning_candidates": int(best_payload.get("tuning_candidates", 0)),
            "threshold": float(sel_thr),
            "beta": float(beta),
            "min_recall_constraint": min_recall,
            "min_precision_constraint": min_precision,
        },
        "classifier_comparison": comparison_rows,
        "csv_feature_set_comparison": feature_summary,
        "metrics": split_metrics,
        "all_model_results": model_results,
        "operating_points": {
            "validation": _operating_points(y_va, _get_scores(best_pipeline, X_va), beta),
            "test": _operating_points(y_te, best_te_scores, beta),
        },
        "artifacts": {
            "model": str(out_model),
            "test_predictions": str(tp_path),
            "threshold_grid_csv": str(grid_path),
            "comparison_table": str(comp_path),
            "experiment_comparison": str(exp_path),
            "feature_set_comparison": str(feature_summary_path),
            "split_images": split_image_artifacts,
        },
    }
    out_rep = Path(args.report)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_rep = _safe_write_text_file(out_rep, json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[*] Report JSON: {out_rep}")
    print(json.dumps({
        "selected_model": report["selected_model"],
        "test_metrics_final_model": split_metrics["test"],
        "artifacts": report["artifacts"],
    }, indent=2))

def run_train(args) -> None:
    # Modalita' predefinita della v9: confronto tra i 5 CSV feature richiesti.
    # Usa --single-csv-mode per tornare alla vecchia modalita' su un solo CSV.
    if bool(getattr(args, "compare_csv_feature_files", True)) and not bool(getattr(args, "single_csv_mode", False)):
        run_train_csv_feature_file_comparison(args)
        return

    global MAX_THRESHOLD_POINTS
    MAX_THRESHOLD_POINTS = max(10, int(getattr(args, "threshold_max_points", MAX_THRESHOLD_POINTS)))
    if bool(getattr(args, "fast", False)):
        MAX_THRESHOLD_POINTS = min(MAX_THRESHOLD_POINTS, 120)
        if int(getattr(args, "pca_components", 30)) > 30:
            args.pca_components = 30
        if int(getattr(args, "svm_cache_mb", 500)) > 500:
            args.svm_cache_mb = 500

    ensure_output_folders()
    args.out_model = str(_resolve_output_file(getattr(args, "out_model", ""), DEFAULT_MODEL_PATH))
    args.report = str(_resolve_output_file(getattr(args, "report", ""), DEFAULT_REPORT_PATH))
    args.threshold_grid_csv = str(_resolve_output_file(getattr(args, "threshold_grid_csv", ""), DEFAULT_THRESHOLD_GRID_CSV))
    args.test_predictions = str(_resolve_output_file(getattr(args, "test_predictions", ""), DEFAULT_TEST_PREDICTIONS_CSV))
    args.split_images_dir = str(_resolve_output_dir(getattr(args, "split_images_dir", ""), OUTPUT_SPLIT_IMAGES_DIR))

    args.seed = _make_run_seed(getattr(args, "seed", None))
    print(f"[*] Seed usato per questa esecuzione: {args.seed}")

    print("[*] Cartelle di salvataggio ordinate:")
    print(f"    Modello: {OUTPUT_MODEL_DIR}")
    print(f"    Report : {OUTPUT_REPORT_DIR}")
    print(f"    CSV    : {OUTPUT_CSV_DIR}")
    print(f"    Split  : {OUTPUT_SPLIT_IMAGES_DIR}")

    beta = float(args.beta)
    beta_key = f"F{beta:g}"
    min_recall = None if float(args.min_recall) < 0 else float(args.min_recall)
    min_precision = None if float(args.min_precision) < 0 else float(args.min_precision)

    csv_path = resolve_feature_csv_path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    image_dir = Path(args.image_dir) if getattr(args, "image_dir", "") else DEFAULT_IMAGE_DIR
    label_dir = Path(args.label_dir) if getattr(args, "label_dir", "") else (image_dir / "labels")

    print(f"[*] Carico {csv_path} …")
    print(f"[*] Cartella immagini impostata: {image_dir}")
    print(f"[*] Cartella label impostata: {label_dir}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Cartella immagini non trovata: {image_dir}")
    if not label_dir.exists():
        print(f"[WARN] Cartella labels non trovata: {label_dir}")
        print("       Il training continua; verranno copiati gli split senza label associate.")

    df = _apply_iou_tp_rule(pd.read_csv(csv_path), TP_IOU_THRESHOLD)
    print(f"[*] Definizione TP ML: se best_iou esiste uso best_iou >= {TP_IOU_THRESHOLD:.2f}; se manca uso label 0/1 gia' presente nel CSV.")

    y = df["label"].to_numpy(dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if len(np.unique(y)) < 2:
        raise ValueError("Servono sia label=0 che label=1.")

    if args.group_col not in df.columns:
        raise ValueError(f"Colonna immagine non trovata nel CSV: {args.group_col}")
    image_dirs_for_split = [image_dir]
    df["_image_group"] = df[args.group_col].apply(
        lambda value: _canonical_image_group(value, csv_path, image_dirs_for_split)
    )

    print(f"    {len(y)} ROI | {n_pos} positivi | {n_neg} negativi")
    print(f"    Immagini fisiche distinte per split: {df['_image_group'].nunique()}")

    train_idx, val_idx, test_idx = _make_splits(df, y, args, group_col="_image_group")
    y_tr = y[train_idx]
    y_va = y[val_idx]
    y_te = y[test_idx]

    print(f"[*] Split train={len(train_idx)} | validation={len(val_idx)} | test={len(test_idx)}")
    print(f"    Train originale → pos={int((y_tr == 1).sum())} neg={int((y_tr == 0).sum())}")
    print(f"    Val            → pos={int((y_va == 1).sum())} neg={int((y_va == 0).sum())}")
    print(f"    Test           → pos={int((y_te == 1).sum())} neg={int((y_te == 0).sum())}")

    # Undersampling definito una volta sola sul train: stesso campione per tutti i feature set.
    dummy_train = np.zeros((len(y_tr), 1), dtype=np.float64)
    _dummy_bal, y_tr_bal, sampled_train_local_idx = _undersample_training_set(
        dummy_train, y_tr, seed=int(args.seed), ratio=float(args.undersample_ratio)
    )
    sampled_train_global_idx = train_idx[sampled_train_local_idx]
    print(f"    Train dopo undersampling → pos={int((y_tr_bal == 1).sum())} neg={int((y_tr_bal == 0).sum())} totale={len(y_tr_bal)}")

    if bool(getattr(args, "skip_split_copy", False)):
        split_image_artifacts = {}
        print("[*] Copia immagini split saltata (--skip-split-copy): training più rapido.")
    else:
        split_image_artifacts = _save_split_images(
            df=df,
            split_indices={"train": train_idx, "train_undersampled": sampled_train_global_idx, "validation": val_idx, "test": test_idx},
            csv_path=csv_path,
            group_col=args.group_col,
            out_root=Path(args.split_images_dir),
            image_dirs=[image_dir],
            label_dir=label_dir,
        )

    compare_feature_sets = bool(getattr(args, "compare_feature_sets", False)) or str(getattr(args, "feature_set", "")).lower() in {"all", "tutti", "full"}
    feature_sets = _csv_list(getattr(args, "feature_sets", ""), DEFAULT_FULL_FEATURE_SETS) if compare_feature_sets else [str(args.feature_set).lower()]
    feature_sets = [fs for fs in feature_sets if fs in AVAILABLE_FEATURE_SETS]
    if not feature_sets:
        raise ValueError("Nessun feature set valido selezionato.")

    full_preprocess = bool(getattr(args, "compare_preprocessing", False))
    normalizations = _normalization_list(getattr(args, "normalizations", ""), full=full_preprocess)
    classifiers = _classifier_list(getattr(args, "classifiers", ""), getattr(args, "model_set", "extended"))
    if bool(getattr(args, "fast", False)):
        classifiers = [c for c in classifiers if c in DEFAULT_CORE_CLASSIFIERS]
        normalizations = ["standard"]
        if compare_feature_sets and len(feature_sets) > 3:
            feature_sets = ["glcm", "texture", "texture_score"]

    print("\n[*] Piano esperimenti supervisionati:")
    print(f"    Feature set     : {', '.join(feature_sets)}")
    print(f"    Normalizzazioni : {', '.join(normalizations)}")
    print(f"    Classificatori  : {', '.join(classifiers)}")
    print(f"    PCA componenti  : {int(args.pca_components)} (0 = no PCA)")
    print(f"    Soglie max      : {MAX_THRESHOLD_POINTS}")
    print(f"    Target soglia   : beta={beta:g}, min_recall={min_recall}, min_precision={min_precision}")

    model_results: Dict[str, dict] = {}
    comparison_rows: List[dict] = []
    pipelines: Dict[str, Pipeline] = {}
    feature_specs_by_key: Dict[str, dict] = {}

    experiment_total = len(feature_sets) * len(normalizations) * len(classifiers)
    experiment_counter = 0
    _train_t0 = time.time()
    print(f"\n[*] Avvio addestramento: {experiment_total} esperimenti totali da completare.")

    for feature_set in feature_sets:
        feature_cols, derived_features, method_levels, X_raw, feature_names = _feature_spec_for_set(df, feature_set)
        print(f"\n[*] Feature set '{feature_set}' → {X_raw.shape[1]} feature")
        X_tr = X_raw[train_idx]
        X_va = X_raw[val_idx]
        X_te = X_raw[test_idx]
        X_tr_bal = X_raw[sampled_train_global_idx]
        for normalization in normalizations:
            for classifier in classifiers:
                experiment_counter += 1
                use_pca = bool(int(getattr(args, "pca_components", 30)) > 0) and (
                    classifier in {"SVM_RBF", "SVM_LINEAR", "LOGISTIC_REGRESSION", "KNN"}
                    or bool(getattr(args, "pca_for_trees", False))
                )
                key = f"{feature_set}__{normalization}__{classifier}"
                _print_training_progress(experiment_counter - 1, experiment_total, key, _train_t0)
                try:
                    pipeline, thr, val_metrics, tuning_report, tuning_candidates, use_pca_effective = _train_tuned_estimator(
                        classifier=classifier,
                        args=args,
                        X_tr_bal=X_tr_bal,
                        y_tr_bal=y_tr_bal,
                        X_va=X_va,
                        y_va=y_va,
                        beta=beta,
                        min_recall=min_recall,
                        min_precision=min_precision,
                        normalization=normalization,
                    )
                    te_scores = _get_scores(pipeline, X_te)
                    test_metrics = _compute_metrics(y_te, te_scores, thr, beta)

                    pipelines[key] = pipeline
                    feature_specs_by_key[key] = {
                        "feature_set": feature_set,
                        "columns": feature_cols,
                        "derived": derived_features,
                        "method_levels": method_levels,
                        "feature_names": feature_names,
                        "feature_count": int(X_raw.shape[1]),
                        "normalization": normalization,
                        "classifier": classifier,
                        "use_pca": bool(use_pca_effective),
                        "tuning_level": _tuning_level(args),
                        "tuning_candidates": int(tuning_candidates),
                        "tuning_report": tuning_report,
                    }
                    model_results[key] = {
                        "classifier": classifier,
                        "feature_set": feature_set,
                        "normalization": normalization,
                        "use_pca": bool(use_pca_effective),
                        "tuning_level": _tuning_level(args),
                        "tuning_candidates": int(tuning_candidates),
                        "tuning_best_params": json.dumps(tuning_report.get("best_params", {}), ensure_ascii=False),
                        "ensemble_base_models": json.dumps(tuning_report.get("base_models", {}), ensure_ascii=False, default=str),
                        "feature_count": int(X_raw.shape[1]),
                        "threshold": float(thr),
                        "validation_metrics": val_metrics,
                        "test_metrics": test_metrics,
                    }
                    comparison_rows.append({
                        "experiment": key,
                        "classifier": classifier,
                        "feature_set": feature_set,
                        "normalization": normalization,
                        "pca": "yes" if use_pca_effective else "no",
                        "feature_count": int(X_raw.shape[1]),
                        f"val_{beta_key}": val_metrics[beta_key],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall_sensitivity_TPR"],
                        "val_F1": val_metrics["F1"],
                        "val_ROC_AUC": val_metrics["ROC_AUC"],
                        "val_PR_AP": val_metrics["PR_AP"],
                        "val_accuracy": val_metrics["accuracy"],
                        "test_precision": test_metrics["precision"],
                        "test_recall": test_metrics["recall_sensitivity_TPR"],
                        "test_F1": test_metrics["F1"],
                        f"test_{beta_key}": test_metrics[beta_key],
                        "test_ROC_AUC": test_metrics["ROC_AUC"],
                        "test_PR_AP": test_metrics["PR_AP"],
                        "test_accuracy": test_metrics["accuracy"],
                        "threshold": round(float(thr), 6),
                        "error": "",
                    })
                except Exception as exc:
                    print(f"\n      [WARN] Esperimento saltato: {exc}")
                    comparison_rows.append({
                        "experiment": key, "classifier": classifier, "feature_set": feature_set,
                        "normalization": normalization, "pca": "?", "feature_count": 0,
                        "error": str(exc),
                    })

    _print_training_done(experiment_total, _train_t0)

    valid_rows = [r for r in comparison_rows if not r.get("error") and f"val_{beta_key}" in r]
    if not valid_rows:
        raise RuntimeError("Nessun esperimento supervisionato completato correttamente.")

    valid_rows = sorted(
        valid_rows,
        key=lambda r: (r[f"val_{beta_key}"], r["val_recall"], r["val_precision"], r["val_ROC_AUC"], r["test_PR_AP"]),
        reverse=True,
    )
    rank_by_experiment = {row["experiment"]: i for i, row in enumerate(valid_rows, start=1)}
    for row in comparison_rows:
        if row.get("experiment") in rank_by_experiment:
            row["rank"] = rank_by_experiment[row["experiment"]]
            row["selected"] = "BEST" if row["rank"] == 1 else ""
        else:
            row["rank"] = ""
            row["selected"] = ""

    best_row = valid_rows[0]
    best_key = best_row["experiment"]
    best_name = str(best_row["classifier"])
    best_pipeline = pipelines[best_key]
    sel_thr = float(model_results[best_key]["threshold"])
    val_metrics = model_results[best_key]["validation_metrics"]
    test_metrics = model_results[best_key]["test_metrics"]
    cm = test_metrics["confusion_matrix"]
    best_spec = feature_specs_by_key[best_key]

    print("\n[*] Top esperimenti supervisionati su VALIDATION e TEST:")
    header = (
        f"{'Rank':>4}  {'Esperimento':<44}  {'Val_'+beta_key:>10}  {'Val_P':>8}  {'Val_R':>8}  "
        f"{'Val_F1':>8}  {'Val_AP':>8}  {'Test_P':>8}  {'Test_R':>8}  {'Test_F1':>8}  {'Test_AP':>8}  {'Thr':>8}  Sel"
    )
    print(header)
    print("-" * len(header))
    for row in valid_rows[:15]:
        print(
            f"{int(rank_by_experiment[row['experiment']]):>4}  "
            f"{row['experiment'][:44]:<44}  "
            f"{float(row[f'val_{beta_key}']):>10.4f}  "
            f"{float(row['val_precision']):>8.4f}  "
            f"{float(row['val_recall']):>8.4f}  "
            f"{float(row['val_F1']):>8.4f}  "
            f"{float(row['val_PR_AP']):>8.4f}  "
            f"{float(row['test_precision']):>8.4f}  "
            f"{float(row['test_recall']):>8.4f}  "
            f"{float(row['test_F1']):>8.4f}  "
            f"{float(row['test_PR_AP']):>8.4f}  "
            f"{float(row['threshold']):>8.4f}  "
            f"{row.get('selected', '')}"
        )

    print(f"\n[*] Modello finale selezionato: {best_key}")
    print(f"    Classificatore={best_name} | feature_set={best_spec['feature_set']} | norm={best_spec['normalization']} | PCA={best_spec['use_pca']}")
    print(f"    Validation {beta_key}={val_metrics[beta_key]:.4f} precision={val_metrics['precision']:.4f} recall={val_metrics['recall_sensitivity_TPR']:.4f} ROC={val_metrics['ROC_AUC']:.4f}")
    print(f"    Soglia selezionata: {sel_thr:.6f}")

    print("\n[*] Performance finali sul TEST SET:")
    print(f"    TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}")
    print(f"    Accuracy    = {test_metrics['accuracy']:.4f}")
    print(f"    Precision   = {test_metrics['precision']:.4f}")
    print(f"    Recall      = {test_metrics['recall_sensitivity_TPR']:.4f}")
    print(f"    Specificity = {test_metrics['specificity_TNR']:.4f}")
    print(f"    F1          = {test_metrics['F1']:.4f}")
    print(f"    {beta_key}       = {test_metrics[beta_key]:.4f}")
    print(f"    ROC AUC     = {test_metrics['ROC_AUC']:.4f}")
    print(f"    PR AP       = {test_metrics['PR_AP']:.4f}")

    # Ricostruisce X del feature set vincente per predizioni, grid soglie e metriche split.
    feature_cols = list(best_spec["columns"])
    derived_features = list(best_spec["derived"])
    method_levels = list(best_spec["method_levels"])
    X_raw, feature_names = _extract_features(df, feature_cols, derived_features, method_levels)
    X_va = X_raw[val_idx]
    X_te = X_raw[test_idx]

    split_metrics: Dict[str, dict] = {}
    split_info: Dict[str, dict] = {}
    for sn, idx_arr in [("train_original", train_idx), ("train_undersampled", sampled_train_global_idx), ("validation", val_idx), ("test", test_idx)]:
        XX = X_raw[idx_arr]
        yy = y[idx_arr]
        scores = _get_scores(best_pipeline, XX)
        split_metrics[sn] = _compute_metrics(yy, scores, sel_thr, beta)
        split_info[sn] = {
            "rows": int(len(idx_arr)),
            "class_counts": {
                "negative_0": int((yy == 0).sum()),
                "positive_1": int((yy == 1).sum()),
                "total": int(len(yy)),
            },
            "images": int(df.iloc[idx_arr]["_image_group"].nunique()),
            "image_groups": sorted(df.iloc[idx_arr]["_image_group"].astype(str).unique().tolist()),
        }

    best_te_scores = _get_scores(best_pipeline, X_te)
    pred_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    tpdf = df.iloc[test_idx][pred_cols].copy()
    tpdf["image_group"] = df.iloc[test_idx]["_image_group"].to_numpy()
    tpdf["ml_model"] = best_name
    tpdf["ml_experiment"] = best_key
    tpdf["ml_feature_set"] = best_spec["feature_set"]
    tpdf["ml_normalization"] = best_spec["normalization"]
    tpdf["ml_score"] = best_te_scores
    tpdf["ml_prediction"] = (best_te_scores >= sel_thr).astype(int)
    tpdf["ml_keep_candidate"] = tpdf["ml_prediction"].map({1: "keep", 0: "reject"})

    tp_path = Path(args.test_predictions)
    tp_path.parent.mkdir(parents=True, exist_ok=True)
    tp_path = _safe_to_csv(tpdf, tp_path, index=False)
    print(f"[*] Predizioni test salvate: {tp_path}")

    grid_rows = []
    grid_rows += _threshold_sweep_rows(y_va, _get_scores(best_pipeline, X_va), beta, "validation")
    grid_rows += _threshold_sweep_rows(y_te, best_te_scores, beta, "test")
    grid_path = Path(args.threshold_grid_csv)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path = _safe_to_csv(pd.DataFrame(grid_rows).sort_values(["split", "F1", "recall", "precision"], ascending=[True, False, False, False]), grid_path, index=False)
    print(f"[*] Grid soglie ML ROI-level salvata: {grid_path}")

    comp_path = DEFAULT_MODEL_COMPARISON_CSV
    exp_path = DEFAULT_EXPERIMENT_COMPARISON_CSV
    feature_summary_path = DEFAULT_FEATURESET_COMPARISON_CSV
    exp_path = _safe_to_csv(pd.DataFrame(comparison_rows).sort_values(["rank"], key=lambda s: pd.to_numeric(s, errors="coerce").fillna(999999)), exp_path, index=False)
    # Mantengo model_comparison.csv come tabella principale, ordinata per ranking.
    comp_path = _safe_to_csv(pd.DataFrame(comparison_rows).sort_values(["rank"], key=lambda s: pd.to_numeric(s, errors="coerce").fillna(999999)), comp_path, index=False)
    print(f"[*] Tabella comparativa completa: {exp_path}")
    print(f"[*] Tabella comparativa compatibile: {comp_path}")

    feature_summary = []
    for fs in feature_sets:
        fs_rows = [r for r in valid_rows if r.get("feature_set") == fs]
        if fs_rows:
            b = fs_rows[0]
            feature_summary.append({
                "feature_set": fs,
                "best_experiment": b["experiment"],
                "best_classifier": b["classifier"],
                "best_normalization": b["normalization"],
                "feature_count": b["feature_count"],
                f"best_val_{beta_key}": b[f"val_{beta_key}"],
                "best_val_precision": b["val_precision"],
                "best_val_recall": b["val_recall"],
                "best_val_F1": b["val_F1"],
                "best_test_precision": b["test_precision"],
                "best_test_recall": b["test_recall"],
                "best_test_F1": b["test_F1"],
                "best_test_PR_AP": b["test_PR_AP"],
            })
    feature_summary_path = _safe_to_csv(pd.DataFrame(feature_summary), feature_summary_path, index=False)
    print(f"[*] Sintesi per feature set: {feature_summary_path}")

    unsupervised_rows = _run_unsupervised_experiments(df, y, train_idx, test_idx, feature_sets, args)
    unsup_path = DEFAULT_UNSUPERVISED_COMPARISON_CSV
    if unsupervised_rows:
        unsup_path = _safe_to_csv(pd.DataFrame(unsupervised_rows), unsup_path, index=False)
        print(f"[*] Confronto unsupervised esplorativo: {unsup_path}")

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    package = {
        "version": 7,
        "style": "sklearn_multi_method_feature_set_comparison_tuning_ensemble",
        "feature_spec": {
            "feature_set": best_spec["feature_set"],
            "columns": feature_cols,
            "derived": derived_features,
            "method_levels": method_levels,
        },
        "feature_names": feature_names,
        "model_type": "single_best",
        "selected_name": best_key,
        "selected_names": [best_key],
        # Salvo solo il modello vincente nel campo usato dalla GUI, per evitare mismatch di feature tra modelli diversi.
        "pipelines": {best_key: best_pipeline},
        "threshold": float(sel_thr),
        "thresholds_by_model": {k: float(v["threshold"]) for k, v in model_results.items()},
        "beta": float(beta),
        "min_recall": None if min_recall is None else float(min_recall),
        "min_precision": None if min_precision is None else float(min_precision),
        "selected_experiment": {
            "key": best_key,
            "classifier": best_name,
            "feature_set": best_spec["feature_set"],
            "normalization": best_spec["normalization"],
            "use_pca": bool(best_spec["use_pca"]),
            "feature_count": int(best_spec["feature_count"]),
        },
        "undersampling": {
            "enabled": True,
            "ratio_negative_to_positive": float(args.undersample_ratio),
            "train_original_counts": {"negative_0": int((y_tr == 0).sum()), "positive_1": int((y_tr == 1).sum())},
            "train_used_counts": {"negative_0": int((y_tr_bal == 0).sum()), "positive_1": int((y_tr_bal == 1).sum())},
        },
        "model_results": model_results,
        "tp_definition": {
            "source_column": "best_iou",
            "comparison": TP_IOU_RULE,
            "iou_threshold": float(TP_IOU_THRESHOLD),
            "rule": f"best_iou >= {TP_IOU_THRESHOLD:.2f}",
        },
        "image_identity": "sha256_del_contenuto; fallback_nome_normalizzato_se_file_non_trovato",
        "split_image_groups": {
            "train": sorted(df.iloc[train_idx]["_image_group"].astype(str).unique().tolist()),
            "validation": sorted(df.iloc[val_idx]["_image_group"].astype(str).unique().tolist()),
            "test": sorted(df.iloc[test_idx]["_image_group"].astype(str).unique().tolist()),
        },
    }

    out_model = _safe_pickle_dump(package, out_model)
    print(f"[*] Modello salvato: {out_model}")

    report = {
        "dataset": {
            "csv": str(csv_path),
            "rows": int(len(df)),
            "images": int(df["_image_group"].nunique()),
            "image_identity": "sha256_del_contenuto; fallback_nome_normalizzato_se_file_non_trovato",
            "class_counts": {"negative_0": n_neg, "positive_1": n_pos, "total": int(len(y))},
            "tp_definition": {
                "source_column": "best_iou",
                "comparison": TP_IOU_RULE,
                "iou_threshold": float(TP_IOU_THRESHOLD),
                "rule": f"best_iou >= {TP_IOU_THRESHOLD:.2f}",
                "iou_equal_to_threshold_is_tp": True,
            },
        },
        "experiment_plan": {
            "feature_sets": feature_sets,
            "normalizations": normalizations,
            "classifiers": classifiers,
            "compare_feature_sets": compare_feature_sets,
            "compare_preprocessing": full_preprocess,
            "run_unsupervised": bool(getattr(args, "run_unsupervised", False)),
        },
        "features": {
            "selected_feature_set": best_spec["feature_set"],
            "selected_feature_count": int(X_raw.shape[1]),
            "glcm_count": sum(n.startswith("glcm_") for n in feature_names),
            "lbp_count": sum(n.startswith("lbp_") for n in feature_names),
            "hog_count": sum(n.startswith("hog_") for n in feature_names),
        },
        "split": split_info,
        "undersampling": package["undersampling"],
        "selected_model": {
            "type": "single_best",
            "name": best_name,
            "experiment": best_key,
            "feature_set": best_spec["feature_set"],
            "normalization": best_spec["normalization"],
            "use_pca": bool(best_spec["use_pca"]),
            "threshold": float(sel_thr),
            "beta": float(beta),
            "min_recall_constraint": min_recall,
            "min_precision_constraint": min_precision,
        },
        "classifier_comparison": comparison_rows,
        "feature_set_summary": feature_summary,
        "unsupervised_comparison": unsupervised_rows,
        "metrics": split_metrics,
        "all_model_results": model_results,
        "operating_points": {
            "validation": _operating_points(y_va, _get_scores(best_pipeline, X_va), beta),
            "test": _operating_points(y_te, best_te_scores, beta),
        },
        "artifacts": {
            "model": str(out_model),
            "test_predictions": str(tp_path),
            "threshold_grid_csv": str(grid_path),
            "comparison_table": str(comp_path),
            "experiment_comparison_table": str(exp_path),
            "feature_set_comparison_table": str(feature_summary_path),
            "unsupervised_comparison_table": str(unsup_path) if unsupervised_rows else "",
            "split_images": split_image_artifacts,
        },
    }

    out_rep = Path(args.report)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_rep = _safe_write_text_file(out_rep, json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[*] Report JSON: {out_rep}")

    print(json.dumps({
        "selected_model": report["selected_model"],
        "test_metrics_final_model": split_metrics["test"],
        "artifacts": report["artifacts"],
    }, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────────────────────────────────────

def run_predict(args) -> None:
    ensure_output_folders()
    args.model = str(_resolve_output_file(getattr(args, "model", ""), DEFAULT_MODEL_PATH))
    args.out = str(_resolve_output_file(getattr(args, "out", ""), DEFAULT_PREDICTIONS_CSV))
    args.filtered_out = str(_resolve_output_file(getattr(args, "filtered_out", ""), DEFAULT_KEPT_CANDIDATES_CSV))

    csv_path = resolve_feature_csv_path(args.csv)
    model_path = Path(args.model)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    df = pd.read_csv(csv_path)
    has_iou_for_metrics = "best_iou" in df.columns
    if has_iou_for_metrics:
        df = _apply_iou_tp_rule(df, TP_IOU_THRESHOLD)
        print(f"[*] Metriche predizione con TP definito da best_iou >= {TP_IOU_THRESHOLD:.2f} (IoU = 0.50 inclusa).")
    elif "label" in df.columns:
        print("[WARN] Colonna best_iou assente: salvo le predizioni, ma non calcolo TP/FP/FN con la nuova regola IoU.")
    with model_path.open("rb") as f:
        pkg = pickle.load(f)

    fs = pkg["feature_spec"]
    X_raw, _ = _extract_features(df, fs["columns"], fs.get("derived", []), fs.get("method_levels", []))

    selected_name = pkg.get("selected_name") or pkg.get("selected_names", [None])[0]
    if selected_name is None:
        raise ValueError("Nel file modello non trovo il nome del modello selezionato.")

    thr = float(args.threshold) if args.threshold is not None else float(pkg["threshold"])
    beta = float(pkg.get("beta", 1.0))
    pipeline = pkg["pipelines"][selected_name]
    scores = _get_scores(pipeline, X_raw)

    pred_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    out_df = df[pred_cols].copy()
    out_df["ml_model"] = selected_name
    out_df["ml_score"] = scores
    out_df["ml_prediction"] = (scores >= thr).astype(int)
    out_df["ml_keep_candidate"] = out_df["ml_prediction"].map({1: "keep", 0: "reject"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = _safe_to_csv(out_df, out_path, index=False)

    if args.filtered_out:
        fp = Path(args.filtered_out)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp = _safe_to_csv(out_df[out_df["ml_prediction"] == 1], fp, index=False)

    if has_iou_for_metrics:
        metrics = _compute_metrics(df["label"].astype(int).to_numpy(), scores, thr, beta)
        print(json.dumps({
            "selected_model": selected_name,
            "threshold": thr,
            "metrics": metrics,
            "out": str(out_path),
            "filtered_out": args.filtered_out,
        }, indent=2))
    else:
        print(json.dumps({
            "selected_model": selected_name,
            "threshold": thr,
            "rows": int(len(out_df)),
            "kept": int((out_df["ml_prediction"] == 1).sum()),
            "rejected": int((out_df["ml_prediction"] == 0).sum()),
            "out": str(out_path),
            "filtered_out": args.filtered_out,
        }, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _add_train_args(tp):
    tp.add_argument("--csv", default="glcm_lbp_hog",
                    help="CSV singolo da usare solo con --single-csv-mode. In v9 il default confronta i 5 CSV feature.")
    tp.add_argument("--csv-feature-files", default="fn,glcm,glcm_lbp_hog,hog,lbp",
                    help="CSV feature da confrontare, separati da virgola o punto e virgola. Default: fratture_mancate_FN, GLCM, GLCM+LBP+HOG, HOG, LBP.")
    tp.add_argument("--compare-csv-feature-files", action="store_true", default=True,
                    help="Confronta i CSV feature prodotti dal C++; attivo di default nella v9.")
    tp.add_argument("--single-csv-mode", action="store_true",
                    help="Disattiva il confronto tra CSV feature e usa la vecchia modalita' su un solo CSV indicato con --csv.")
    tp.add_argument("--out-model", default=str(DEFAULT_MODEL_PATH))
    tp.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    tp.add_argument("--threshold-grid-csv", default=str(DEFAULT_THRESHOLD_GRID_CSV),
                    help="CSV con tutte le soglie ML ROI-level provate su validation e test")
    tp.add_argument("--test-predictions", default=str(DEFAULT_TEST_PREDICTIONS_CSV))
    tp.add_argument("--split-images-dir", default=str(OUTPUT_SPLIT_IMAGES_DIR),
                    help="Cartella dove copiare le immagini usate per train/validation/test.")
    tp.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR),
                    help="Cartella dove cercare le immagini originali da copiare negli split.")
    tp.add_argument("--label-dir", default=str(DEFAULT_LABEL_DIR),
                    help="Cartella labels YOLO associata alle immagini. Se non esiste, il training continua senza copiare label.")
    tp.add_argument("--feature-set", default="texture_score",
                    choices=["glcm", "lbp", "hog", "glcm_lbp", "texture", "texture_score", "all"],
                    help="Feature set singolo. Usa 'all' oppure --compare-feature-sets per confrontarli tutti.")
    tp.add_argument("--compare-feature-sets", action="store_true",
                    help="Vecchia modalita': confronta set di colonne dentro un singolo CSV. Usa insieme a --single-csv-mode.")
    tp.add_argument("--feature-sets", default="glcm,lbp,hog,glcm_lbp,texture,texture_score",
                    help="Lista feature set da confrontare, separati da virgola. Usato con --compare-feature-sets.")
    tp.add_argument("--model-set", default="extended", choices=["core", "extended", "all"],
                    help="Set modelli. Default: SVM_RBF, SVM_LINEAR, KNN, RANDOM_FOREST, SOFT_VOTING, STACKING.")
    tp.add_argument("--classifiers", default="",
                    help="Lista classificatori da usare, es. SVM_RBF,RANDOM_FOREST,KNN,SOFT_VOTING,STACKING. Vuoto = secondo --model-set.")
    tp.add_argument("--tuning-level", default="quick", choices=["none", "fast", "quick", "full"],
                    help="Livello di tuning iper-parametri: none, fast, quick o full. Default: quick.")
    tp.add_argument("--compare-preprocessing", action="store_true",
                    help="Confronta anche normalizzazioni diverse: none, standard, minmax.")
    tp.add_argument("--normalizations", default="",
                    help="Lista normalizzazioni separate da virgola: none,standard,minmax,robust. Vuoto = standard o set completo con --compare-preprocessing.")
    tp.add_argument("--pca-for-trees", action="store_true",
                    help="Applica PCA anche ai modelli ad albero. Default: PCA solo per SVM/LR/KNN.")
    tp.add_argument("--run-unsupervised", action="store_true",
                    help="Esegue confronto esplorativo K-Means/GMM. Non viene usato come modello finale GUI.")
    tp.add_argument("--unsupervised-clusters", type=int, default=2,
                    help="Numero cluster/componenti per K-Means/GMM. Per ROI frattura default 2: frattura/non frattura.")
    tp.add_argument("--unsupervised-pca", type=int, default=30,
                    help="Componenti PCA per confronto unsupervised. 0 = no PCA.")
    tp.add_argument("--group-col", default="image")
    tp.add_argument("--val-size", type=float, default=0.20)
    tp.add_argument("--test-size", type=float, default=0.20)
    tp.add_argument("--seed", type=int, default=None,
                    help="Seed per rendere ripetibile lo split. Default: casuale a ogni esecuzione.")
    # Default: criterio che preferisce l'F2 (beta=2, piu' peso al recall, nessun vincolo minimo).
    tp.add_argument("--beta", type=float, default=2.0)
    tp.add_argument("--min-recall", type=float, default=-1.0)
    tp.add_argument("--min-precision", type=float, default=-1.0)
    tp.add_argument("--undersample-ratio", type=float, default=1.0,
                    help="Rapporto negativi/positivi nel training dopo undersampling. 1.0 = bilanciato.")
    tp.add_argument("--n-jobs", type=int, default=-1)

    # SVM
    tp.add_argument("--svm-kernel", default="rbf", choices=["rbf", "linear", "poly", "sigmoid"])
    tp.add_argument("--svm-c", type=float, default=10.0)
    tp.add_argument("--svm-gamma", default="scale")
    tp.add_argument("--svm-degree", type=int, default=3)
    tp.add_argument("--svm-coef0", type=float, default=1.0)
    tp.add_argument("--svm-cache-mb", type=int, default=1000)

    # Random Forest
    tp.add_argument("--rf-n-estimators", type=int, default=300,
                    help="Numero alberi Random Forest. Default 300.")
    tp.add_argument("--rf-max-depth", type=int, default=0,
                    help="Profondita' massima Random Forest. 0 = nessun limite.")
    tp.add_argument("--rf-min-samples-leaf", type=int, default=1,
                    help="Minimo campioni per foglia Random Forest. Default 1.")

    # Gradient Boosting
    tp.add_argument("--gb-n-estimators", type=int, default=200,
                    help="Numero stimatori Gradient Boosting. Default 200.")
    tp.add_argument("--gb-learning-rate", type=float, default=0.05,
                    help="Learning rate Gradient Boosting. Default 0.05.")
    tp.add_argument("--gb-max-depth", type=int, default=3,
                    help="Profondita' massima dei piccoli alberi nel Gradient Boosting. Default 3.")

    # Altri modelli del confronto metodologico
    tp.add_argument("--knn-neighbors", type=int, default=7,
                    help="Numero vicini KNN. Default 7.")
    tp.add_argument("--knn-weights", default="distance", choices=["uniform", "distance"],
                    help="Peso KNN. Default distance.")
    tp.add_argument("--logreg-max-iter", type=int, default=3000,
                    help="Iterazioni massime Logistic Regression.")
    tp.add_argument("--xgb-n-estimators", type=int, default=250,
                    help="Numero alberi XGBoost, se installato.")
    tp.add_argument("--xgb-max-depth", type=int, default=3,
                    help="Profondita' XGBoost, se installato.")
    tp.add_argument("--xgb-learning-rate", type=float, default=0.05,
                    help="Learning rate XGBoost, se installato.")

    # PCA per SVM
    tp.add_argument("--pca-components", type=int, default=30,
                    help="Componenti PCA per SVM. Usa 0 per disattivare PCA. Default 30 per ridurre i tempi.")
    tp.add_argument("--threshold-max-points", type=int, default=300,
                    help="Numero massimo di soglie da provare negli sweep ML. Default 300 per non rallentare.")
    tp.add_argument("--skip-split-copy", action="store_true",
                    help="Non copia immagini/label negli split: più veloce, ma non aggiorna ml/split_images.")
    tp.add_argument("--fast", action="store_true",
                    help="Profilo rapido: max 150 soglie, PCA al massimo 30, cache SVM più piccola.")


def _interactive_main():
    print("\n===============================================================")
    print("  MODALITÀ INTERATTIVA - CONFRONTO / TRAINING ML")
    print("===============================================================")
    print("Seleziona l'operazione da eseguire:")
    print("  [1] Addestra e confronta più metodologie ML supervised/unsupervised")
    print("  [2] Addestra modello operativo migliore: SVM_RBF")
    print("  [3] Applica il modello finale a un nuovo CSV")
    print("  [4] Esci")

    scelta = ""
    while scelta not in ["1", "2", "3", "4"]:
        try:
            scelta = input("\nInserisci la tua scelta (1-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUscita.")
            sys.exit(0)

    if scelta == "4":
        print("Arrivederci!")
        sys.exit(0)

    class InteractiveArgs:
        pass

    args = InteractiveArgs()

    def _set_training_defaults(prompt_paths=True):
        args.command = "train"
        args.out_model = str(DEFAULT_MODEL_PATH)
        args.report = str(DEFAULT_REPORT_PATH)
        args.threshold_grid_csv = str(DEFAULT_THRESHOLD_GRID_CSV)
        args.test_predictions = str(DEFAULT_TEST_PREDICTIONS_CSV)
        args.split_images_dir = str(OUTPUT_SPLIT_IMAGES_DIR)
        if prompt_paths:
            image_dir_in = input(f"Cartella immagini originali [default: {DEFAULT_IMAGE_DIR}]: ").strip()
            label_dir_in = input(f"Cartella labels YOLO [default: {DEFAULT_LABEL_DIR}]: ").strip()
            args.image_dir = image_dir_in if image_dir_in else str(DEFAULT_IMAGE_DIR)
            args.label_dir = label_dir_in if label_dir_in else str(DEFAULT_LABEL_DIR)
        else:
            args.image_dir = str(DEFAULT_IMAGE_DIR)
            args.label_dir = str(DEFAULT_LABEL_DIR)

        args.feature_set = "texture_score"
        args.feature_sets = "glcm,lbp,hog,glcm_lbp,texture,texture_score"
        args.normalizations = ""
        args.group_col = "image"
        args.val_size = 0.20
        args.test_size = 0.20
        args.seed = None  # None = combinazioni diverse a ogni esecuzione
        args.n_jobs = -1
        args.threshold_max_points = 300
        args.skip_split_copy = False
        args.fast = False
        args.tuning_level = "quick"
        args.pca_for_trees = False
        args.unsupervised_clusters = 2
        args.unsupervised_pca = 30
        args.svm_kernel = "rbf"
        args.svm_c = 10.0
        args.svm_gamma = "scale"
        args.svm_degree = 3
        args.svm_coef0 = 1.0
        args.svm_cache_mb = 1000
        args.rf_n_estimators = 300
        args.rf_max_depth = 0
        args.rf_min_samples_leaf = 1
        args.gb_n_estimators = 200
        args.gb_learning_rate = 0.05
        args.gb_max_depth = 3
        args.pca_components = 30
        args.knn_neighbors = 7
        args.knn_weights = "distance"
        args.logreg_max_iter = 3000
        args.et_n_estimators = 300
        args.xgb_n_estimators = 250
        args.xgb_max_depth = 3
        args.xgb_learning_rate = 0.05

    def _ask_threshold_profile():
        # Criterio soglia FISSO e automatico: si preferisce sempre l'F2 (beta=2,
        # senza vincoli di recall/precision minimi), quindi il punto operativo
        # massimizza direttamente l'F2 (piu' peso al recall). Non viene piu' chiesto
        # di scegliere tra Clinical Recall / Balanced / High Precision.
        args.beta = 2.0
        args.min_recall = -1.0
        args.min_precision = -1.0
        print("\n[*] Criterio di soglia automatico: ottimizzazione dell'F2 "
              "(beta=2, piu' peso al recall, nessun vincolo minimo).")

        ratio_in = input("Rapporto negativi/positivi dopo undersampling [default: 1.0]: ").strip()
        args.undersample_ratio = float(ratio_in) if ratio_in else 1.0

    if scelta == "1":
        print("\n--- IMPOSTAZIONI ADDESTRAMENTO / CONFRONTO COMPLETO ---")
        csv_in = input("Percorso CSV feature [default: glcm_lbp_hog]: ").strip()
        args.csv = csv_in if csv_in else "glcm_lbp_hog"
        args.csv_feature_files = "fn,glcm,glcm_lbp_hog,hog,lbp"
        args.compare_csv_feature_files = True
        args.single_csv_mode = False
        _set_training_defaults(prompt_paths=True)

        full_cmp = input("Vuoi confrontare tutti i feature set? [S/n]: ").strip().lower()
        args.compare_feature_sets = (full_cmp != "n")
        prep_cmp = input("Vuoi confrontare anche le normalizzazioni? [s/N]: ").strip().lower()
        args.compare_preprocessing = (prep_cmp == "s")
        args.model_set = "extended"
        args.classifiers = ""
        unsup_cmp = input("Vuoi aggiungere K-Means/GMM unsupervised esplorativi? [s/N]: ").strip().lower()
        args.run_unsupervised = (unsup_cmp == "s")
        _ask_threshold_profile()

        # Default confronto metodologico: SVM_RBF, SVM_LINEAR, KNN, RANDOM_FOREST, SOFT_VOTING, STACKING.
        args.model_set = "extended"
        args.classifiers = "SVM_RBF,SVM_LINEAR,KNN,RANDOM_FOREST,SOFT_VOTING,STACKING"
        args.compare_feature_sets = False
        args.compare_preprocessing = False
        args.normalizations = ""
        args.run_unsupervised = False

        print(f"\n[*] Avvio training/confronto completo con CSV iniziale: {args.csv}")
        print(f"[*] CSV feature confrontati: {args.csv_feature_files}")
        print(f"[*] Modelli confrontati: {args.classifiers}")
        print(f"[*] Un TP richiede best_iou >= {TP_IOU_THRESHOLD:.2f}; IoU = 0.50 viene inclusa.\n")
        run_train(args)

    elif scelta == "2":
        print("\n--- ADDESTRAMENTO OPERATIVO SVM_RBF ---")
        print("Verrà usato in sola lettura SOLO il CSV combinato GLCM+LBP+HOG:")
        print(f"  - {CSV_FEATURE_LABELS.get('glcm_lbp_hog', 'GLCM+LBP+HOG')}: {FEATURE_CSV_FILES['glcm_lbp_hog']}")

        # Modalità operativa richiesta:
        # - NON confronta i 5 CSV feature;
        # - usa solo il CSV combinato roi_feature_glcm_lbp_hog_labeled.csv;
        # - addestra e salva solo SVM_RBF come modello operativo per GUI/evaluate.
        args.csv = "glcm_lbp_hog"
        args.csv_feature_files = "glcm_lbp_hog"
        args.compare_csv_feature_files = False
        args.single_csv_mode = True

        _set_training_defaults(prompt_paths=True)
        args.compare_feature_sets = False
        args.compare_preprocessing = False
        args.normalizations = "standard"
        args.model_set = "core"
        args.classifiers = "SVM_RBF"
        args.run_unsupervised = False
        _ask_threshold_profile()

        print("\n[*] Avvio training operativo: SOLO SVM_RBF sul CSV combinato GLCM+LBP+HOG.")
        print("[*] Gli altri modelli e gli altri CSV restano disponibili nell'opzione 1 per il confronto metodologico.")
        print(f"[*] CSV operativo: {FEATURE_CSV_FILES['glcm_lbp_hog']}")
        print(f"[*] Il modello finale verra' salvato in: {DEFAULT_MODEL_PATH}")
        print(f"[*] Un TP richiede best_iou >= {TP_IOU_THRESHOLD:.2f}; IoU = 0.50 viene inclusa.\n")
        run_train(args)

    elif scelta == "3":
        args.command = "predict"
        print("\n--- IMPOSTAZIONI PREDIZIONE ---")
        csv_in = ""
        while not csv_in:
            csv_in = input("Inserisci il percorso del CSV da analizzare: ").strip()
            if not csv_in:
                print("Errore: il percorso del CSV è obbligatorio.")
        args.csv = csv_in
        model_in = input(f"Percorso modello salvato [default: {DEFAULT_MODEL_PATH}]: ").strip()
        args.model = model_in if model_in else str(DEFAULT_MODEL_PATH)
        out_in = input(f"Percorso output predizioni [default: {DEFAULT_PREDICTIONS_CSV}]: ").strip()
        args.out = out_in if out_in else str(DEFAULT_PREDICTIONS_CSV)
        keep_in = input(f"Percorso output candidati mantenuti [default: {DEFAULT_KEPT_CANDIDATES_CSV}]: ").strip()
        args.filtered_out = keep_in if keep_in else str(DEFAULT_KEPT_CANDIDATES_CSV)
        args.threshold = None
        run_predict(args)

def main():
    if len(sys.argv) == 1:
        _interactive_main()
        return

    parser = argparse.ArgumentParser(description="Filtro ML per ROI di frattura: confronto multi-metodologia con undersampling e compatibilita GUI.")
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Addestra e confronta più feature set, preprocessing e modelli ML.")
    _add_train_args(tp)

    pp = sub.add_parser("predict", help="Applica il modello finale salvato a un nuovo CSV.")
    pp.add_argument("--csv", required=True)
    pp.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    pp.add_argument("--out", default=str(DEFAULT_PREDICTIONS_CSV))
    pp.add_argument("--filtered-out", default=str(DEFAULT_KEPT_CANDIDATES_CSV))
    pp.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
