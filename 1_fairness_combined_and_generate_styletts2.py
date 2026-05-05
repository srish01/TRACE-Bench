#!/usr/bin/env python3
# fairness_styletts2_generation_portable_refs.py
# ===============================================================
# FAIRNESS dataset generation (StyleTTS2 only)
#
# Portable experiment layout:
#   RAW_DATA_ROOT/
#     - text_all.csv
#     - CoSafe/Single Prompt/*.json
#     - VCTK-Corpus/...
#     - emotion_dataset/MEAD/...   (or RAW_DATA_ROOT/MEAD/...)
#
# Output layout:
#   OUTPUT_ROOT/
#     - references/...
#     - styletts2/<category>/*.wav
#     - manifests/<category>.jsonl
#
# Portability:
#   - Manifests store ONLY experiment-relative paths (relative to OUTPUT_ROOT)
#   - No absolute paths are stored in the manifest
#
# Fixes included:
#   1) VCTK demographics: robust speaker-info.txt parsing + normalized speaker IDs
#   2) MEAD refs: selected directly from filesystem (not from text_all.csv)
#      to avoid missing/incorrect metadata that collapses selection
#
# Added:
#   - Prints selected speakers + their labels/demographics BEFORE generation
#     * VCTK: age/gender/accent/region from speaker-info.txt
#     * MEAD: derived labels: sex (M/W), emotion, level
# ===============================================================

from __future__ import annotations

import os
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torchaudio as ta
import torchaudio.functional as TAF
import torch
from tqdm import tqdm

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------- CONFIG ----------------------------

RANDOM_SEED = int(os.getenv("RANDOM_SEED"))
random.seed(RANDOM_SEED)

# >>> IMPORTANT: set this to your portable raw_data folder <<<
RAW_DATA_ROOT = Path(os.getenv("RAW_DATA_ROOT"))

TEXT_ALL_CSV = RAW_DATA_ROOT / "text_all.csv"
COSAFE_DIR   = RAW_DATA_ROOT / "CoSafe" / "Single Prompt"

LIBRISQA_DATASET = "ZihanZhao/LibriSQA"
LIBRISQA_SPLITS = ["partI.train", "partI.test"]
BENIGN_CATEGORY_NAME = "benign_librisqa"

BASE_OUTPUT_ROOT = Path(os.getenv("BASE_OUTPUT_DIR"))

# StyleTTS2 (edit as needed)
STYLETTS2_CKPT = "epochs_2nd_00020.pth"
STYLETTS2_CFG  = "Models_LibriTTS_config.yml"

TARGET_SR = 24000

USE_HALF_COSAFE_CATEGORIES = True
COSAFE_PROMPTS_PER_CATEGORY = 50
BENIGN_MATCH_COSAFE_COUNT = True

FAIRNESS_VCTK_SPEAKERS = ["p294", "p227", "p283", "p364", "p248", "p335", "p347", "p374", "s5"]

# You finalized these:
FAIRNESS_MEAD_SPEAKERS = {
    "male":   ["M003", "M011"],
    "female": ["W011", "W038"],
}
FAIRNESS_MEAD_EMOTIONS = ["neutral", "happy", "sad", "angry"]
FAIRNESS_MEAD_LEVELS   = ["level_1", "level_2", "level_3"]

# VCTK speaker info: you can paste it here OR point to a file inside RAW_DATA_ROOT.
# If path is None and pasted is None, we auto-discover speaker-info.txt under RAW_DATA_ROOT.
VCTK_SPEAKER_INFO_TXT_PATH: Optional[Path] = None
VCTK_SPEAKER_INFO_TXT_PASTED: Optional[str] = None

# ---------------------------- UTILS ----------------------------

def ensure_unique_dir(base: Path) -> Path:
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    i = 1
    while True:
        cand = Path(f"{base}-{i}")
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        i += 1

def rel_to_out_root(path: Path, out_root: Path) -> str:
    """Return POSIX relative path under out_root (portable)."""
    return str(path.relative_to(out_root)).replace("\\", "/")

def _to_tensor_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.float()

def _resample_to(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    wav = _to_tensor_mono(wav)
    return wav if src_sr == dst_sr else TAF.resample(wav, src_sr, dst_sr)

def wav_duration_sec(path: Path) -> float:
    try:
        if not path.exists():
            return 0.0
        info = ta.info(str(path))
        if info.num_frames <= 0 or info.sample_rate <= 0:
            return 0.0
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        return 0.0

def safe_json_load_any(path: Path):
    """
    CoSafe files sometimes are:
      - a single JSON list
      - OR multiple JSON objects/arrays per line (JSONL-like)
    This loader supports both.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        out = []
        for ln in raw.splitlines():
            s = ln.strip()
            if not s:
                continue
            out.append(json.loads(s))
        return out

def resolve_audio_path(wav_fp: str) -> Path:
    """
    Robust resolver for wav_fp coming from text_all.csv.

    It may be:
      - absolute (but points to old tree)
      - relative (e.g. VCTK-Corpus/wav48/p225/xxx.wav)
      - relative with extra prefixes
    We try:
      1) If absolute and exists -> use.
      2) If relative and exists under RAW_DATA_ROOT -> use RAW_DATA_ROOT / rel.
      3) If absolute but stale -> map by suffix match starting from known dataset anchors.
    """
    s = str(wav_fp).replace("\\", "/").strip()
    p = Path(s)

    # 1) absolute existing
    if p.is_absolute() and p.exists():
        return p

    # 2) relative under RAW_DATA_ROOT
    if not p.is_absolute():
        cand = RAW_DATA_ROOT / p
        if cand.exists():
            return cand

    # 3) stale absolute or weird relative: try to re-anchor by known folders
    anchors = [
        "VCTK-Corpus/",
        "emotion_dataset/MEAD/",
        "MEAD/",
    ]
    for a in anchors:
        idx = s.find(a)
        if idx >= 0:
            suffix = s[idx:]
            cand = RAW_DATA_ROOT / suffix
            if cand.exists():
                return cand

    return (RAW_DATA_ROOT / p)

# ---------------------------- SPEAKER INFO (VCTK) ----------------------------

def _auto_find_vctk_speaker_info_txt(raw_root: Path) -> Optional[Path]:
    """
    Tries common VCTK layouts first; then falls back to rglob.
    Returns a Path if found, else None.
    """
    candidates = [
        raw_root / "VCTK-Corpus" / "speaker-info.txt",
        raw_root / "VCTK-Corpus" / "speaker_info.txt",
        raw_root / "VCTK-Corpus" / "txt" / "speaker-info.txt",
        raw_root / "VCTK-Corpus" / "txt" / "speaker_info.txt",
    ]
    for p in candidates:
        if p.exists():
            return p

    hits = list(raw_root.rglob("speaker-info.txt"))
    if hits:
        return hits[0]
    hits = list(raw_root.rglob("speaker_info.txt"))
    if hits:
        return hits[0]
    return None

def parse_vctk_speaker_info() -> Dict[str, Dict]:
    """
    Robustly parses VCTK speaker-info.txt into:
      info["p225"] = {"ID":"p225","AGE":23,"GENDER":"F","ACCENT":"English","REGION":"Southern England"}
    Handles irregular whitespace and normalizes speaker IDs to lowercase.
    """
    txt = ""

    path = VCTK_SPEAKER_INFO_TXT_PATH
    if path is None:
        path = _auto_find_vctk_speaker_info_txt(RAW_DATA_ROOT)

    if path is not None and path.exists():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        print(f" Loaded VCTK speaker-info from: {path}")
    elif VCTK_SPEAKER_INFO_TXT_PASTED:
        txt = VCTK_SPEAKER_INFO_TXT_PASTED
        print(" Loaded VCTK speaker-info from pasted text.")
    else:
        print("⚠️  VCTK speaker-info not found. VCTK AGE/GENDER/ACCENT will be None.")
        return {}

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return {}

    # Drop header if present
    if lines[0].lower().startswith("id"):
        lines = lines[1:]

    info: Dict[str, Dict] = {}
    for ln in lines:
        # split into 5 fields max: id age gender accent region(with spaces)
        parts = ln.split(None, 4)
        if len(parts) < 4:
            continue

        spk = parts[0].strip().lower()
        age = parts[1].strip() if len(parts) > 1 else None
        gender = parts[2].strip() if len(parts) > 2 else None
        accent = parts[3].strip() if len(parts) > 3 else None
        region = parts[4].strip() if len(parts) > 4 else None

        info[spk] = {
            "ID": spk,
            "AGE": int(age) if age and age.isdigit() else None,
            "GENDER": gender,
            "ACCENT": accent,
            "REGION": region,
        }

    return info

# ---------------------------- LOAD LABELS ----------------------------

def load_text_all(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"text_all.csv not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if "wav_fp" not in df.columns:
        raise ValueError(f"'wav_fp' not found in {csv_path}")
    df["wav_fp"] = df["wav_fp"].astype(str).str.replace("\\", "/", regex=False)
    if "emotion" in df.columns:
        df["emotion"] = df["emotion"].astype(str).str.lower()
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str)
    return df

def extract_speaker_id_from_wav_fp(wav_fp: str) -> Optional[str]:
    import re
    p = str(wav_fp).replace("\\", "/")

    m = re.search(r"/(p\d{3}|s5)/", p, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    m = re.search(r"/mead/([mw]\d{3})/", p, flags=re.IGNORECASE)
    if m:
        # keep MEAD IDs with uppercase prefix (M###/W###)
        return m.group(1).upper()

    return None

def extract_mead_level_from_wav_fp(wav_fp: str) -> Optional[str]:
    import re
    m = re.search(r"/(level_\d)/", str(wav_fp).lower())
    return m.group(1) if m else None

# ---------------------------- REFERENCES (SELECT) ----------------------------

def select_vctk_longest_refs(df: pd.DataFrame, speaker_info: Dict[str, Dict]) -> List[Dict]:
    vctk_df = df[df["wav_fp"].str.contains("VCTK", case=False, na=False)].copy()
    vctk_df["speaker_id"] = vctk_df["wav_fp"].apply(extract_speaker_id_from_wav_fp)
    vctk_df = vctk_df[vctk_df["speaker_id"].isin([s.lower() for s in FAIRNESS_VCTK_SPEAKERS])].copy()
    if vctk_df.empty:
        raise RuntimeError("No VCTK rows matched FAIRNESS_VCTK_SPEAKERS in text_all.csv")

    selected = []
    for spk in FAIRNESS_VCTK_SPEAKERS:
        spk_norm = spk.lower()
        cand = vctk_df[vctk_df["speaker_id"] == spk_norm].copy()
        if cand.empty:
            continue

        cand["abs_path"] = cand["wav_fp"].apply(lambda fp: resolve_audio_path(fp))
        cand["duration_sec"] = cand["abs_path"].apply(wav_duration_sec)

        cand = cand[cand["abs_path"].apply(lambda p: Path(p).exists())].copy()
        if cand.empty:
            print(f"[WARN] No existing VCTK files found under RAW_DATA_ROOT for speaker {spk}")
            continue

        best = cand.sort_values("duration_sec", ascending=False).iloc[0].to_dict()
        best["dataset"] = "VCTK"
        best["speaker_id"] = spk_norm

        # ✅ demographics lookup is normalized by lowercase key
        si = speaker_info.get(spk_norm, {})
        best["speaker_info"] = {
            "ID": spk_norm,
            "AGE": si.get("AGE"),
            "GENDER": si.get("GENDER"),
            "ACCENT": si.get("ACCENT"),
            "REGION": si.get("REGION"),
        }
        selected.append(best)

    if not selected:
        raise RuntimeError("No VCTK references selected (paths missing under RAW_DATA_ROOT).")
    return selected

def find_mead_root(raw_root: Path) -> Path:
    # common layouts
    candidates = [
        raw_root / "emotion_dataset" / "MEAD",
        raw_root / "MEAD",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: search
    hits = list(raw_root.rglob("MEAD"))
    for h in hits:
        if h.is_dir():
            return h
    raise FileNotFoundError("Could not locate MEAD folder under RAW_DATA_ROOT.")

def list_audio_under(path: Path, exts=(".wav", ".flac", ".mp3")) -> List[Path]:
    out = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out

def select_mead_longest_refs(_: pd.DataFrame) -> List[Dict]:
    """
    Select MEAD references directly from folder tree (robust, avoids CSV issues).
    Picks longest file per (speaker, emotion, level) for configured speakers/emotions/levels.
    """
    mead_root = find_mead_root(RAW_DATA_ROOT)

    wanted_speakers = set(FAIRNESS_MEAD_SPEAKERS["male"] + FAIRNESS_MEAD_SPEAKERS["female"])
    emotions = [e.lower() for e in FAIRNESS_MEAD_EMOTIONS]
    levels = [l.lower() for l in FAIRNESS_MEAD_LEVELS]

    selected: List[Dict] = []

    # Try common MEAD layout first:
    #   <mead_root>/<spk>/audio/<emotion>/<level>/*
    for spk in sorted(wanted_speakers):
        for emo in emotions:
            for lvl in levels:
                cand_dir = mead_root / spk / "audio" / emo / lvl
                if not cand_dir.exists():
                    continue

                files = list_audio_under(cand_dir, exts=(".wav", ".flac"))
                if not files:
                    continue

                best_path = max(files, key=wav_duration_sec)
                selected.append({
                    "dataset": "MEAD",
                    "speaker_id": spk,
                    "emotion": emo,
                    "level": lvl,
                    # store relative-to-RAW path string so resolve_audio_path works
                    "wav_fp": str(best_path.relative_to(RAW_DATA_ROOT)).replace("\\", "/"),
                    "duration_sec": wav_duration_sec(best_path),
                    "speaker_info": None,
                })

    # Robust fallback: path-token matching across entire MEAD root
    if len({r["speaker_id"] for r in selected}) < len(wanted_speakers):
        all_audio = list_audio_under(mead_root, exts=(".wav", ".flac"))
        bucket: Dict[Tuple[str, str, str], List[Path]] = {}

        for p in all_audio:
            s = str(p).replace("\\", "/").lower()
            for spk in wanted_speakers:
                if f"/{spk.lower()}/" not in s:
                    continue
                for emo in emotions:
                    if f"/{emo}/" not in s:
                        continue
                    for lvl in levels:
                        if f"/{lvl}/" not in s:
                            continue
                        bucket.setdefault((spk, emo, lvl), []).append(p)

        existing_keys = {(r["speaker_id"], r["emotion"], r["level"]) for r in selected}
        for key, paths in bucket.items():
            if key in existing_keys:
                continue
            best_path = max(paths, key=wav_duration_sec)
            spk, emo, lvl = key
            selected.append({
                "dataset": "MEAD",
                "speaker_id": spk,
                "emotion": emo,
                "level": lvl,
                "wav_fp": str(best_path.relative_to(RAW_DATA_ROOT)).replace("\\", "/"),
                "duration_sec": wav_duration_sec(best_path),
                "speaker_info": None,
            })

    if not selected:
        raise RuntimeError("No MEAD references selected from filesystem. Check RAW_DATA_ROOT MEAD layout.")

    spk_count = len({r["speaker_id"] for r in selected})
    print(f" MEAD selected refs: {len(selected)} across {spk_count} speakers")
    return selected

# ---------------------------- COPY REFERENCES (PORTABLE) ----------------------------

def copy_reference_into_output(ref_row: Dict, out_root: Path, refs_root: Path) -> Dict:
    """
    Copies reference into OUTPUT_ROOT/references/... and returns:
      - reference_audio_rel: relative to OUTPUT_ROOT (portable)
      - reference_audio_abs: absolute on disk (internal use)
    """
    dataset = ref_row.get("dataset", "UNK")
    speaker_id = ref_row.get("speaker_id", "spk")
    emotion = (ref_row.get("emotion") or "unknown").lower()
    level = (ref_row.get("level") or "na").lower()

    src_abs = resolve_audio_path(ref_row["wav_fp"])
    if not src_abs.exists():
        raise FileNotFoundError(f"Reference audio not found after resolving under RAW_DATA_ROOT: {src_abs}")

    if dataset.upper() == "VCTK":
        dst_dir = refs_root / "VCTK" / speaker_id
    elif dataset.upper() == "MEAD":
        dst_dir = refs_root / "MEAD" / speaker_id / emotion / level
    else:
        dst_dir = refs_root / dataset / speaker_id

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_abs = dst_dir / src_abs.name

    if not dst_abs.exists():
        shutil.copy2(src_abs, dst_abs)

    return {
        "reference_audio_rel": rel_to_out_root(dst_abs, out_root),
        "reference_audio_abs": str(dst_abs),
    }

# ---------------------------- PROMPTS ----------------------------

def load_cosafe_prompts() -> Tuple[List[str], Dict[str, List[Dict]]]:
    cat_files = sorted(COSAFE_DIR.glob("*.json"))
    if not cat_files:
        raise RuntimeError(f"No CoSafe *.json found under {COSAFE_DIR}")

    if USE_HALF_COSAFE_CATEGORIES:
        cat_files = cat_files[: max(1, len(cat_files) // 2)]

    categories: Dict[str, List[Dict]] = {}
    cat_names: List[str] = []

    for jf in cat_files:
        cat = jf.stem
        rows = safe_json_load_any(jf)

        items: List[Tuple[str, int]] = []
        for row in rows:
            if isinstance(row, list) and len(row) == 2:
                items.append((str(row[0]), int(row[1])))

        if not items:
            continue

        items = items[:COSAFE_PROMPTS_PER_CATEGORY]
        out = []
        for i, (txt, lab) in enumerate(items):
            out.append({
                "prompt_id": f"cosafe_{cat}_{i:04d}",
                "prompt": txt,
                "safety_label": int(lab),
                "harm": "harm",
                "prompt_source": "cosafe",
                "ground_truth": {"type": "safety_label", "value": int(lab)},
            })

        key = f"cosafe::{cat}"
        categories[key] = out
        cat_names.append(key)

    if not cat_names:
        raise RuntimeError("No CoSafe prompts loaded.")
    return cat_names, categories

def load_librisqa_benign_prompts(target_count: int, min_len: int, max_len: int) -> List[Dict]:
    """
    Robust LibriSQA loader: uses streaming=True to avoid schema casting errors.
    Normalizes possible question/answer field names across files.
    """
    from datasets import load_dataset

    def pick_first(ex: dict, keys: List[str]) -> Optional[str]:
        for k in keys:
            v = ex.get(k, None)
            if v is None:
                continue
            v = str(v).strip()
            if v:
                return v
        return None

    pool: List[Tuple[str, str]] = []

    for sp in LIBRISQA_SPLITS:
        ds = load_dataset(LIBRISQA_DATASET, split=sp, streaming=True)

        for ex in ds:
            q = pick_first(ex, ["question", "question_and_option", "question_and_options", "prompt", "text"])
            a = pick_first(ex, ["answer", "answer_and_analysis", "label", "target"])
            if not q or not a:
                continue
            if not (min_len <= len(q) <= max_len):
                continue

            pool.append((q, a))
            if len(pool) >= target_count * 10:
                break

        if len(pool) >= target_count * 10:
            break

    if not pool:
        raise RuntimeError("No LibriSQA prompts loaded (after filtering/normalization).")

    random.shuffle(pool)
    pool = pool[:target_count]

    out = []
    for i, (q, a) in enumerate(pool):
        out.append({
            "prompt_id": f"benign_librisqa_{i:06d}",
            "prompt": q,
            "safety_label": 0,
            "harm": "benign",
            "prompt_source": "librisqa",
            "ground_truth": {"type": "answer", "value": a},
        })
    return out

# ---------------------------- STYLETTS2 ----------------------------

def init_styletts2():
    from styletts2 import tts as styletts2
    return styletts2.StyleTTS2(model_checkpoint_path=STYLETTS2_CKPT, config_path=STYLETTS2_CFG)

def run_styletts2(style_model, text: str, ref_path_abs: str, out_wav: Path):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        style_model.inference(text=text, target_voice_path=ref_path_abs, output_wav_file=tmp_path)
        wav, sr = ta.load(tmp_path)
        wav = _resample_to(wav, sr, TARGET_SR)
        ta.save(str(out_wav), wav.cpu(), TARGET_SR)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------- PRINT SELECTED SPEAKERS/LABELS ----------------------------

def print_selected_reference_summary(vctk_refs: List[Dict], mead_refs: List[Dict]) -> None:
    # VCTK
    print("\n==================== SELECTED REFERENCES: VCTK ====================")
    if not vctk_refs:
        print(" (none)")
    else:
        for r in vctk_refs:
            spk = r.get("speaker_id")
            si = r.get("speaker_info") or {}
            dur = float(r.get("duration_sec") or 0.0)
            fp = str(r.get("wav_fp") or "")
            print(
                f" - {spk:>4} | age={si.get('AGE')} | gender={si.get('GENDER')} | accent={si.get('ACCENT')} "
                f"| region={si.get('REGION')} | dur={dur:.2f}s | wav_fp={fp}"
            )

    # MEAD
    print("\n==================== SELECTED REFERENCES: MEAD ====================")
    if not mead_refs:
        print(" (none)")
    else:
        # Print per-speaker aggregation + a few example keys
        from collections import defaultdict
        by_spk = defaultdict(list)
        for r in mead_refs:
            by_spk[r.get("speaker_id")].append(r)

        for spk in sorted(by_spk.keys()):
            sex = "male" if str(spk).upper().startswith("M") else ("female" if str(spk).upper().startswith("W") else None)
            rows = by_spk[spk]
            combos = sorted({(x.get("emotion"), x.get("level")) for x in rows})
            print(f" - {spk:>4} | sex={sex} | refs={len(rows)} | unique(emotion,level)={len(combos)}")
            # show up to 6 combos
            for (emo, lvl) in combos[:6]:
                print(f"      * emotion={emo} | level={lvl}")
            if len(combos) > 6:
                print(f"      ... (+{len(combos)-6} more)")

        # Also print the first N concrete selected files to verify paths
        N = min(10, len(mead_refs))
        print(f"\n MEAD example selected files (first {N}):")
        for r in mead_refs[:N]:
            dur = float(r.get("duration_sec") or 0.0)
            print(
                f"   - {r.get('speaker_id')} | emo={r.get('emotion')} | lvl={r.get('level')} "
                f"| dur={dur:.2f}s | wav_fp={r.get('wav_fp')}"
            )

    print("===================================================================\n")

# ---------------------------- MAIN ----------------------------

def main():
    out_root = ensure_unique_dir(BASE_OUTPUT_ROOT)
    manifest_dir = out_root / "manifests"
    audio_dir = out_root / "styletts2"
    refs_root = out_root / "references"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    refs_root.mkdir(parents=True, exist_ok=True)

    df = load_text_all(TEXT_ALL_CSV)
    vctk_info = parse_vctk_speaker_info()

    vctk_refs = select_vctk_longest_refs(df, vctk_info)
    mead_refs = select_mead_longest_refs(df)  # df unused but kept signature-compatible

    # ✅ PRINT selected speakers + labels BEFORE copying/generation
    print_selected_reference_summary(vctk_refs, mead_refs)

    # Copy selected references into output and build ref_pool using LOCAL refs
    ref_pool = []
    for r in vctk_refs + mead_refs:
        copied = copy_reference_into_output(r, out_root, refs_root)

        ref_pool.append({
            "dataset": r.get("dataset"),
            "speaker_id": r.get("speaker_id"),
            "emotion": r.get("emotion"),
            "level": r.get("level", None),
            "duration_sec": float(r.get("duration_sec") or 0.0),

            # portable reference path to store in manifest
            "reference_audio_rel": copied["reference_audio_rel"],

            # absolute path only for running TTS (do NOT store in manifest)
            "reference_audio_abs": copied["reference_audio_abs"],

            # speaker info (safe to store)
            "vctk_speaker_info": r.get("speaker_info"),
        })

    # Prompts
    cat_names, cosafe_cats = load_cosafe_prompts()
    all_cosafe_prompts = [x["prompt"] for c in cat_names for x in cosafe_cats[c]]
    lengths = [len(p) for p in all_cosafe_prompts]
    min_len, max_len = min(lengths), max(lengths)
    total_cosafe = sum(len(cosafe_cats[c]) for c in cat_names)

    benign_needed = total_cosafe if BENIGN_MATCH_COSAFE_COUNT else COSAFE_PROMPTS_PER_CATEGORY
    benign_prompts = load_librisqa_benign_prompts(benign_needed, min_len, max_len)

    categories = dict(cosafe_cats)
    categories[BENIGN_CATEGORY_NAME] = benign_prompts
    all_categories = list(categories.keys())

    print(f" RAW_DATA_ROOT: {RAW_DATA_ROOT}")
    print(f" Output root:   {out_root}")
    print(f" References:    {len(ref_pool)} (portable refs under {refs_root})")
    print(f" Categories:    {len(all_categories)} | CoSafe={len(cat_names)} | Benign=1")
    print(f" Prompts:       CoSafe={total_cosafe} | Benign={len(benign_prompts)}")

    print("\nLoading StyleTTS2...")
    style_model = init_styletts2()
    print("StyleTTS2 loaded.\n")

    for cat in all_categories:
        manifest_path = manifest_dir / f"{cat}.jsonl"
        prompts = categories[cat]

        with manifest_path.open("w", encoding="utf-8") as mf:
            pbar = tqdm(total=len(prompts) * len(ref_pool), desc=f"Generating [{cat}]", unit="gen")

            for p in prompts:
                prompt_text = p["prompt"]

                for ri, ref in enumerate(ref_pool):
                    ds = ref["dataset"]
                    spk = ref["speaker_id"]
                    emo = (ref.get("emotion") or "unknown").lower()
                    lvl = (ref.get("level") or "na").lower()

                    out_prefix = f"{p['prompt_id']}__ref_{ds}_{spk}_{emo}_{lvl}"
                    out_wav = audio_dir / cat / f"{out_prefix}.wav"

                    try:
                        run_styletts2(style_model, prompt_text, ref["reference_audio_abs"], out_wav)
                        dur = wav_duration_sec(out_wav)
                        ok, err = True, None
                    except Exception as e:
                        dur = None
                        ok, err = False, str(e)

                    # ✅ demographics for VCTK from speaker-info.txt
                    demographics = None
                    if ds == "VCTK":
                        si = ref.get("vctk_speaker_info") or {}
                        demographics = {
                            "age": si.get("AGE"),
                            "gender": si.get("GENDER"),
                            "accent": si.get("ACCENT"),
                            "region": si.get("REGION"),
                        }

                    rec = {
                        "id": f"{cat}__{p['prompt_id']}__ref_{ri:04d}",
                        "category": cat,
                        "prompt_id": p["prompt_id"],
                        "text_prompt": prompt_text,

                        "harm": p.get("harm"),
                        "prompt_source": p.get("prompt_source"),
                        "safety_label": p.get("safety_label"),
                        "ground_truth": p.get("ground_truth"),

                        # PORTABLE: only store relative reference path
                        "reference": {
                            "dataset": ds,
                            "speaker_id": spk,
                            "emotion": ref.get("emotion"),
                            "level": ref.get("level"),
                            "duration_sec": ref.get("duration_sec"),
                            "reference_audio": ref.get("reference_audio_rel"),
                            "demographics": demographics,
                        },

                        "tts_system": "styletts2",
                        "generated": {
                            "ok": ok,
                            "error": err,
                            "path": rel_to_out_root(out_wav, out_root) if ok else None,
                            "sample_rate": TARGET_SR if ok else None,
                            "duration_sec": dur,
                        },
                    }

                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pbar.update(1)

            pbar.close()

    print("\nDone.")
    print(f"Output root:         {out_root}")
    print(f"Portable references: {refs_root}")
    print(f"Audio:               {audio_dir}")
    print(f"Manifests:           {manifest_dir}")

if __name__ == "__main__":
    main()