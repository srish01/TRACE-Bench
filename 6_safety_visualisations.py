import json
import csv
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path("Safety_evaluations")

OUT_AUDIO_CSV = "safety_audio_analysis.csv"
OUT_CROSS_CSV = "safety_cross_modal_analysis.csv"
OUT_AVG_MODEL_CSV = "safety_per_model_analysis.csv"


# -----------------------------
# Utilities
# -----------------------------
def extract_category(key: str) -> str:
    return (
        key.replace("cosafe::", "")
           .replace("_select_100", "")
    )


def r2(x):
    return round(x, 2)


def signed(x):
    return f"{x:+.2f}"


# -----------------------------
# Load all model JSONs
# -----------------------------
def load_models(base_dir: Path):
    """
    Loads:
      Safety_evaluations/{model}/{model}_global_stats.json
    """
    models = {}
    categories = set()

    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        json_fp = model_dir / f"{model_name}_global_stats.json"

        if not json_fp.exists():
            raise FileNotFoundError(json_fp)

        with open(json_fp, "r") as f:
            data = json.load(f)

        models[model_name] = data["categories"]

        for key in data["categories"]:
            categories.add(extract_category(key))

    if not models:
        raise RuntimeError(f"No models found in {base_dir}")

    return models, sorted(categories)


# =====================================================
# 1) AUDIO SAFETY CSV
# =====================================================
def export_audio_safety_csv(models, categories, out_path):
    model_names = sorted(models.keys())

    metrics = ["C-RR", "C-SR", "C-ASR", "Mean Toxicity"]

    header = ["category"]
    for model in model_names:
        for m in metrics:
            header.append(f"{model}::{m}")

    rows = []

    for cat in categories:
        row = {"category": cat}

        for model in model_names:
            key = f"cosafe::{cat}_select_100"
            stats = models[model][key]["safety_stats"]

            row[f"{model}::C-RR"] = round(stats["c-rr"] * 100, 2)
            row[f"{model}::C-SR"] = round((stats["c-srr"] - stats["c-rr"]) * 100, 2)
            row[f"{model}::C-ASR"] = round(stats["c-asr"] * 100, 2)
            row[f"{model}::Mean Toxicity"] = round(
                stats["mean_unsafe_audio_tox"] * 100, 2
            )

        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)



# =====================================================
# 2) CROSS-MODAL CSV
# =====================================================
def export_cross_modal_csv(models, categories, out_path):
    model_names = sorted(models.keys())

    header = (
        ["category"]
        + [f"{m}::CMSI" for m in model_names]
        + [f"{m}::CMTS" for m in model_names]
    )

    rows = []

    for cat in categories:
        row = {"category": cat}

        for model in model_names:
            key = f"cosafe::{cat}_select_100"
            entry = models[model][key]

            cmsi = 1.0 - entry["cross_modal_stats"]["cmsc"]
            delta = entry["safety_stats"]["mean_delta_tox"]

            row[f"{model}::CMSI"] = r2(cmsi)
            row[f"{model}::CMTS"] = signed(r2(delta))

        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

def export_cross_modal_csv(models, categories, out_path):
    model_names = sorted(models.keys())

    metrics = ["CMSI", "CMTS"]

    header = ["category"]
    for model in model_names:
        for m in metrics:
            header.append(f"{model}::{m}")

    rows = []

    for cat in categories:
        row = {"category": cat}

        for model in model_names:
            key = f"cosafe::{cat}_select_100"
            entry = models[model][key]

            cmsi = 1.0 - entry["cross_modal_stats"]["cmsc"]
            delta = entry["safety_stats"]["mean_delta_tox"]

            row[f"{model}::CMSI"] = r2(cmsi)
            row[f"{model}::CMTS"] = signed(r2(delta))
        
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

def export_model_averages_csv(models, categories, out_path):
    import csv
    from statistics import mean

    header = [
        "model",
        "C-RR",
        "C-SR",
        "C-ASR",
        "Mean Toxicity",
        "CMSI",
        "CMTS",
    ]

    rows = []

    for model, data in models.items():
        c_rr_vals = []
        c_sr_vals = []
        c_asr_vals = []
        tox_vals = []
        cmsi_vals = []
        cmts_vals = []

        for cat in categories:
            key = f"cosafe::{cat}_select_100"
            entry = data[key]

            s = entry["safety_stats"]
            c = entry["cross_modal_stats"]

            c_rr_vals.append(s["c-rr"] * 100)
            c_sr_vals.append((s["c-srr"] - s["c-rr"]) * 100)
            c_asr_vals.append(s["c-asr"] * 100)
            tox_vals.append(s["mean_unsafe_audio_tox"] * 100)

            cmsi_vals.append(1.0 - c["cmsc"])
            cmts_vals.append(s["mean_delta_tox"])

        row = {
            "model": model,
            "C-RR": round(mean(c_rr_vals), 2),
            "C-SR": round(mean(c_sr_vals), 2),
            "C-ASR": round(mean(c_asr_vals), 2),
            "Mean Toxicity": round(mean(tox_vals), 2),
            "CMSI": round(mean(cmsi_vals), 2),
            "CMTS": f"{mean(cmts_vals):+.2f}",
        }

        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)



# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    models, categories = load_models(BASE_DIR)

    export_audio_safety_csv(models, categories, OUT_AUDIO_CSV)
    export_cross_modal_csv(models, categories, OUT_CROSS_CSV)
    export_model_averages_csv(models, categories, OUT_AVG_MODEL_CSV)

    print(f"Saved: {OUT_AUDIO_CSV}")
    print(f"Saved: {OUT_CROSS_CSV}")
