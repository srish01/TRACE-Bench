from collections import defaultdict
from pathlib import Path
import json

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

def rewrite_id(args, old_id: str) -> str:
    parts = old_id.split("__")

    # Example:
    # parts[0] = "cosafe::discrimination,stereotype,injustice_select_100"
    # parts[1] = "cosafe_discrimination,stereotype,injustice_select_100_0000"
    # parts[2] = "ref_0000"
    if args.setting == "security":
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else None
        
        prompt_id = prefix.split("_")[-1]
        new_id = old_id
    else:
        prefix = parts[0]  
        middle = parts[1]
        suffix = parts[2]

        # Extract the numeric index (0000)
        prompt_id = middle.split("_")[-1]
        new_id = f"{prefix}_{prompt_id}__{suffix}"
    
    return new_id, prompt_id

def infer_gender(data_id, speaker_id):
    # if data_id == "VCTK":
    #     demo = entry["reference"].get("demographics", {})
    #     return demo.get("gender")
    if data_id == "VCTK":
        # Security may not have entry["reference"], so use fixed mapping
        vctk_speaker_gender = {
            "p294": "F",
            "p227": "M",
            "p364": "M",
            "p374": "M",
            "p248": "F",
            "p251": "M"
        }
        return vctk_speaker_gender.get(speaker_id)


    elif data_id == "MEAD":
        if speaker_id and speaker_id.startswith("W"):
            return "F"
        elif speaker_id and speaker_id.startswith("M"):
            return "M"
        else:
            return None

    elif data_id == "FaiST-EV":
        if speaker_id == "african_212_speaker_00":
            return "M"
        elif speaker_id == "african_225_speaker_00":
            return "F"
        else:
            return None

    return None

def infer_main_char(data_id):
    if data_id == "MEAD":
        return "emotion"
    else:
        return "accent"
    
def infer_demographic(data_id, entry):
    ref = entry.get("reference", {}) or {}
    ref = entry["reference"]

    if data_id == "MEAD":
        emotion = ref.get("emotion")
        level = ref.get("level")
        if emotion and level:
            return f"{emotion}_{level}"
        elif emotion:
            return emotion
        else:
            return None

    elif data_id == "VCTK":
        demo = ref.get("demographics", {}) or {}
        accent = demo.get("accent")
        if accent:
            return f"accent_{accent}"
        return None

    elif data_id == "FaiST-EV":
        return "accent_african"

    return None

def infer_demographic_sec(data_id, new_id):
    
    vctk_speaker_accent = {
        "p294": "accent_American",
        "p227": "accent_English",
        "p364": "accent_Irish",
        "p374": "accent_Australian",
        "p248": "accent_Indian",
        "251": "accent_Indian"
    }

    if data_id == "MEAD":
        suffix = new_id.split("__ref_MEAD_")[1]
        parts = suffix.split("_")
        variant_id = "_".join(parts[1:])
        return variant_id if variant_id else None
    
    elif data_id == "VCTK":
        suffix = new_id.split("__ref_VCTK_")[1]
        speaker_id = suffix.split("_")[0]
        variant_id = vctk_speaker_accent.get(speaker_id, None)
        return variant_id
    
    elif data_id == "FaiST-EV":
        return "accent_african"
    
    return None

def infer_harm_from_category(category: str) -> str:
    if category == "benign_librisqa":
        return "benign"
    if category.startswith("cosafe::"):
        return "harm"
    return None  # fallback if needed


def split_security_manifest_to_category(
    adv_manifest_path: Path,
    out_manifest_dir: Path,
):
    out_manifest_dir.mkdir(parents=True, exist_ok=True)

    buckets = defaultdict(list)

    with open(adv_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            adv_path = Path(entry["adv_path"])
            category = adv_path.parent.name
            harm = infer_harm_from_category(category)

            fairness_like_entry = {
                "id": adv_path.stem,
                "category": category,
                "harm": harm,               # optional
                "text_prompt": None,        # security = audio-only
                "generated": {
                    "path": str(adv_path),
                },
                "reference": {},            # no speaker metadata
                "attack": entry["attack"],
                "epsilon": entry["epsilon"],
                "alpha": entry["alpha"],
                "iters": entry["iters"]
            }

            buckets[category].append(fairness_like_entry)

    # Write one manifest per category
    for category, entries in buckets.items():
        out_path = out_manifest_dir / f"{category}.jsonl"
        with open(out_path, "w", encoding="utf-8") as out_f:
            for e in entries:
                out_f.write(json.dumps(e, ensure_ascii=False) + "\n")


def load_prompt_lookup(fairness_results_path: Path):
    prompt_map = {}

    with open(fairness_results_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec.get("prompt_id")
            prompt = rec.get("prompt")

            if pid is not None and prompt is not None:
                prompt_map[pid] = prompt

    return prompt_map


def extract_data_speaker_dec_data(new_id: str):
    """
    Extract data_id and speaker_id from new_id.
    
    Examples:
    - "cosafe_..._0031__ref_MEAD_M011_sad_level_2" -> MEAD, M011
    - "cosafe_..._0048__ref_VCTK_p251_neutral_na" -> VCTK, p251
    - "cosafe_..._0029__ref_FaiST-EV_african_225_speaker_00_unknown_na" -> FaiST-EV, african_225_speaker_00
    """

    # Detect data_id
    for d_id in ["MEAD", "VCTK", "FaiST-EV"]:
        if f"__ref_{d_id}_" in new_id:
            data_id = d_id
            break
    else:
        # fallback
        return None, None

    # Extract speaker_id
    suffix = new_id.split(f"__ref_{data_id}_")[1]

    if data_id in ["MEAD", "VCTK"]:
        # speaker_id is the first segment before next "_"
        speaker_id = suffix.split("_")[0]
    elif data_id == "FaiST-EV":
        # There are only two possible speaker_ids; pick the matching one
        for spk in ["african_212_speaker_00", "african_225_speaker_00"]:
            if spk in suffix:
                speaker_id = spk
                break
        else:
            speaker_id = None
    else:
        speaker_id = None

    return data_id, speaker_id

import re
import json

def extract_json(response: str):
    # Remove markdown fences if present
    response = response.strip()
    response = re.sub(r"```json|```", "", response)

    # Extract first JSON object
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")

    json_str = match.group(0)

    # Fix common issues
    json_str = json_str.replace("'", '"')  # single → double quotes

    return json.loads(json_str)