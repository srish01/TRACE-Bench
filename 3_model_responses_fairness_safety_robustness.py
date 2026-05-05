from __future__ import annotations
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

import time
import torch
import librosa
import argparse
import sys
from utils.utils import *
from utils.models import *
# from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()



# =========================
# DATA LOADING
# =========================

def load_manifest(manifest_path: Path) -> List[Dict]:
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


# =========================
# MAIN EVALUATION LOOP
# =========================


def evaluate_manifest_audio_input(args, manifest_path: Path, output_dir_audio: Path, audio_root: Path, model: BaseAudioModel, prompt_lookup: dict = None):
    print("------ Evaluating audio prompts ------")
    category_name = manifest_path.stem
    print(f"\n[Collect] Category: {category_name}")
    results_path = output_dir_audio / f"{category_name}_results_{model.name}.jsonl"      
    entries = load_manifest(manifest_path)
    print(f"  - {len(entries)} records after loading manifest.")

    if results_path.exists():
        print(f"{results_path} exists. Skipping...")
        return
    
    # Check if model supports batch processing (Gemini, GPT-4o, etc.)
    supports_batch = hasattr(model, 'process_batch_simple')
    
    if supports_batch and args.batch_size > 1:
        print(f"[INFO] Using batch processing with batch_size={args.batch_size}, max_workers={args.max_workers}")
        evaluate_manifest_audio_input_batch(args, manifest_path, output_dir_audio, audio_root, model, prompt_lookup, results_path, entries)
    else:
        if supports_batch:
            print(f"[INFO] Batch processing available but batch_size={args.batch_size}, using sequential mode")
        else:
            print(f"[INFO] Model does not support batch processing, using sequential mode")
        evaluate_manifest_audio_input_sequential(args, manifest_path, output_dir_audio, audio_root, model, prompt_lookup, results_path, entries)


def evaluate_manifest_audio_input_batch(args, manifest_path: Path, output_dir_audio: Path, audio_root: Path, 
                                       model: BaseAudioModel, prompt_lookup: dict, results_path: Path, entries: list):
    """
    Batch processing version for models that support concurrent API calls (e.g., Gemini).
    """
    # Filter valid entries
    valid_entries = []
    valid_audio_paths = []
    valid_text_prompts = []
    
    for i, entry in enumerate(entries):
        audio_rel_path = entry["generated"]["path"]
        if not audio_rel_path:
            continue
            
        # if args.setting == "robustness":
        #     audio_path = Path(audio_rel_path)
        # else:
        audio_path = audio_root / audio_rel_path

        if not audio_path.exists():
            print(f"[WARN] Missing audio: {audio_path}")
            continue
        
        valid_entries.append((i, entry))
        valid_audio_paths.append(str(audio_path))
        valid_text_prompts.append(entry.get("text_prompt"))
    
    print(f"[INFO] Processing {len(valid_entries)} entries in batches of {args.batch_size}")
    
    with open(results_path, "w", encoding="utf-8") as out_f:
        # Process in batches
        for batch_start in range(0, len(valid_entries), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(valid_entries))
            batch_entries = valid_entries[batch_start:batch_end]
            batch_audio_paths = valid_audio_paths[batch_start:batch_end]
            batch_text_prompts = valid_text_prompts[batch_start:batch_end]
            
            print(f"[BATCH] Processing entries {batch_start} to {batch_end-1}")
            t_batch = time.time()
            
            # Batch generate responses
            try:
                batch_responses = model.process_batch_simple(
                    batch_audio_paths, 
                    batch_text_prompts,
                    max_workers=args.max_workers
                )
            except Exception as e:
                print(f"[ERROR] Batch processing failed: {e}, falling back to sequential")
                batch_responses = []
                for ap, tp in zip(batch_audio_paths, batch_text_prompts):
                    try:
                        batch_responses.append(model.generate(audio_path=ap, text_prompt=tp))
                    except Exception as e2:
                        batch_responses.append(f"ERROR: {str(e2)}")
            
            batch_time = time.time() - t_batch
            print(f"[BATCH] Completed in {batch_time:.2f}s ({batch_time/len(batch_entries):.2f}s per entry)")
            
            # Process each entry in the batch
            for (i, entry), audio_path, resp_a in zip(batch_entries, batch_audio_paths, batch_responses):
                try:
                    t0 = time.time()
                    new_id, prompt_idx = rewrite_id(args, entry.get("id"))
                    
                    # SPEAKER ANALYSIS (sequential, usually fast)
                    speaker_info = None
                    try:
                        speaker_info = model.infer_speaker(audio_path)
                    except Exception as e:
                        print(f"[WARN] speaker analysis failed for entry {i}: {e}")

                    # TRANSCRIPTION (sequential, usually fast)
                    transcription = None
                    try:
                        transcription = model.transcribe(audio_path)
                    except Exception as e:
                        print(f"[WARN] transcription failed for entry {i}: {e}")
                    
                    audio_latency = time.time() - t0
                    
                    # if args.setting == "security":
                    #     data_id, speaker_id = extract_data_speaker_dec_data(new_id)
                    #     variant_id = infer_demographic_sec(data_id, new_id=new_id)
                    #     if prompt_lookup is not None:
                    #         prompt_text = prompt_lookup.get(prompt_idx, entry.get("text_prompt"))
                    #     else:
                    #         prompt_text = "null"
                    # else:
                    #SG 29/04: we dont need prompt lookup for security data -- commented lines above
                    prompt_text = entry.get("text_prompt")
                    data_id = entry.get("reference", {}).get("dataset")
                    ref_speaker_id = (entry.get("reference", {}) or {}).get("speaker_id")
                    variant_id = infer_demographic(data_id, entry)
                        
                    ref_gender = infer_gender(data_id, ref_speaker_id)
                    main_characteristic = infer_main_char(data_id)

                    record = {
                        "modality": "audio",
                        "safety_category": entry.get("harm"),
                        "model": model.name,
                        "id": new_id,
                        "category": entry.get("category"),
                        "data_id": data_id,
                        "ref_speaker_id": ref_speaker_id,
                        "prompt_id": prompt_idx,
                        "gender": ref_gender,
                        "variant_type": main_characteristic,
                        "variant_id": variant_id,
                        "audio_path": audio_path,
                        "prompt": prompt_text,
                        "pred_transcription": transcription,
                        "pred_speaker_info": speaker_info,
                        "model_response": resp_a,
                        "reasoning_latency": float(audio_latency)
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                except Exception as e:
                    out_f.write(json.dumps({
                        "id": entry.get("id"),
                        "model": model.name,
                        "error": str(e)
                    }, ensure_ascii=False) + "\n")
                    out_f.flush()


def evaluate_manifest_audio_input_sequential(args, manifest_path: Path, output_dir_audio: Path, audio_root: Path, 
                                             model: BaseAudioModel, prompt_lookup: dict, results_path: Path, entries: list):
    """
    Sequential processing version (original implementation).
    """
    with open(results_path, "w", encoding="utf-8") as out_f:
        for i, entry in enumerate(entries):
                # if i>3:
                #     break
                if entry["is_multiturn"] == True:
                    audio_rel_path = entry["generated"]["turns"][4]["path"]
                    prompt_text = entry["conversation"][4]["text"]
                else:
                    prompt_text = entry.get("text_prompt")
                    if args.setting == "robustness":
                        audio_rel_path = entry.get("adv_audio_filepath", entry["generated"]["path"])
                    else:
                        audio_rel_path = entry["generated"]["path"]
                if not audio_rel_path:
                    continue
                if args.setting == "robustness":
                    audio_path = Path(audio_rel_path)
                else:
                    audio_path = audio_root / audio_rel_path

                if not audio_path.exists():
                    print(f"[WARN] Missing audio: {audio_path}")
                    continue

                new_id, prompt_idx = rewrite_id(args, entry.get("id"))
                print(f"=> generating response for entry {i}")
                
                transcription = None
                try:
                    t0 = time.time()
                    # SPEAKER ANALYSIS
                    # -------------------------
                    try:
                        speaker_info = model.infer_speaker(str(audio_path))
                    except Exception as e:
                        print(f"[WARN] speaker analysis failed: {e}")

                    # TRANSCRIPTION
                    # -------------------------
                    try:
                        transcription = model.transcribe(str(audio_path))
                    except Exception as e:
                        print(f"[WARN] transcription failed: {e}")
                    
                    # MODEL RESPONSE
                    # -------------------------
                    if entry["is_multiturn"] == True:
                        resp_a = model.generate_multiturn(
                            audio_root=audio_root,
                            chat=entry["generated"]["turns"]
                        )
                    else:
                        resp_a = model.generate(
                            audio_path=str(audio_path),
                            text_prompt=entry.get("text_prompt")
                        )
                    # emb_a = model.compute_semantic_embedding(resp_a)
                    audio_latency = time.time() - t0

                    # if args.setting == "security":
                    #     data_id, speaker_id = extract_data_speaker_dec_data(new_id)
                    #     variant_id = infer_demographic_sec(data_id, new_id=new_id)
                    #     if prompt_lookup is not None:
                    #         prompt_text = prompt_lookup.get(prompt_idx, entry.get("text_prompt"))
                    #     else:
                    #         prompt_text = "null"
                    # else:
                    #SG 29/04: we dont need prompt lookup for security data -- commented lines above
                    
                    
                    data_id = entry.get("reference", {}).get("dataset")
                    ref_speaker_id = (entry.get("reference", {}) or {}).get("speaker_id")
                    variant_id = infer_demographic(data_id, entry)      # SG: change this for new data format
                        
                    ref_gender = infer_gender(data_id, ref_speaker_id)       # SG: change this for new data format
                    main_characteristic = infer_main_char(data_id)           # SG: change this for new data format

                    record = {
                        "modality": "audio",
                        "safety_category": entry.get("harm"),
                        "model": model.name,
                        "id": new_id,
                        "category": entry.get("category"),
                        "data_id": data_id,
                        "ref_speaker_id": ref_speaker_id,
                        "prompt_id": prompt_idx,
                        "gender": ref_gender,
                        "variant_type": main_characteristic,
                        "variant_id": variant_id,
                        "audio_path": str(audio_path),
                        "prompt": prompt_text,
                        # "response_char_length": len(resp_a),
                        "pred_transcription": transcription,
                        "pred_speaker_info": speaker_info,
                        "model_response": resp_a,
                        "reasoning_latency": float(audio_latency)
                        # "embeddings": emb_a.tolist()
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                except Exception as e:
                    out_f.write(json.dumps({
                        "id": new_id,
                        "model": model.name,
                        "error": str(e)
                    }, ensure_ascii=False) + "\n")
                    out_f.flush()


def evaluate_manifest_text_input(args, manifest_path: Path, output_dir_text: Path, model: BaseAudioModel):
    print("------ Evaluating text prompts ------")
    category_name = manifest_path.stem
    print(f"\n[Collect TEXT] Category: {category_name}")

    results_path = output_dir_text / f"{category_name}_results_{model.name}.jsonl"

    entries = load_manifest(manifest_path)
    print(f"  - {len(entries)} total records loaded.")

    # Deduplicate by prompt_id
    seen = {}
    for entry in entries:
        pid = entry.get("prompt_id")
        if pid not in seen:
            seen[pid] = entry

    unique_entries = list(seen.values())
    print(f"  - {len(unique_entries)} unique prompts after deduplication.")

    with open(results_path, "w", encoding="utf-8") as out_f:
        for i, entry in enumerate(unique_entries):
            # if i>5:
            #     break
            new_id, prompt_idx = rewrite_id(args, entry.get("id"))
            print(f"=> generating response for entry {i}")
            
            try:
                t0 = time.time()
                resp_t = model.generate(
                    audio_path=None,
                    text_prompt=entry.get("text_prompt", "")
                )
                text_latency = time.time() - t0
                # emb_text = model.compute_semantic_embedding(resp_t)

                data_id = (entry.get("reference", {}) or {}).get("dataset")
                speaker_id = (entry.get("reference", {}) or {}).get("speaker_id")
                # not relevant for text input
                # gender = infer_gender(data_id, speaker_id, entry)
                # main_characteristic = infer_main_char(data_id)
                # variant_id = infer_demographic(data_id, entry)
                    
                record = {
                    "modality": "text",
                    "safety_category": entry.get("harm"),
                    "model": model.name,
                    "id": new_id,
                    "category": entry.get("category"),
                    "data_id": data_id,
                    "speaker_id": speaker_id,
                    "prompt_id": prompt_idx,
                    "gender": None,               # not relevant for text input
                    "variant_type": None,    # not relevant for text input
                    "variant_id": None,      # not relevant for text input
                    "audio_path": None,
                    "prompt": entry.get("text_prompt"),
                    "model_response": resp_t,
                    "response_char_length": len(resp_t),
                    "latency": float(text_latency),
                    # "embeddings": emb_text.tolist(),
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e:
                out_f.write(json.dumps({
                    "id": entry.get("id"),
                    "model": model.name,
                    "error": str(e)
                }, ensure_ascii=False) + "\n")
                out_f.flush()

# =========================
# ENTRY POINT
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_model", type=str, help="select the audio model for FSS evaluation", choices=["qwen2", "hugginggpt", "covoaudio", "audiogpt", "moshi", "gemini3", "phi4mm"])
    parser.add_argument("--setting", type=str, required=True, choices=["fairness", "safety", "robustness"])
    parser.add_argument("--modality", type=str, default="audio", required=False, choices=["audio", "text"])
    parser.add_argument("--perturbation", type=str, default="none", required=False, choices=["pgd", "fgsm", "gaussian", "ambient", "freq-shift", "time-stretch"])
    parser.add_argument("--robustness_variant", type=str, choices=["real", "stress"])
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for concurrent API calls (for models that support it, e.g., Gemini)")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of concurrent workers for batch processing")
    args = parser.parse_args()

    try:
        if args.audio_model == "qwen2":
            model = Qwen2AudioModel()
        elif args.audio_model == "hugginggpt":
            model = HuggingGPTAudioModel()
        elif args.audio_model == "spiritlm":
            model = SpiritLMAudioModel()
        elif args.audio_model == "audiogpt":
            model = AudioGPTAudioModel()
        elif args.audio_model == "moshi":
            model = MoshiAudioModel()
        elif args.audio_model == "gemini3":
            model = GeminiAudioModel(model_name="gemini-3-flash-preview")
        elif args.audio_model == "phi4mm":
            model = Phi4MultimodalAudioModel()
        else:
            raise ValueError(f"Unsupported model selected: {args.audio_model}")
    except Exception as e:
        print(f"[ERROR] Failed to load model '{args.audio_model}': {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.setting == "safety":
        root_dir = Path("Safety_TTS_subset_p227")
        output_dir = Path("Safety_outputs")
    elif args.setting == "fairness":
        root_dir = Path("Fairness_TTS_subset")
        output_dir = Path("Fairness_outputs")
    elif args.setting == "robustness":
        if args.perturbation not in ["pgd", "fgsm", "gaussian", "ambient", "freq-shift", "time-stretch"]:
            raise ValueError(f"Unsupported perturbation selected for robustness setting: {args.perturbation}")
        if args.robustness_variant not in ["real", "stress"]:
            raise ValueError(f"Unsupported robustness variant: {args.robustness_variant}")
        if args.robustness_variant == "real" and args.perturbation == "gaussian":
            variant_suffix = "5"
        elif args.robustness_variant == "real" and args.perturbation == "time-stretch":
            variant_suffix = "0.7"
        elif args.robustness_variant == "stress" and args.perturbation == "time-stretch":
            variant_suffix = "1.3"
        elif args.robustness_variant == "real" and args.perturbation == "freq-shift":
            variant_suffix = "-200"
        elif args.robustness_variant == "real":
            variant_suffix = "+5"
        else:
            variant_suffix = "-5"
        root_dir = Path(f"ROBUSTNESS_{args.perturbation.upper()}_{variant_suffix}")
        print("Input Directory:", root_dir)
        output_dir = Path(f"Robustness_outputs/{args.perturbation}_perturbations_{variant_suffix}")
    else:
        raise ValueError(f"Unsupported setting selected: {args.setting}")
    

    output_dir_audio = output_dir / "audio_responses_v2"    # SG: edit: for testing
    output_dir_audio.mkdir(exist_ok=True, parents=True)
    output_dir_text = output_dir / "text_responses_v2"      # SG: edit: for testing
    output_dir_text.mkdir(exist_ok=True, parents=True)

    manifest_dir = root_dir / "manifests"
    audio_root = root_dir

    # if args.setting == "security":
    #     if not manifest_dir.exists():
    #         split_security_manifest_to_category(adv_manifest_path = Path(f"Security_TTS_subset/{args.attack}_attacks/adv_manifest.jsonl"),
    #                                        out_manifest_dir = root_dir / "manifests")

    if not manifest_dir.exists():
        raise FileNotFoundError(f"manifests directory not found: {manifest_dir}")
        
    for manifest_file in manifest_dir.glob("*.jsonl"):
        print(f"Evaluating: {manifest_file.name}")
        prompt_lookup = None
        # if args.setting == "robustness":
        #     prompt_lookup = load_prompt_lookup(Path(f"Fairness_outputs/text_responses/{manifest_file.stem}_results_{model.name}.jsonl"))
        if args.modality == "audio":
            evaluate_manifest_audio_input(args, manifest_file, output_dir_audio, audio_root, model, prompt_lookup)
        elif args.modality == "text":
            evaluate_manifest_text_input(args, manifest_file, output_dir_text, model)       # comment for meralion

if __name__ == "__main__":
    main()