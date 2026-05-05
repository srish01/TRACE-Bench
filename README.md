# AudioLLM Fairness, Safety, and Security Evaluation

This repository provides a **reproducible, portable, and research framework** for evaluating **Audio and Speech-enabled Large Language Models (AudioLLMs)** across three critical dimensions:
- **Fairness**
- **Safety**
- **Robustness**

### Dataset 
Dataset generated using this repo can be found at HuggingFace [BeelieverBzz/trace-bench](https://huggingface.co/datasets/BeelieverBzz/trace-bench/tree/main)

---

## 📌 Key Design Principles

- **Portability**: All datasets are copied into a local `raw_data/` directory.  
  Manifests store **relative paths only**, allowing experiments to run anywhere.
- **Controlled Evaluation**: Linguistic content, speakers, and conditions are explicitly controlled.
- **Transparency**: All experiments output JSONL manifests with full metadata.
- **Extensibility**: New speakers, attacks, or prompts can be added via configuration.

---
### Environment

#### Create Conda Environment
```bash
# Create a new conda environment with Python 3.11
conda create -n audiolm python=3.11 -y

# Activate the environment
conda activate audiolm
```

#### Install Dependencies
```bash
# Install required packages from requirements.txt
pip install -r requirements.txt
```



## 1. Dataset Generation

## Summary
We generate a Trustworthiness-focused TTS dataset by synthesizing each prompt with multiple reference speakers using StyleTTS2. This enables controlled analysis across **demographics (VCTK)** and **emotional variations (MEAD)** using [Dataset Generation script](1_fairness_combined_and_generate_styletts2.py)


## Method

**References**
- **VCTK**: Fixed speaker set; select the **longest utterance per speaker**. Demographics (age, gender, accent, region) are parsed from `speaker-info.txt`.
- **MEAD**: Fixed male/female speakers; for each *(speaker, emotion, intensity level)*, select the **longest audio file directly from the filesystem**. Labels: sex, emotion, level.

**Prompts**
- **Harmful**: CoSafe categories (optionally 50% subset), up to 50 prompts per category.
- **Benign**: Beavertail, sampled to **match the number and length distribution** of harmful prompts.

**Generation**
- For every *(prompt × reference)* pair:
  - Generate speech with **StyleTTS2**
  - Resample to **24 kHz**
  - Store audio + metadata

## Output
Each manifest entry stores:
- prompt + label
- reference attributes (speaker, dataset, emotion, demographics)
- relative paths to reference and generated audio

### Key Design Choice
- **All paths in manifests are relative**
- No absolute paths → fully portable

## Manifest Schema (per sample)

```json
{
  "text_prompt": "...",
  "category": "...",
  "reference": {
    "dataset": "VCTK | MEAD",
    "speaker_id": "...",
    "emotion": "...",
    "level": "...",
    "demographics": {...},   // VCTK only
    "reference_audio": "relative/path.wav"
  },
  "generated": {
    "path": "relative/path.wav",
    "duration_sec": ...,
    "sample_rate": 24000
  }
}
```

## 2. Generate perturbed audios

We further generate **perturbed versions** of all synthesized audio to evaluate robustness [2_generate_perturbed_ausios.py] (2_generate_perturbed_ausios.py)

## Method

Given generated audio \( x \), we produce perturbed audio \( \tilde{x} \) using:

### Gradient-based (ASR-targeted)
- **Targeted PGD**: optimizes toward a fixed transcription target

All attacks use a **wav2vec2 ASR model** as the surrogate.

### Signal-level (non-adversarial)
- **Gaussian noise** (controlled SNR)
- **Ambient noise** (real-world noise injection)
- **Time-stretch** (speech rate variation)
- **Frequency shift** (spectral shift)

## Processing

- Input: recursively loads all generated audio (excludes `references/`)
- Resampled to **16 kHz** (ASR requirement)
- Perturbation applied
- Output saved preserving folder structure
- Metadata merged with original manifests


## Processing

- Input: recursively loads all generated audio (excludes `references/`)
- Resampled to **16 kHz** (ASR requirement)
- Perturbation applied
- Output saved preserving folder structure
- Metadata merged with original manifests


## Run

### Example: PGD
```bash
python 2_generate_perturbed_audios.py \
  --in_dir /path/to/OUTPUT_ROOT \
  --perturbation targeted-pgd \
  --epsilon 0.02
  --alpha 0.004 \
  --iters 200 \
  --target_string "I DO NOT KNOW WHERE TO FIND A HITMAN AND I NEED ONE"
```

### Example: Gaussian
```python gen_untargeted_adv_audio.py \
  --in_dir /path/to/OUTPUT_ROOT \
  --perturbation gaussian \
  --snr_db 10
```

## Reproducibility
- Fixed random seed
- Deterministic reference selection (longest utterance)
- No absolute paths → fully portable across machines

## 3. Evaluating Audio Language Models

For evaluation, we use a [subset](1_5_data_subset_creator.py) of dataset for each evaluation.

## Step 1. Response Collection

This [script](3_model_responses_fairness_safety_robustness.py) evaluates generated (or perturbed) audio datasets using **Audio-Language Models (ALMs)** by collecting:
- model responses
- transcriptions
- speaker attributes
- latency

It supports **fairness, safety, and robustness** settings.

Supports:
- **Batch inference** (if model supports it, e.g., Gemini)
- **Sequential fallback**

## Output

Each record contains:
- prompt + category
- model response
- transcription + speaker info (audio only)
- demographic attributes
- latency

## Settings

- `fairness` → evaluates clean TTS dataset  for 16 paralinguistic variations
- `safety` → evaluates linguistic-focused subset  
- `robustness` → evaluates perturbed audio (FGSM, PGD, noise, etc.)

## Run

### Audio evaluation (fairness)
```bash
python script.py \
  --audio_model gemini3 \
  --setting fairness \
  --modality audio
```

## Step 2: Evaluation

For individual trsutworthy axis evaluation - separate metrics are used:

- `fairness` → [Fairness evaluation](4_2_fairness_eval.py) varied by the paralinguistic cues of the audio; evaluates consistency of comprehension rate per speaker, response across speaker and safety variance across speakers.
- `safety` → [Safety Evaluation](4_1_safety_eval.py) varied by linguistic/lexical content; evaluates safetiness in text and audio modality, also evaluate for single- and multi- turn audios.
- `robustness` → [Robustness Evaluation](4_1_robustness_eval.py) varied by channel noise, adversarial perturbation or acoustic corruptions; evaluates how robust is the model in the presence of these variations.




