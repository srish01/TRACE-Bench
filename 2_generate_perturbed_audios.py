#!/usr/bin/env python3
# gen_untargeted_adv_audio.py
# ==========================================
# Generate adversarial audio (UNTARGETED) for
# FGSM or PGD using an in-file ASRAttacks class.
#
# - Loads WAV/FLAC from an input folder (recursive)
# - Resamples to 16 kHz (wav2vec2 expectation)
# - Generates adversarial waveform (FGSM/PGD untargeted)
# - Saves adversarial WAVs mirroring folder structure
# - Writes a JSONL manifest with clean/adv paths + params
#
# Requirements:
#   pip install torchaudio tqdm numpy
#
# Notes:
# - This script uses torchaudio's WAV2VEC2_ASR_BASE_960H bundle.
# - Attacks are Linf (epsilon) for FGSM/PGD.
# ==========================================
from __future__ import annotations
from scipy.signal import butter, lfilter
import librosa
from scipy.signal import hilbert


import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio as ta
import torchaudio.functional as TAF
from tqdm import tqdm
#from utils import wav_duration_sec

TARGET_SR = 16000
MIN_AUDIO_SAMPLES = 16000  # 1 second at 16kHz

# ----------------------------
# IO helpers
# ----------------------------
def list_audio_files(root: Path, exts=(".wav", ".flac", ".mp3", ".m4a", ".ogg")) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        # --- NEW: Tell it to ignore the references folder entirely ---
        if "references" in p.parts:
            continue
            
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files

def load_mono(path: Path) -> Tuple[torch.Tensor, int]:
    wav, sr = ta.load(str(path))
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.float(), sr


def resample_to(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav
    return TAF.resample(wav, src_sr, dst_sr)


def clamp_audio(wav: torch.Tensor) -> torch.Tensor:
    return torch.clamp(wav, -1.0, 1.0)

# --- ADDED GAUSSIAN MATH FUNCTIONS ---
def _lowpass_filter(signal, cutoff_hz, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    cutoff_hz = min(cutoff_hz, nyquist * 0.999)
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be > 0")
    norm_cutoff = cutoff_hz / nyquist
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

def add_gaussian_noise(audio, snr_db, sample_rate, noise_bandwidth=8000, eps=1e-10):
    audio = audio.astype(np.float32)
    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / (snr_linear + eps)
    noise = np.random.randn(*audio.shape).astype(np.float32)
    
    if noise_bandwidth is not None:
        if audio.ndim == 1:
            noise = _lowpass_filter(noise, noise_bandwidth, sample_rate)
        else:
            noise = np.stack([_lowpass_filter(ch, noise_bandwidth, sample_rate) for ch in noise], axis=0)
            
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + eps)
    noise = noise * np.sqrt(noise_power)
    noisy = audio + noise

    
    peak = np.max(np.abs(noisy)) + eps
    if peak > 1.0:
        noisy = noisy / peak
    return noisy
# ---------------------------------------

# --- AMBIENT NOISE FUNCTION ---

def add_ambient_noise(audio, snr_db, sample_rate, ambient_noise_path):
    """Add ambient noise from file to audio at specified decibel level"""
    
    # Load ambient noise
    
    noise, noise_sr = librosa.load(ambient_noise_path, sr=sample_rate)
    
    # Repeat noise if shorter than audio
    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)
    
    # Trim noise to match audio length
    noise = noise[:len(audio)]
    
    # Calculate audio and noise power
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate scaling factor for target SNR
    audio_db = 10 * np.log10(audio_power)
    target_noise_db = audio_db - snr_db
    target_noise_power = 10 ** (target_noise_db / 10)
    
    # Scale noise
    noise = noise * np.sqrt(target_noise_power / noise_power)
    
    # Add noise to audio
    return audio + noise

# --- Pitch Shift ---
# def pitch_shift_audio(audio, sample_rate, steps):
#     """Shift pitch of audio by specified semitone steps"""
#     return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=steps)


import numpy as np
import librosa

# --- Time Stretch ---

def time_stretch_audio(audio, rate):
    """
    Time-stretch audio without changing pitch.

    Args:
        audio (np.ndarray): shape (T,) mono audio
        sample_rate (int): sampling rate (Hz)
        rate (float): stretch factor
                      >1.0 → faster (shorter duration)
                      <1.0 → slower (longer duration)

    Returns:
        np.ndarray: time-stretched audio (float32)
    """

    # librosa expects float32
    audio = audio.astype(np.float32)

    # Apply time-stretch
    stretched = librosa.effects.time_stretch(audio, rate=rate)

    # Normalize to avoid clipping
    stretched = stretched / (np.max(np.abs(stretched)) + 1e-8)

    return stretched.astype(np.float32)


# --- Frequency Shift ---
def frequency_shift_audio(audio, sample_rate, shift_hz):
    """Shift all frequencies of audio additively by shift_hz (SSB modulation)"""
    from scipy.signal import hilbert
    analytic = hilbert(audio)
    t = np.arange(len(audio)) / float(sample_rate)
    shifted = np.real(analytic * np.exp(2j * np.pi * shift_hz * t))
    return shifted.astype(np.float32)


# ----------------------------
# ASR wrapper (torchaudio wav2vec2 ASR)
# ----------------------------
class Wav2Vec2ASRWrapper(torch.nn.Module):
    """
    Wrap torchaudio wav2vec2 bundle as a module compatible with ASRAttacks:
      emissions, lengths = model(wav)
    where wav is expected to be (1, time) or (time,).
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

        bundle = WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model().to(device).eval()
        self.labels = list(bundle.get_labels())
        self.device = device

    def forward(self, wav_16k: torch.Tensor):
        # Always feed (batch, time)
        if wav_16k.dim() == 1:
            wav_16k = wav_16k.unsqueeze(0)  # (1, time)
        elif wav_16k.dim() == 2:
            # if someone accidentally passed (channels, time), average channels
            if wav_16k.size(0) != 1:
                wav_16k = wav_16k.mean(dim=0, keepdim=True)
        return self.model(wav_16k.to(self.device))




# ----------------------------
# Attacks (UNTARGETED FGSM + PGD)
# ----------------------------
class ASRAttacks(object):
    """
    Adversarial attacks for wav2vec2 ASR (torchaudio bundle).
    Implemented here for the needs of this script:
      - FGSM_ATTACK (untargeted)
      - PGD_ATTACK  (untargeted)
    """

    def __init__(self, model: torch.nn.Module, device: str, labels: List[str]):
        self.model = model
        self.device = device
        self.labels = labels
        self.label2idx = {c: i for i, c in enumerate(labels)}
        # torchaudio wav2vec2 bundles use blank index 0 by default in this label set
        self.blank = 0

    def _normalize_transcription(self, t: Union[str, List[str]]) -> str:
        # Accept either "AB|CD" or ["A","B","|","C"] or ["my name"]-style inputs.
        if isinstance(t, list):
            t = "".join(t)
        # If spaces are present, convert to '|' (wav2vec2 word separator in this label set)
        t = t.replace(" ", "|")
        return t

    def _encode_transcription(self, transcription: Union[str, List[str]]) -> torch.Tensor:
        transcription = self._normalize_transcription(transcription)
        # Encode each character using label set
        try:
            encoded = [self.label2idx[ch] for ch in transcription]
        except KeyError as e:
            raise ValueError(
                f"Character {str(e)} not present in the model label set. "
                f"Check transcription/labels alignment."
            )
        return torch.tensor(encoded, dtype=torch.long)

    def INFER(self, input_: torch.Tensor) -> str:
        """
        Greedy decode wav2vec2 emissions to a label string (includes '|' word separators).
        input_: (time,) or (1,time)
        """
        if input_.dim() == 1:
            input_ = input_.unsqueeze(0)  # (1, time)
        elif input_.dim() != 2:
            raise ValueError(f"Expected input of shape (time,) or (1, time), got {input_.shape}")

        emissions, _ = self.model(input_.to(self.device))
        # emissions: (batch, time, vocab)
        tokens = torch.argmax(emissions, dim=-1)[0]  # (time,)
        tokens = torch.unique_consecutive(tokens)
        tokens = [t.item() for t in tokens if t.item() != self.blank]
        return "".join([self.labels[i] for i in tokens])

    def _wer(self, reference, prediction) -> Tuple[float, Tuple[int, int, int]]:
        correct = 0
        substitution = 0
        insertion = 0
        deletion = 0
        for i in range(len(reference)):
            if prediction[i] == reference[i]:
                correct += 1
            elif prediction[i] != reference[i] and prediction[i] != "<eps>" and reference[i] != "<eps>":
                substitution += 1
            elif prediction[i] == "<eps>":
                deletion += 1
            elif prediction[i] != reference[i] and reference[i] == "<eps>":
                insertion += 1
        wer = (substitution + insertion + deletion) / max((correct + substitution + deletion + insertion), 1)
        return wer, (substitution, insertion, deletion)





    def TARGETED_PGD_ATTACK(
        self,
        input__: torch.Tensor,
        target_string: str,
        epsilon: float = 0.04,
        alpha: float = 0.003,
        num_iter: int = 2000,
        check_every: int = 50,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        """
        Targeted PGD: minimize CTC loss against a fixed `target_string`.
        Returns (adv_audio_numpy, success_flag).
        """
        x = input__.clone().detach().to(self.device).float()
        x = torch.clamp(x, -1.0, 1.0)

        if x.numel() < MIN_AUDIO_SAMPLES:
            print(f"[WARN] Targeted PGD: audio too short ({x.numel()} samples)")
            return x.cpu().numpy(), False

        encoded_target = self._encode_transcription(target_string).to(self.device)
        target_lengths = torch.tensor([encoded_target.numel()], dtype=torch.long, device=self.device)

        delta = torch.zeros_like(x, requires_grad=True, device=self.device)
        target_clean = target_string.replace(" ", "|").upper()
        success = False

        for i in range(num_iter):
            inp = torch.clamp(x + delta, -1.0, 1.0).unsqueeze(0)
            emissions, _ = self.model(inp)
            logp = F.log_softmax(emissions, dim=-1).transpose(0, 1)
            output_lengths = torch.tensor([emissions.shape[1]], dtype=torch.long, device=self.device)

            loss = F.ctc_loss(
                logp, encoded_target.unsqueeze(0),
                output_lengths, target_lengths,
                blank=self.blank, reduction="mean", zero_infinity=True,
            )
            loss.backward()

            with torch.no_grad():
                delta -= alpha * delta.grad.sign()      # DESCEND (targeted)
                delta.clamp_(-epsilon, epsilon)
            delta.grad.zero_()

            if (i + 1) % check_every == 0:
                with torch.no_grad():
                    pred = self.INFER(torch.clamp(x + delta, -1.0, 1.0))
                if verbose:
                    print(f"  [Step {i+1:4d}] loss={loss.item():.4f}  pred: {pred!r}")
                if pred.strip().upper() == target_clean.strip():
                    if verbose:
                        print(f"  [SUCCESS] Target reached at iteration {i+1}")
                    success = True
                    break

        adv = torch.clamp(x + delta.detach(), -1.0, 1.0)
        return adv.detach().cpu().numpy(), success



    def FGSM_ATTACK(
        self,
        input__: torch.Tensor,
        target: Optional[Union[str, List[str]]] = None,
        epsilon: float = 0.02,
        targeted: bool = False,
    ) -> np.ndarray:
        """
        Untargeted FGSM: maximize CTC loss against the current transcription.
        input__: (time,)
        """
        if targeted:
            raise ValueError("This script is for UNTARGETED attacks only. Set targeted=False.")

        input_ = input__.clone().detach().to(self.device).float()
        input_ = torch.clamp(input_, -1.0, 1.0)

        # ---- Skip short audio ----
        if input_.numel() < MIN_AUDIO_SAMPLES:
            print(f"[WARN] Skipping FGSM: audio too short ({input_.numel()} samples)")
            return input_.cpu().numpy()

        # ---- Add batch dimension ----
        input_ = input_.unsqueeze(0)  # (1, time)
        input_.requires_grad = True

        emissions, _ = self.model(input_)
        logp = F.log_softmax(emissions, dim=-1)  # (B,T,V)

        # Use model's own decoded transcription as "target" to move away from
        untarget = self.INFER(input_.detach())
        encoded = self._encode_transcription(untarget).to(self.device)

        # CTC expects (T,B,V)
        logp_ctc = logp.transpose(0, 1)  # (T,B,V)
        output_lengths = torch.tensor([logp.shape[1]], dtype=torch.long, device=self.device)
        target_lengths = torch.tensor([encoded.numel()], dtype=torch.long, device=self.device)

        loss = F.ctc_loss(
            logp_ctc,
            encoded.unsqueeze(0),  # (N,S)
            output_lengths,
            target_lengths,
            blank=self.blank,
            reduction="mean",
            zero_infinity=True,
        )
        loss.backward()

        # Untargeted: ascend loss
        perturbation = epsilon * input_.grad.sign()
        adv = torch.clamp(input_ + perturbation, -1.0, 1.0)
        return adv.detach().cpu().numpy()

    def PGD_ATTACK(
        self,
        input__: torch.Tensor,
        target: Optional[Union[str, List[str]]] = None,
        epsilon: float = 0.02,
        alpha: float = 0.004,
        num_iter: int = 10,
        nested: bool = True,
        targeted: bool = False,
        early_stop: bool = False,
    ) -> np.ndarray:
        """
        Untargeted PGD (Linf): maximize CTC loss against the current transcription.
        input__: (time,)
        """
        if targeted:
            raise ValueError("This script is for UNTARGETED attacks only. Set targeted=False.")

        x = input__.clone().detach().to(self.device).float()
        x = torch.clamp(x, -1.0, 1.0)

        # ---- Skip short audio ----
        if x.numel() < MIN_AUDIO_SAMPLES:
            print(f"[WARN] Skipping audio too short for PGD ({x.numel()} samples)")
            return x.cpu().numpy()

        # delta is the perturbation variable, constrained in [-epsilon, epsilon]
        delta = torch.zeros_like(x, requires_grad=True, device=self.device)

        leave = False if nested else True

        for _ in tqdm(range(num_iter), desc="PGD", leave=leave):
            # Decode current transcription from (x + delta) each step (closer to your original code’s intent)
            with torch.no_grad():
                untarget = self.INFER(torch.clamp(x + delta, -1.0, 1.0))

            encoded = self._encode_transcription(untarget).to(self.device)

            inp = torch.clamp(x + delta, -1.0, 1.0).unsqueeze(0)  # (1, T)
            emissions, _ = self.model(inp)
            # emissions, _ = self.model(torch.clamp(x + delta, -1.0, 1.0))
            logp = F.log_softmax(emissions, dim=-1)  # (B,T,V)

            logp_ctc = logp.transpose(0, 1)  # (T,B,V)
            output_lengths = torch.tensor([logp.shape[1]], dtype=torch.long, device=self.device)
            target_lengths = torch.tensor([encoded.numel()], dtype=torch.long, device=self.device)

            loss = F.ctc_loss(
                logp_ctc,
                encoded.unsqueeze(0),  # (N,S)
                output_lengths,
                target_lengths,
                blank=self.blank,
                reduction="mean",
                zero_infinity=True,
            )

            loss.backward()

            # Untargeted: ascend loss
            with torch.no_grad():
                delta += alpha * delta.grad.sign()
                delta.clamp_(-epsilon, epsilon)

            delta.grad.zero_()

            # Optional early stop: stop if transcription changes from the initial decode
            if early_stop:
                with torch.no_grad():
                    new_pred = self.INFER(torch.clamp(x + delta, -1.0, 1.0))
                if new_pred != untarget:
                    break

        adv = torch.clamp(x + delta.detach(), -1.0, 1.0)
        return adv.detach().cpu().numpy()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Input folder (recursive) containing audio.")
    ap.add_argument("--out_dir", type=str, help="Output folder for adversarial WAVs.")
    # --- ADDED GAUSSIAN TO CHOICES ---
    ap.add_argument("--perturbation", type=str, choices=["fgsm", "pgd", "gaussian", "ambient", "time-stretch", "freq-shift", "targeted-pgd"], default="targeted-pgd")
    ap.add_argument("--epsilon", type=float, default=0.02, help="Max Linf perturbation (e.g., 0.005–0.03).")
    ap.add_argument("--alpha", type=float, default=0.004, help="PGD step size (e.g., epsilon/5 to epsilon/10).")
    # --- ADDED SNR_DB ARGUMENT ---
    ap.add_argument("--snr_db", type=float, default=10.0, help="SNR in dB for Gaussian noise.")
    ap.add_argument("--speech_rate", type=float, default=1.0, help="Speech rate multiplier for time-stretching (e.g., 0.8 for slower, 1.2 for faster).")
    # ap.add_argument("--n_steps", type=float, default=2.0, help="Semitones for pitch shift (e.g., ±1 to ±4).")
    ap.add_argument("--shift_hz", type=float, default=100.0, help="Hz shift for frequency shift (e.g., ±50 to ±200).")
    ap.add_argument("--target_string", type=str, default="I DO NOT KNOW WHERE TO FIND A HITMAN AND I NEED ONE",
                help="Target transcription for targeted-pgd (uppercase A-Z and spaces only).")
    ap.add_argument("--iters", type=int, default=10, help="PGD iterations (ignored for FGSM).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exts", type=str, default=".wav,.flac", help="Comma-separated extensions to include.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest", type=str, default="adv_manifest.jsonl", help="Manifest filename (JSONL).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if args.out_dir is None:
        out_dir = in_dir.parent / f"{in_dir.name}_{args.perturbation}"        #in_dir.name
    else:
        out_dir = Path(args.out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])

    # ASR model + attacker
    asr = Wav2Vec2ASRWrapper(device=args.device)
    attacker = ASRAttacks(model=asr, device=args.device, labels=asr.labels)

    # Gather inputs
    #audio_root = in_dir/"styletts2"
    #files = list_audio_files(audio_root, exts=exts)

    # Gather ALL inputs recursively across all subfolders
    files = list_audio_files(in_dir, exts=exts)
    if not files:
        raise RuntimeError(f"No audio found under {in_dir} with extensions {exts}")
    
    # ----------------------------
    # Load and merge manifests
    # ----------------------------
    # manifest_dir = in_dir / "manifests"
    # manifest_files = sorted(manifest_dir.glob("*.jsonl"))

    manifest_files = sorted(in_dir.rglob("*.jsonl"))
    base_rows = []
    for mf in manifest_files:
        with mf.open("r", encoding="utf-8") as f:
            for line in f:
                base_rows.append(json.loads(line))

    print(f"[INFO] Loaded {len(base_rows)} rows from {len(manifest_files)} manifest files")

    # Build mapping: generated.path -> row
    def norm(p):
        return str(Path(p).as_posix())

    base_map = {}
    for r in base_rows:
        try:
            rel_path = norm(r["generated"]["path"])
            base_map[rel_path] = r
        except KeyError:
            continue
    
    # sanity check that merged_manifest was loaded correctly
    print(f"[INFO] Manifest entries: {len(base_map)}, Audio files: {len(files)}")


    manifest_path = out_dir / "manifests"/ args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with manifest_path.open("w", encoding="utf-8") as mf:
        for src_path in tqdm(files, desc=f"Generating {args.perturbation} audio"):
            try:
                rel = src_path.relative_to(in_dir)        # SG
                #rel = src_path.relative_to(audio_root)      #SG
            except ValueError:
                raise RuntimeError(f"{src_path} is not under input root {in_dir}")
            # rel = src_path.relative_to(in_dir)
            dst_path = (out_dir / rel).with_suffix(".wav")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            _success = None  # set by targeted-pgd; None for other perturbations
 
             # --- METADATA MATCHING LOGIC ---
            try:
                styletts_idx = src_path.parts.index("styletts2")
                rel_src = norm(Path(*src_path.parts[styletts_idx:]))
            except ValueError:
                rel_src = norm(rel)


            if dst_path.exists() and not args.overwrite:
                # Match using relative path from in_dir
                #rel_src = norm(src_path.relative_to(in_dir))

                base = base_map.get(rel_src)

                if base is None:
                    print(f"[WARN] No manifest match for: {rel_src}")
                    base = {}
                else:
                    base = base.copy()

                base.update({
                    "adv_audio_filepath": str(dst_path),
                    "perturbation": args.perturbation,
                    "target_string": args.target_string if args.perturbation == "targeted-pgd" else None,
                    "target_reached": bool(_success) if args.perturbation == "targeted-pgd" else None,
                    "epsilon": float(args.epsilon),
                    "alpha": float(args.alpha),
                    "snr_db": float(args.snr_db) if args.perturbation in ["gaussian", "ambient"] else None,
                    "speech_rate": float(args.speech_rate) if args.perturbation == "time-stretch" else 1.0,
                    # "n_steps": float(args.n_steps) if args.perturbation == "pitch-shift" else None,
                    "shift_hz": float(args.shift_hz) if args.perturbation == "freq-shift" else None,
                    "iters": int(args.iters) if args.perturbation == "pgd" else 1,
                    "sample_rate": TARGET_SR,
                    "status": "skipped_exists" if (dst_path.exists() and not args.overwrite) else "ok",
                })

                mf.write(json.dumps(base, ensure_ascii=False) + "\n")

                continue

            # Load + preprocess
            wav, sr = load_mono(src_path)  # (1, time)
            wav16 = clamp_audio(resample_to(wav, sr, TARGET_SR)).to(args.device)

            # Attacks expect 1D tensor (time,)
            x = wav16.squeeze(0)

            # Generate adversarial example
            if args.perturbation == "fgsm":
                adv_np = attacker.FGSM_ATTACK(
                    x,
                    target=None,
                    epsilon=float(args.epsilon),
                    targeted=False,
                )
            elif args.perturbation == "pgd":
                adv_np = attacker.PGD_ATTACK(
                    x,
                    target=None,
                    epsilon=float(args.epsilon),
                    alpha=float(args.alpha),
                    num_iter=int(args.iters),
                    nested=True,
                    targeted=False,
                    early_stop=False,
                )
            elif args.perturbation == "gaussian":
                adv_np = add_gaussian_noise(
                    audio=x.detach().cpu().numpy(),
                    snr_db=float(args.snr_db),
                    sample_rate=TARGET_SR,
                    noise_bandwidth=8000
                )

            elif args.perturbation == "ambient":
                ambient_noise_path = Path("misc/ambient_noise.mp3")  # Path to your ambient noise file
                adv_np = add_ambient_noise(
                    audio=x.detach().cpu().numpy(),
                    snr_db=float(args.snr_db),
                    sample_rate=TARGET_SR,
                    ambient_noise_path=ambient_noise_path
                )
            # elif args.perturbation == "pitch-shift":
            #     adv_np = pitch_shift_audio(
            #         audio=x.detach().cpu().numpy(),
            #         sample_rate=TARGET_SR,
            #         steps=float(args.n_steps),
            #     )
            elif args.perturbation == "time-stretch":
                adv_np = time_stretch_audio(
                    audio=x.detach().cpu().numpy(),
                    rate=float(args.speech_rate),
                )
            elif args.perturbation == "freq-shift":
                adv_np = frequency_shift_audio(
                    audio=x.detach().cpu().numpy(),
                    sample_rate=TARGET_SR,
                    shift_hz=float(args.shift_hz),
                )
            elif args.perturbation == "targeted-pgd":
                adv_np, _success = attacker.TARGETED_PGD_ATTACK(
                    x,
                    target_string=args.target_string,
                    epsilon=float(args.epsilon),
                    alpha=float(args.alpha),
                    num_iter=int(args.iters),
                    check_every=50,
                    verbose=True,
                )
            else:
                _success = None  # not applicable for non-targeted attacks    

            adv = torch.from_numpy(np.asarray(adv_np)).float()
            if adv.dim() == 1:
                adv = adv.unsqueeze(0)
            adv = clamp_audio(adv).cpu()

            # Save adversarial wav
            ta.save(str(dst_path), adv, TARGET_SR)

            # Manifest line
            # Match using relative path from in_dir
            # rel_src = norm(src_path.relative_to(in_dir))

            base = base_map.get(rel_src)

            if base is None:
                print(f"[WARN] No manifest match for: {rel_src}")
                base = {}
            else:
                base = base.copy()

            base.update({
                "adv_audio_filepath": str(dst_path),
                "perturbation": args.perturbation,
                "target_string": args.target_string if args.perturbation == "targeted-pgd" else None,
                "target_reached": bool(_success) if args.perturbation == "targeted-pgd" else None,
                "epsilon": float(args.epsilon) if args.perturbation in ["fgsm", "pgd", "targeted-pgd"] else None,
                "alpha": float(args.alpha) if args.perturbation in ["fgsm", "pgd", "targeted-pgd"] else None,
                "snr_db": float(args.snr_db) if args.perturbation in ["gaussian", "ambient"] else None, # ADDED
                "speech_rate": float(args.speech_rate) if args.perturbation == "time-stretch" else 1.0, # ADDED
                # "n_steps": float(args.n_steps) if args.perturbation == "pitch-shift" else None,
                "shift_hz": float(args.shift_hz) if args.perturbation == "freq-shift" else None,
                "iters": int(args.iters) if args.perturbation in ["pgd", "targeted-pgd"] else 1,
                "sample_rate": TARGET_SR,
                "status": "skipped_exists" if (dst_path.exists() and not args.overwrite) else "ok",
            })

            mf.write(json.dumps(base, ensure_ascii=False) + "\n")

    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()