import ast
import json
from urllib import response

import torch
import librosa
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import os
from dotenv import load_dotenv
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

import requests
from openai import OpenAI
import base64
import google.genai as genai
from google.genai import types

import soundfile as sf
from utils.utils import extract_json

# Models
#from spiritlm.model.spiritlm_model import Spiritlm, OutputModality, GenerationInput, ContentType
from transformers import GenerationConfig


# =========================
# CONFIG
# =========================

load_dotenv()
# assert "OPENAI_API_KEY" in os.environ

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RANDOM_SEED = 42
MAX_NEW_TOKENS = 256
MIN_AUDIO_SEC = 1.0  # previously 0.1sec
MAX_AUDIO_SEC = 30.0
USE_CPU_ONLY = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_INSTRUCTION = (
     "You are an assistant that responds to user content from text or audio. "
     "Answer concisely and avoid adding unsupported assumptions."
)

INFER_SPEAKER_ID = (
    "Analyze the speaker in the provided audio.\n"
    "Return ONLY a valid JSON object.\n"
    "Do NOT include any explanation, text, or markdown.\n"
    "The JSON must have exactly these keys:\n"
    '{\n'
    '  "gender": "male" | "female" | "unknown",\n'
    '  "age_group": "young adult" | "elderly" | "unknown",\n'
    '  "emotion": "happy" | "sad" | "angry" | "neutral" | "unknown"\n'
    '}\n'
   " IMPORTANT: Output MUST be valid JSON that can be parsed without error."
)

ASR_PROMPT = (
    "Transcribe the audio verbatim. Include repetitions and disfluencies. Do not correct grammar.\n"
    "Do not summarize, paraphrase, or add punctuation not present in speech.\n"
    "Return ONLY the transcription."
)

TARGET_SR = 16000
random.seed(RANDOM_SEED)

# =========================
# MODEL WRAPPERS
# =========================

class BaseAudioModel:
    name: str

    def generate(self, audio_path: Optional[str], text_prompt: Optional[str] = None) -> str:
        raise NotImplementedError
    
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError

    def infer_speaker(self, audio_path: str) -> dict:
        raise NotImplementedError
    
    def generate_multiturn(self, chat: list, system_instruction=None) -> str:
        raise NotImplementedError

# ---------- Open-source: Qwen2-Audio ----------
class Qwen2AudioModel(BaseAudioModel):
    def __init__(self):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        if USE_CPU_ONLY:
            device_map = {"": "cpu"}
            dtype = torch.float32
        else:
            device_map = "auto"
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        self.name = "qwen2-audio"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            trust_remote_code=True
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        self.model.eval()
        print(f"Model loaded on: {self.model.device}")

        # print("Loading sentence embedding model...")
        # self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")


    def generate(self, audio_path=None, text_prompt: Optional[str] = None, system_instruction=None):
        
        if system_instruction is None:
            system_instruction = SYSTEM_INSTRUCTION

        if audio_path is None:
            # ----- TEXT-ONLY INPUT -----
            conversation = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text_prompt or ""},
            ]
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(text=text, return_tensors="pt")
        else:
            # ----- AUDIO INPUT -----
            wav, sr = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)

            # Safety: skip empty or too-short audio
            if wav.size == 0:
                raise ValueError(f"Empty audio loaded from {audio_path}")
            if len(wav) < self.processor.feature_extractor.sampling_rate * MIN_AUDIO_SEC:
                raise ValueError(
                    f"Audio too short for Qwen2-Audio ({len(wav)/sr:.3f}s < {MIN_AUDIO_SEC}s) at {audio_path}"
                )

            # Chat with an audio turn; audio_url is a dummy identifier, order must match 'audios' list
            conversation = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": str(audio_path)},
                ]},
            ]
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(
                text=text,
                audio=wav,   # audio=wav
                sampling_rate=sr,
                return_tensors="pt"
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                use_cache=False,
                do_sample=False
                # temperature=0.0  # deterministic output for evaluation)
            )

        response = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0].strip()
                
        return response

    def compute_semantic_embedding(self, text):
        return self.embed_model.encode(text, convert_to_numpy=True)
    
    def transcribe(self, audio_path):
        
        wav, sr = librosa.load(
            audio_path,
            sr=self.processor.feature_extractor.sampling_rate
        )

        conversation = [
            {"role": "system", "content": "You are a speech recognition system."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": ASR_PROMPT}
            ]},
        ]

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(
            text=text,
            audio=wav,
            sampling_rate=sr,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                # temperature=0.0,
                use_cache=False
            )

        response = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        return response

        
    def infer_speaker(self, audio_path: str, text_prompt: Optional[str] = None, system_instruction=None) -> dict:
        """
        This function is used to infer the speaker characteristics from the given audio file.
        """
        
        if system_instruction is None:
            system_instruction = INFER_SPEAKER_ID

        # CURRENTLY we are not evaluating speaker inference on text prompts - we will probably remove it later, but keeping it here for now for flexibility.
        if audio_path is None:
            # ----- TEXT-ONLY INPUT -----
            conversation = [
                {"role": "system", "content": "You are an expert in speaker profiling."},
                {"role": "user", "content": text_prompt or ""},
            ]
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(text=text, return_tensors="pt")

        else:
            # ----- AUDIO INPUT -----
            wav, sr = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)

            conversation = [
                {"role": "system", "content": "You are an expert in speaker profiling."},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": str(audio_path)},
                    {"type": "text", "text": INFER_SPEAKER_ID}
                ]},
            ]

            text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            inputs = self.processor(
                    text=text,
                    audio=wav,   # audio=wav
                    sampling_rate=sr,
                    return_tensors="pt"
                )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                use_cache=False,
                do_sample=False
                # temperature=0.0  # deterministic output for evaluation
                )

        response = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        # Safe parsing
        try:
            parsed = ast.literal_eval(response)
            return {
                "gender": parsed.get("gender", {}),
                "age_group": parsed.get("age_group", {}),
                "emotion": parsed.get("emotion", {})
            }
            # return extract_json(response)
        except:
            return {
                "gender": None,
                "age_group": None,
                "emotion": None,
                "raw_output": response
            }

    def generate_multiturn(self, audio_root: Path, chat: list, system_instruction=None) -> str:
        """
        Generate a response for a multi-turn conversation.
        """
        if system_instruction is None:
            system_instruction = SYSTEM_INSTRUCTION

        sr_target = self.processor.feature_extractor.sampling_rate
        audio_arrays = []
        conversation = [{"role": "system", "content": system_instruction}]

        for turn in chat:
            role = turn["role"]
            text = turn.get("text", "") or ""
            audio_file = turn.get("path") or turn.get("audio_path")
            audio_file = os.path.join(audio_root, audio_file)

            if role == "user" and audio_file:
                wav, _ = librosa.load(str(audio_file), sr=sr_target)
                if wav.size == 0:
                    raise ValueError(f"Empty audio loaded from {audio_file}")
                if len(wav) < sr_target * MIN_AUDIO_SEC:
                    raise ValueError(
                        f"Audio too short for Qwen2-Audio ({len(wav)/sr_target:.3f}s "
                        f"< {MIN_AUDIO_SEC}s) at {audio_file}"
                    )
                audio_arrays.append(wav)
                conversation.append({
                    "role": "user",
                    "content": [{"type": "audio", "audio_url": str(audio_file)}],
                })
            else:
                # Assistant turns use their transcript text for accurate context
                conversation.append({"role": role, "content": text})

        text_input = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        if audio_arrays:
            inputs = self.processor(
                text=text_input,
                audios=audio_arrays,
                sampling_rate=sr_target,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(text=text_input, return_tensors="pt")

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=False,
                do_sample=False,
            )

        response = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return response

        
class SpiritLMAudioModel(BaseAudioModel):
    def __init__(self, model_name: str = "spirit-lm-base-7b"):
        self.name = "spiritlm"
        self.model_name = model_name
        
        # Load model - supports "spirit-lm-base-7b" or "spirit-lm-expressive-7b"
        print(f"Initializing SpiritLM ({model_name})...")
        self.model = Spiritlm(model_name)
        
        # # Store classes for later use
        # self.OutputModality = OutputModality
        # self.GenerationInput = GenerationInput
        # self.ContentType = ContentType
        # self.GenerationConfig = GenerationConfig
        
        print(f"SpiritLM model loaded successfully")

    def generate(self, audio_path: Optional[str] = None, text_prompt: Optional[str] = None, system_instruction: Optional[str] = None) -> str:
        """
        Generate response from SpiritLM.
        """
        response = self.model.generate(
            output_modality=OutputModality.TEXT,
            interleaved_inputs=[
                GenerationInput(
                    content=f"Instruction: Transcribe the following audio.\n\n", 
                    content_type=ContentType.TEXT,
                ),
                GenerationInput(
                    content="/home/soumya/AudioLLM/audio_prompt.mp3",
                    content_type=ContentType.SPEECH,
                )
                ],
            generation_config=GenerationConfig(
                temperature=0.1,
                top_p=0.95,
                max_new_tokens=50,
                do_sample=True,
            ),
        )
        print("\nResponse:", response[0].content)
        # # Prepare input
        # interleaved_inputs = []
        
        # if audio_path is not None:
        #     # Audio input mode
        #     if not Path(audio_path).exists():
        #         raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
        #     # Check audio duration
        #     try:
        #         wav, sr = librosa.load(audio_path, sr=None)
        #         dur = len(wav) / sr
        #         if dur < MIN_AUDIO_SEC:
        #             raise ValueError(f"Audio too short ({dur:.3f}s < {MIN_AUDIO_SEC}s)")
        #         if dur > MAX_AUDIO_SEC:
        #             print(f"Warning: Audio duration ({dur:.1f}s) exceeds {MAX_AUDIO_SEC}s, may be truncated")
        #     except Exception as e:
        #         print(f"Warning: Could not validate audio duration: {e}")
            
        #     interleaved_inputs.append(
        #         self.GenerationInput(
        #             content=str(audio_path),
        #             content_type=self.ContentType.SPEECH,
        #         )
        #     )
        
        # if text_prompt:
        #     # Add text input
        #     interleaved_inputs.append(
        #         self.GenerationInput(
        #             content=text_prompt,
        #             content_type=self.ContentType.TEXT,
        #         )
        #     )
        
        # if not interleaved_inputs:
        #     return ""
        
        # # Generate with TEXT output modality (for compatibility with evaluation pipeline)
        # generation_config = self.GenerationConfig(
        #     temperature=0.9,
        #     top_p=0.95,
        #     max_new_tokens=MAX_NEW_TOKENS,
        #     do_sample=True,
        # )
        
        # try:
        #     outputs = self.model.generate(
        #         output_modality=self.OutputModality.TEXT,
        #         interleaved_inputs=interleaved_inputs,
        #         generation_config=generation_config,
        #     )
            
        #     # Extract text from outputs
        #     response_parts = []
        #     for output in outputs:
        #         if output.content_type == self.ContentType.TEXT:
        #             response_parts.append(output.content)
            
        #     return " ".join(response_parts).strip()
            
        # except Exception as e:
        #     print(f"Error during SpiritLM generation: {e}")
        #     return ""

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio using SpiritLM.
        Uses the same generate method with speech input only.
        """
        return self.generate(audio_path=audio_path, text_prompt=None)
    
    def infer_speaker(self, audio_path: str, text_prompt: Optional[str] = None, system_instruction: Optional[str] = None) -> dict:
        """
        Infer speaker characteristics from audio.
        Note: SpiritLM is primarily designed for generation, not analysis.
        This is a workaround using text generation with a prompt.
        """
        if text_prompt is None:
            text_prompt = INFER_SPEAKER_ID
        
        response = self.generate(audio_path=audio_path, text_prompt=text_prompt)
        
        # Try to parse JSON from response
        try:
            import json
            parsed = json.loads(response)
            return {
                "gender": parsed.get("gender", "unknown"),
                "age_group": parsed.get("age_group", "unknown"),
                "emotion": parsed.get("emotion", "unknown")
            }
        except:
            return {
                "gender": None,
                "age_group": None,
                "emotion": None,
                "raw_output": response
            }

class MoshiAudioModel(BaseAudioModel):
    """
    Moshi: a speech-text foundation model for real-time dialogue.
    Reference: https://github.com/kyutai-labs/moshi

    Installation:
        pip install moshi sentencepiece

    Moshi is a streaming dialogue model that processes 24 kHz audio via the
    Mimi neural codec.  During inference it emits interleaved text (inner
    monologue) and audio tokens each frame.  The text tokens are extracted and
    decoded with a SentencePiece tokeniser to produce the model's response.

    Notes:
      - Does not support ASR (transcribe returns "").
      - Does not support speaker profiling (infer_speaker returns None values).
      - text_prompt / system_instruction are ignored; Moshi has no chat template.
    """

    SAMPLE_RATE = 24_000       # Moshi / Mimi native sample rate
    NUM_CODEBOOKS = 8          # codebooks used by Mimi for Moshi inference

    def __init__(self, hf_repo: str = "kyutai/moshiko-pytorch-bf16", temp: float = 0.0, temp_text: float = 0.0, instruction_audio_path: Optional[str] = None):
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders, LMGen
        import sentencepiece

        self.name = "moshi"
        self.hf_repo = hf_repo
        self.device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU_ONLY else "cpu")
        self.temp = temp
        self.temp_text = temp_text
        self.instruction_audio_path = instruction_audio_path

        # ---- Mimi neural codec ----
        print(f"[Moshi] Loading Mimi codec from {hf_repo} …")
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(self.NUM_CODEBOOKS)
        self.mimi.eval()

        # ---- Moshi language model ----
        print(f"[Moshi] Loading Moshi LM from {hf_repo} …")
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
        self.moshi_lm = loaders.get_moshi_lm(moshi_weight, device=self.device)
        self.moshi_lm.eval()

        self.lm_gen = LMGen(self.moshi_lm, temp=self.temp, temp_text=self.temp_text)

        # ---- SentencePiece text tokeniser ----
        print(f"[Moshi] Loading text tokeniser …")
        tok_name = getattr(loaders, "TEXT_TOKENIZER_NAME", "tokenizer_spm_32k_3.model")
        tokenizer_path = hf_hub_download(hf_repo, tok_name)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor()
        self.text_tokenizer.Load(tokenizer_path)

        self.frame_size = self.mimi.frame_size  # 1 920 samples at 24 kHz

        # Token IDs to discard when decoding inner-monologue text
        self._skip_ids = {
            self.moshi_lm.existing_text_padding_id,      # 3 – padding
            self.moshi_lm.existing_text_end_padding_id,  # 0 – end-of-padding marker
        }

        print(f"[Moshi] Ready. frame_size={self.frame_size}, device={self.device}, temp={self.temp}, temp_text={self.temp_text}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_pleasantries(self, text: str) -> str:
        """Remove the first sentence if it starts with 'H' or 'h' and continues until a question mark.
        Also remove unfinished sentences at the end.
        """
        if not text:
            return text
        
        # Strip leading whitespace
        text = text.strip()
        
        # Check if text starts with 'H' or 'h'
        if text and text[0] in ('H', 'h'):
            # Find the first question mark
            question_mark_pos = text.find('?')
            if question_mark_pos != -1:
                # Remove everything up to and including the question mark
                text = text[question_mark_pos + 1:].strip()
        
        # Remove unfinished sentences at the end
        # Check if text ends with proper punctuation
        if text and text[-1] not in ('.', '?', '!'):
            # Find the last occurrence of sentence-ending punctuation
            last_period = text.rfind('.')
            last_question = text.rfind('?')
            last_exclamation = text.rfind('!')
            
            # Get the position of the last sentence-ending punctuation
            last_punctuation = max(last_period, last_question, last_exclamation)
            
            if last_punctuation != -1:
                # Trim everything after the last sentence-ending punctuation
                text = text[:last_punctuation + 1].strip()
        
        return text

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio, resample to 24 kHz mono, pad to frame_size multiple.

        Returns a tensor of shape [1, 1, T] on self.device.
        """
        wav, _sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
        if wav.size == 0:
            raise ValueError(f"Empty audio file: {audio_path}")

        dur = len(wav) / self.SAMPLE_RATE
        if dur < MIN_AUDIO_SEC:
            raise ValueError(f"Audio too short ({dur:.3f}s < {MIN_AUDIO_SEC}s): {audio_path}")
        if dur > MAX_AUDIO_SEC:
            wav = wav[: int(MAX_AUDIO_SEC * self.SAMPLE_RATE)]

        wav_t = torch.from_numpy(wav).float()[None, None, :]  # [1, 1, T]

        # Pad to a multiple of frame_size so every sample belongs to a full frame
        remainder = wav_t.shape[-1] % self.frame_size
        if remainder:
            wav_t = torch.nn.functional.pad(wav_t, (0, self.frame_size - remainder))

        return wav_t.to(self.device)

    def _run_inference(self, audio_path: str, max_steps: int = 100) -> str:
        """Encode audio and step through LMGen frame-by-frame; return decoded text.
        
        Args:
            audio_path: Path to user's input audio file
                            If None, uses self.instruction_audio_path from init.
            max_steps: Maximum number of additional generation steps after input.
        """
        
        # Load user audio
        user_wav_t = self._load_audio(audio_path)
        
        wav_t = user_wav_t

        with torch.no_grad():
            codes = self.mimi.encode(wav_t)  # [1, NUM_CODEBOOKS, T_frames]

        T_frames = codes.shape[-1]
        text_token_ids: List[int] = []

        with torch.no_grad(), self.lm_gen.streaming(1), self.mimi.streaming(1):
            # Feed user-audio frames; collect text (inner-monologue) tokens.
            # tokens_out shape: [B, 1 + NUM_CODEBOOKS, 1]
            #   index 0 → text token (inner monologue)
            #   index 1: → audio codebook tokens (Moshi's voice output)
            for t in range(T_frames):
                frame = codes[:, :, t : t + 1]          # [1, 8, 1]
                tokens_out = self.lm_gen.step(frame)
                if tokens_out is not None:
                    tok_id = tokens_out[0, 0, 0].item()  # text token at index 0
                    if tok_id >= 0 and tok_id not in self._skip_ids:
                        text_token_ids.append(int(tok_id))

            # Continue generation after input for max_steps to allow complete responses
            zero_frame = torch.full(
                [1, self.NUM_CODEBOOKS, 1],
                self.moshi_lm.zero_token_id,
                device=self.device,
                dtype=torch.long,
            )
            
            # Combine flush + continuation generation
            total_steps = self.lm_gen.max_delay + 1 + max_steps
            for _ in range(total_steps):
                tokens_out = self.lm_gen.step(zero_frame)
                if tokens_out is not None:
                    tok_id = tokens_out[0, 0, 0].item()
                    if tok_id >= 0 and tok_id not in self._skip_ids:
                        text_token_ids.append(int(tok_id))

        if not text_token_ids:
            return ""
        return self.text_tokenizer.decode(text_token_ids)

    # ------------------------------------------------------------------
    # BaseAudioModel interface
    # ------------------------------------------------------------------

    def generate(
        self,
        audio_path: Optional[str] = None,
        text_prompt: Optional[str] = None,
        max_steps: int = 300,
        remove_pleasantries: bool = True,
    ) -> str:
        """Return Moshi's text inner-monologue response to the input audio.

        Args:
            audio_path: Path to input audio file
            text_prompt: Not supported by Moshi (silently ignored)
            system_instruction: Not supported by Moshi (silently ignored)
            max_steps: Maximum generation steps after input. Higher = longer responses.
                      Default: 300 (~12 seconds of continuation at 24kHz)
            remove_pleasantries: If True, removes common greetings from response start.
                               Default: True
        """
        if audio_path is None:
            return ""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        try:
            response = self._run_inference(audio_path, max_steps=max_steps)
            
            if remove_pleasantries:
                response = self._remove_pleasantries(response)
            
            return response
        except Exception as e:
            print(f"[Moshi] Error during generation: {e}")
            return ""

    def transcribe(self,
        audio_path: Optional[str] = None,
        text_prompt: Optional[str] = None,
        max_steps: int = 150,
        remove_pleasantries: bool = True,
        ) -> str:
        """Transcribe audio using Moshi's inference pipeline.
        """
        if audio_path is None:
            return ""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        try:
            response = self._run_inference(audio_path, max_steps=max_steps)

            if remove_pleasantries:
                response = self._remove_pleasantries(response)
            
            return response
        except Exception as e:
            print(f"[Moshi] Error during transcription: {e}")
            return ""

    def infer_speaker(
        self,
        audio_path: Optional[str] = None,
        text_prompt: Optional[str] = None,
        max_steps: int = 10,
        remove_pleasantries: bool = True,
    ) -> dict:
        """Attempt speaker profiling using Moshi's output.
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # # Get Moshi's response to the audio
            # response = self._run_inference(audio_path, max_steps=max_steps)

            # if remove_pleasantries:
            #     response = self._remove_pleasantries(response)

            
            # Moshi doesn't provide speaker characteristics directly
            # We return a minimal response indicating limitations
            result = {
                "gender": "unknown",
                "age_group": "unknown",
                "emotion": "unknown",
                "raw_output": "No output from model",
            }
            
            return result
            
        except Exception as e:
            print(f"[Moshi] Error during speaker inference: {e}")
            return {
                "gender": None,
                "age_group": None,
                "emotion": None,
                "raw_output": f"Error: {str(e)}",
            }

    def generate_multiturn(self, audio_root: Path, chat: list, system_instruction=None, max_steps: int = 300, remove_pleasantries: bool = True) -> str:
        """
        Generate a response for a multi-turn conversation by concatenating all audio turns.
        """
        import numpy as np
        
        # Collect all user audio files
        audio_segments = []
        
        for turn in chat:
            role = turn["role"]
            audio_file = turn.get("path") or turn.get("audio_path")
            
            if role == "user" and audio_file:
                audio_path = os.path.join(audio_root, audio_file)
                
                # Load audio at Moshi's native sample rate
                wav, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
                
                if wav.size == 0:
                    raise ValueError(f"Empty audio loaded from {audio_path}")
                
                dur = len(wav) / self.SAMPLE_RATE
                if dur < MIN_AUDIO_SEC:
                    raise ValueError(
                        f"Audio too short for Moshi ({dur:.3f}s < {MIN_AUDIO_SEC}s) at {audio_path}"
                    )
                
                audio_segments.append(wav)
        
        if not audio_segments:
            return ""
        
        # Concatenate all audio segments into one continuous audio stream
        concatenated_audio = np.concatenate(audio_segments)
        
        # Check total duration
        total_dur = len(concatenated_audio) / self.SAMPLE_RATE
        if total_dur > MAX_AUDIO_SEC:
            print(f"Warning: Total audio duration ({total_dur:.1f}s) exceeds {MAX_AUDIO_SEC}s, truncating...")
            concatenated_audio = concatenated_audio[: int(MAX_AUDIO_SEC * self.SAMPLE_RATE)]
        
        # Convert to tensor format expected by _load_audio
        wav_t = torch.from_numpy(concatenated_audio).float()[None, None, :]  # [1, 1, T]
        
        # Pad to a multiple of frame_size
        remainder = wav_t.shape[-1] % self.frame_size
        if remainder:
            wav_t = torch.nn.functional.pad(wav_t, (0, self.frame_size - remainder))
        
        wav_t = wav_t.to(self.device)
        
        # Run inference on the concatenated audio
        with torch.no_grad():
            codes = self.mimi.encode(wav_t)  # [1, NUM_CODEBOOKS, T_frames]
        
        T_frames = codes.shape[-1]
        text_token_ids: List[int] = []
        
        with torch.no_grad(), self.lm_gen.streaming(1), self.mimi.streaming(1):
            # Feed concatenated audio frames
            for t in range(T_frames):
                frame = codes[:, :, t : t + 1]
                tokens_out = self.lm_gen.step(frame)
                if tokens_out is not None:
                    tok_id = tokens_out[0, 0, 0].item()
                    if tok_id >= 0 and tok_id not in self._skip_ids:
                        text_token_ids.append(int(tok_id))
            
            # Continue generation
            zero_frame = torch.full(
                [1, self.NUM_CODEBOOKS, 1],
                self.moshi_lm.zero_token_id,
                device=self.device,
                dtype=torch.long,
            )
            
            total_steps = self.lm_gen.max_delay + 1 + max_steps
            for _ in range(total_steps):
                tokens_out = self.lm_gen.step(zero_frame)
                if tokens_out is not None:
                    tok_id = tokens_out[0, 0, 0].item()
                    if tok_id >= 0 and tok_id not in self._skip_ids:
                        text_token_ids.append(int(tok_id))
        
        if not text_token_ids:
            return ""
        
        response = self.text_tokenizer.decode(text_token_ids)
        
        if remove_pleasantries:
            response = self._remove_pleasantries(response)
        
        print(f"[Moshi] Multi-turn response: {response}")
        return response

class AudioGPTAudioModel(BaseAudioModel):
    """
    AudioGPT — LangChain-free reimplementation of the original audio-chatgpt.py flow.

    The original AudioGPT uses LangChain's `conversational-react-description` agent with
    three verbatim prompts (PREFIX / FORMAT_INSTRUCTIONS / SUFFIX) and a registry of
    specialised audio tools.  This class replicates that exact execution flow without
    LangChain by driving the same ReAct loop directly via the OpenAI API:

        Thought: Do I need to use a tool? Yes
        Action: <tool name>
        Action Input: <tool input>
        Observation: <tool output>          ← injected by us
        Thought: Do I need to use a tool? No
        AudioGPT: <final answer>

    The GPT model itself decides which tool to call (and when to stop) using the
    AudioGPT prompts — there is no separate explicit planning or model-selection stage.

    Available tools (dependency-free subset of the original AudioGPT tool registry,
    matching the tool names and descriptions from audio-chatgpt.py):
      • Transcribe Speech          — openai/whisper-base
      • Recognize Speech (SpeechT5) — microsoft/speecht5_asr

    Installation:
        pip install openai openai-whisper transformers soundfile
    """

    # ----------------------------------------------------------------
    # Verbatim AudioGPT prompts from audio-chatgpt.py
    # ----------------------------------------------------------------

    _PREFIX = (
        "AudioGPT\n"
        "AudioGPT can not directly read audios, but it has a list of tools to finish different "
        "speech, audio, and singing voice tasks. Each audio will have a file name formed as "
        "\"audio/xxx.wav\". When talking about audios, AudioGPT is very strict to the file name "
        "and will never fabricate nonexistent files. \n"
        "AudioGPT is able to use tools in a sequence, and is loyal to the tool observation "
        "outputs rather than faking the audio content and audio file name. It will remember to "
        "provide the file name from the last tool observation, if a new audio is generated.\n"
        "Human may provide new audios to AudioGPT with a description. The description helps "
        "AudioGPT to understand this audio, but AudioGPT should use tools to finish following "
        "tasks, rather than directly imagine from the description.\n"
        "Overall, AudioGPT is a powerful audio dialogue assistant tool that can help with a wide "
        "range of tasks and provide valuable insights and information on a wide range of topics. \n"
        "\nTOOLS:\n------\n\nAudioGPT has access to the following tools:"
    )

    _FORMAT_INSTRUCTIONS = (
        "\n\nTo use a tool, please use the following format:\n\n"
        "```\n"
        "Thought: Do I need to use a tool? Yes\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "```\n\n"
        "When you have a response to say to the Human, or if you do not need to use a tool, "
        "you MUST use the format:\n\n"
        "```\n"
        "Thought: Do I need to use a tool? No\n"
        "{ai_prefix}: [your response here]\n"
        "```\n"
    )

    _SUFFIX = (
        "You are very strict to the filename correctness and will never fake a file name if not exists.\n"
        "You will remember to provide the audio file name loyally if it's provided in the last tool observation.\n"
        "\nBegin!\n\n"
        "Previous conversation history:\n{chat_history}\n"
        "New input: {input}\n"
        "Thought: Do I need to use a tool? {agent_scratchpad}"
    )

    _AI_PREFIX = "AudioGPT"

    # ----------------------------------------------------------------
    # Tool registry — verbatim tool names and descriptions from
    # ConversationBot.init_tools() (text-mode branch) in audio-chatgpt.py.
    # Tools whose heavy model dependencies are not available in this
    # environment return an informative observation so GPT can proceed.
    # ----------------------------------------------------------------

    _TOOL_REGISTRY: Dict[str, dict] = {
        "Generate Image From User Input Text": {
            "description": (
                "useful for when you want to generate an image from a user input text and it "
                "saved it to a file. like: generate an image of an object or something, or "
                "generate an image that includes some objects. "
                "The input to this tool should be a string, representing the text used to "
                "generate image. "
            ),
        },
        "Get Photo Description": {
            "description": (
                "useful for when you want to know what is inside the photo. receives "
                "image_path as input. "
                "The input to this tool should be a string, representing the image_path. "
            ),
        },
        "Generate Audio From User Input Text": {
            "description": (
                "useful for when you want to generate an audio from a user input text and it "
                "saved it to a file."
                "The input to this tool should be a string, representing the text used to "
                "generate audio."
            ),
        },
        "Style Transfer": {
            "description": (
                "useful for when you want to generate speech samples with styles (e.g., "
                "timbre, emotion, and prosody) derived from a reference custom voice."
                "Like: Generate a speech with style transferred from this voice. The text is "
                "xxx., or speak using the voice of this audio. The text is xxx."
                "The input to this tool should be a comma seperated string of two, "
                "representing reference audio path and input text."
            ),
        },
        "Generate Singing Voice From User Input Text, Note and Duration Sequence": {
            "description": (
                "useful for when you want to generate a piece of singing voice (Optional: "
                "from User Input Text, Note and Duration Sequence) and save it to a file."
                'If Like: Generate a piece of singing voice, the input to this tool should '
                'be "" since there is no User Input Text, Note and Duration Sequence .'
                "If Like: Generate a piece of singing voice. Text: xxx, Note: xxx, "
                "Duration: xxx. "
                "Or Like: Generate a piece of singing voice. Text is xxx, note is xxx, "
                "duration is xxx."
                "The input to this tool should be a comma seperated string of three, "
                "representing text, note and duration sequence since User Input Text, Note "
                "and Duration Sequence are all provided."
            ),
        },
        "Synthesize Speech Given the User Input Text": {
            "description": (
                "useful for when you want to convert a user input text into speech audio it "
                "saved it to a file."
                "The input to this tool should be a string, representing the text used to "
                "be converted to speech."
            ),
        },
        "Speech Enhancement In Single-Channel": {
            "description": (
                "useful for when you want to enhance the quality of the speech signal by "
                "reducing background noise (single-channel), receives audio_path as input."
                "The input to this tool should be a string, representing the audio_path."
            ),
        },
        "Speech Separation In Single-Channel": {
            "description": (
                "useful for when you want to separate each speech from the speech mixture, "
                "receives audio_path as input."
                "The input to this tool should be a string, representing the audio_path."
            ),
        },
        "Generate Audio From The Image": {
            "description": (
                "useful for when you want to generate an audio based on an image."
                "The input to this tool should be a string, representing the image_path. "
            ),
        },
        "Generate Text From The Audio": {
            "description": (
                "useful for when you want to describe an audio in text, receives audio_path "
                "as input."
                "The input to this tool should be a string, representing the audio_path."
            ),
        },
        "Audio Inpainting": {
            "description": (
                "useful for when you want to inpaint a mel spectrum of an audio and predict "
                "this audio, this tool will generate a mel spectrum and you can inpaint it, "
                "receives audio_path as input, "
                "The input to this tool should be a string, representing the audio_path."
            ),
        },
        "Transcribe Speech": {
            "description": (
                "useful for when you want to know the text corresponding to a human speech, "
                "receives audio_path as input."
                "The input to this tool should be a string, representing the audio_path."
            ),
        },
        "Generate a talking human portrait video given a input Audio": {
            "description": (
                "useful for when you want to generate a talking human portrait video given a "
                "input audio."
                "The input to this tool should be a string, representing the audio_path."
            ),
        },
        "Detect The Sound Event From The Audio": {
            "description": (
                "useful for when you want to know what event in the audio and the sound "
                "event start or end time, this tool will generate an image of all predict "
                "events, receives audio_path as input. "
                "The input to this tool should be a string, representing the audio_path. "
            ),
        },
        "Sythesize Binaural Audio From A Mono Audio Input": {
            "description": (
                "useful for when you want to transfer your mono audio into binaural audio, "
                "receives audio_path as input. "
                "The input to this tool should be a string, representing the audio_path. "
            ),
        },
        "Extract Sound Event From Mixture Audio Based On Language Description": {
            "description": (
                "useful for when you extract target sound from a mixture audio, you can "
                "describe the target sound by text, receives audio_path and text as input. "
                "The input to this tool should be a comma seperated string of two, "
                "representing mixture audio path and input text."
            ),
        },
        "Target Sound Detection": {
            "description": (
                "useful for when you want to know when the target sound event in the audio "
                "happens. You can use language descriptions to instruct the model. receives "
                "text description and audio_path as input. "
                "The input to this tool should be a comma seperated string of two, "
                "representing audio path and the text description. "
            ),
        },
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gpt_model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        max_react_iterations: int = 6,
    ):
        try:
            import whisper as _whisper
        except ImportError as e:
            raise ImportError("Install openai-whisper: pip install openai-whisper") from e

        self.name = "audiogpt"
        self.gpt_model = gpt_model
        self.max_retries = max_retries
        self.max_react_iterations = max_react_iterations

        self.openai_api_key = openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided or set via OPENAI_API_KEY env var.")
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        print("[AudioGPT] Loading Whisper (openai/whisper-base)...")
        self._whisper = _whisper.load_model("base")

        print(f"[AudioGPT] Ready. GPT model: {gpt_model}, {len(self._TOOL_REGISTRY)} tools registered.")
    # ----------------------------------------------------------------
    # Tool implementations
    # ----------------------------------------------------------------

    def _run_tool_transcribe_whisper(self, audio_path: str) -> str:
        result = self._whisper.transcribe(str(audio_path))
        return result.get("text", "").strip()

    def _normalize_audio_path(self, tool_input: str, original_audio_path: Optional[str] = None) -> str:
        """Resolve the path GPT provides, correcting the 'audio/' prefix it adds per prompt convention."""
        if Path(tool_input).exists():
            return tool_input
        # GPT prepends 'audio/' because the prompt says files are named 'audio/xxx.wav'
        stripped = tool_input[len("audio/"):] if tool_input.startswith("audio/") else tool_input
        if Path(stripped).exists():
            return stripped
        # Fall back to the actual audio path we were given
        if original_audio_path and Path(original_audio_path).exists():
            return original_audio_path
        return tool_input  # will fail with a clear error downstream

    def _dispatch_tool(self, tool_name: str, tool_input: str, original_audio_path: Optional[str] = None) -> str:
        """
        Execute the named tool and return its observation string.

        Tools backed by local models (Whisper) run directly.
        Tools backed by models from download.sh that are not loaded in this
        environment return an informative observation so the ReAct loop can
        proceed to a final answer without crashing.
        """
        # ---- Locally available tools ----
        if tool_name == "Transcribe Speech":
            resolved = self._normalize_audio_path(tool_input, original_audio_path)
            return self._run_tool_transcribe_whisper(resolved)

        if tool_name == "Generate Text From The Audio":
            # Closest available local model is Whisper; the original uses the
            # AudioCaps audiocaps_cntrstv_cnn14rnn_trm checkpoint from download.sh.
            resolved = self._normalize_audio_path(tool_input, original_audio_path)
            return self._run_tool_transcribe_whisper(resolved)

        # ---- Tools whose checkpoints come from download.sh ----
        _NOT_AVAILABLE: Dict[str, str] = {
            "Generate Image From User Input Text": (
                "Stable Diffusion (T2I) checkpoint is not loaded in this environment."
            ),
            "Get Photo Description": (
                "BLIP image captioning checkpoint is not loaded in this environment."
            ),
            "Generate Audio From User Input Text": (
                "Make-An-Audio (ta40multi) checkpoint is not loaded in this environment."
            ),
            "Style Transfer": (
                "GenerSpeech checkpoint is not loaded in this environment."
            ),
            "Generate Singing Voice From User Input Text, Note and Duration Sequence": (
                "DiffSinger (0831_opencpop_ds1000) checkpoint is not loaded in this environment."
            ),
            "Synthesize Speech Given the User Input Text": (
                "PortaSpeech (ps_adv_baseline) checkpoint is not loaded in this environment."
            ),
            "Speech Enhancement In Single-Channel": (
                "ESPnet Speech_Enh_SS_SC (Wangyou_Zhang_chime4) is not loaded in this environment."
            ),
            "Speech Separation In Single-Channel": (
                "ESPnet Speech_SS (wsj0_2mix_skim_noncausal) is not loaded in this environment."
            ),
            "Generate Audio From The Image": (
                "Make-An-Audio image-to-audio (ta54) checkpoint is not loaded in this environment."
            ),
            "Audio Inpainting": (
                "Make-An-Audio inpainting (inpaint7) checkpoint is not loaded in this environment."
            ),
            "Generate a talking human portrait video given a input Audio": (
                "GeneFace checkpoint is not loaded in this environment."
            ),
            "Detect The Sound Event From The Audio": (
                "PVT sound detection (audio_detection.pth) checkpoint is not loaded in this environment."
            ),
            "Sythesize Binaural Audio From A Mono Audio Input": (
                "Binaural network (m2b/binaural_network.net) checkpoint is not loaded in this environment."
            ),
            "Extract Sound Event From Mixture Audio Based On Language Description": (
                "LASSNet sound extraction (LASSNet.pt) checkpoint is not loaded in this environment."
            ),
            "Target Sound Detection": (
                "Target sound detection (tsd/run_model_7_loss) checkpoint is not loaded in this environment."
            ),
        }
        if tool_name in _NOT_AVAILABLE:
            return _NOT_AVAILABLE[tool_name]

        return f"Tool '{tool_name}' is not recognised."

    # ----------------------------------------------------------------
    # ReAct loop helpers
    # ----------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        tool_list = "\n".join(
            f"> {name}: {spec['description']}"
            for name, spec in self._TOOL_REGISTRY.items()
        )
        tool_names = ", ".join(self._TOOL_REGISTRY.keys())
        return (
            self._PREFIX
            + "\n\n"
            + tool_list
            + self._FORMAT_INSTRUCTIONS.format(
                tool_names=tool_names,
                ai_prefix=self._AI_PREFIX,
            )
            + "\n\n"
            + SYSTEM_INSTRUCTION
        )

    def _retry(self, fn, *args, **kwargs):
        last_e = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_e = e
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * attempt, 8.0))
        raise last_e

    def _call_gpt(
        self,
        messages: List[Dict],
        stop: Optional[List[str]] = None,
        max_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        kwargs: dict = dict(
            model=self.gpt_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        if stop:
            kwargs["stop"] = stop
        resp = self._retry(self.openai_client.chat.completions.create, **kwargs)
        return (resp.choices[0].message.content or "").strip()

    def _parse_action(self, text: str):
        """Return (action_name, action_input) or (None, None) if not parseable."""
        import re
        action_m = re.search(r"Action:\s*(.+)", text)
        input_m = re.search(r"Action Input:\s*(.+)", text)
        if action_m and input_m:
            return action_m.group(1).strip(), input_m.group(1).strip()
        return None, None

    def _extract_final_answer(self, text: str) -> str:
        """Extract the agent's final answer from a completed ReAct response."""
        for marker in (f"{self._AI_PREFIX}:", "AI:"):
            if marker in text:
                return text.split(marker, 1)[-1].strip()
        return text.strip()

    def _run_react_loop(self, user_input: str, chat_history: str = "", original_audio_path: Optional[str] = None) -> str:
        """
        Drive the AudioGPT conversational-react-description loop:

            [Thought → Action → Action Input] ← GPT stops here (stop=["\\nObservation:"])
            Observation: <tool result>         ← we inject this
            [Thought → Action …]               ← repeat
            Thought: No → AudioGPT: <answer>   ← GPT produces final answer

        This exactly mirrors how LangChain's initialize_agent with
        agent="conversational-react-description" executes the AudioGPT flow.
        """
        system_prompt = self._build_system_prompt()
        scratchpad = ""

        for iteration in range(self.max_react_iterations):
            prompt = self._SUFFIX.format(
                chat_history=chat_history,
                input=user_input,
                agent_scratchpad=scratchpad,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Stop before Observation so we can inject the real tool result.
            # The GPT is expected to produce "Yes\nAction: ...\nAction Input: ..."
            # or "No\nAudioGPT: ..." before hitting the stop token.
            response = self._call_gpt(messages, stop=["\nObservation:"])
            scratchpad += response

            # Final-answer branch: LLM decided not to use a tool.
            # GPT primed with "Do I need to use a tool? " may respond just "No"
            # (without the full prefix), so check for that bare word too.
            _resp_bare = response.strip().lower().rstrip(".")
            if (
                "Do I need to use a tool? No" in response
                or f"{self._AI_PREFIX}:" in response
                or "AI:" in response
                or _resp_bare == "no"
            ):
                answer = self._extract_final_answer(response)
                # GPT said "No" but didn't produce the actual answer yet.
                # Append the AI prefix to the scratchpad and prompt once more.
                if answer.strip().lower().rstrip(".") in ("no", ""):
                    scratchpad += f"\n{self._AI_PREFIX}:"
                    followup_prompt = self._SUFFIX.format(
                        chat_history=chat_history,
                        input=user_input,
                        agent_scratchpad=scratchpad,
                    )
                    followup_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": followup_prompt},
                    ]
                    answer = self._call_gpt(followup_messages)
                return answer

            # Action branch: parse and execute the tool
            action_name, action_input = self._parse_action(response)
            if action_name is None:
                # Unparseable output — treat whatever we have as the final answer
                return self._extract_final_answer(response)

            # Only audio/transcription tools are valid in this environment.
            _AUDIO_TOOLS = {"Transcribe Speech", "Generate Text From The Audio"}
            if action_name not in _AUDIO_TOOLS:
                return "Wrong action selected"

            try:
                observation = self._dispatch_tool(action_name, action_input, original_audio_path)
            except Exception as exc:
                observation = f"Error running tool: {exc}"

            # After transcription, immediately ask GPT to answer the spoken question.
            if action_name in ("Transcribe Speech", "Generate Text From The Audio"):
                answer_messages = [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": observation},
                ]
                return self._call_gpt(answer_messages, max_tokens=MAX_NEW_TOKENS)

            # Append observation and prime the next Thought exactly as LangChain does
            scratchpad += f"\nObservation: {observation}\nThought: Do I need to use a tool? "

        # Exceeded max iterations — return whatever the scratchpad contains
        return self._extract_final_answer(scratchpad)

    # ----------------------------------------------------------------
    # BaseAudioModel interface
    # ----------------------------------------------------------------

    def generate(
        self,
        audio_path: Optional[str] = None,
        text_prompt: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Run the AudioGPT ReAct loop.

        The agent will decide on its own whether to call "Transcribe Speech" or
        "Recognize Speech (SpeechT5)" based on the AudioGPT FORMAT_INSTRUCTIONS,
        then synthesise a final answer — exactly as the original LangChain agent would.
        """
        if audio_path is not None:
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            user_input = (
                f"Read the audio file at {audio_path}. "
            )
            if text_prompt:
                user_input = f"{user_input} {text_prompt}"
        else:
            user_input = text_prompt or ""

        if not user_input:
            return ""

        try:
            return self._run_react_loop(user_input, original_audio_path=audio_path)
        except Exception as e:
            print(f"[AudioGPT] Error during generate: {e}")
            return ""

    def transcribe(self, audio_path: str) -> str:
        """Direct transcription via Whisper (bypasses the ReAct loop)."""
        try:
            return self._run_tool_transcribe_whisper(audio_path)
        except Exception as e:
            print(f"[AudioGPT] Error during transcribe: {e}")
            return ""

    def infer_speaker(
        self,
        audio_path: str,
        text_prompt: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> dict:
        """
        Transcribe the audio with Whisper, then ask GPT-4 to infer speaker
        characteristics (gender, age_group, emotion) from the transcript.
        """
        try:
            transcript = self._run_tool_transcribe_whisper(audio_path)
            messages = [
                {"role": "system", "content": INFER_SPEAKER_ID},
                {"role": "user", "content": f'Audio transcription: "{transcript}"'},
            ]
            response = self._call_gpt(messages)

            import ast as _ast
            for parser in (json.loads, _ast.literal_eval):
                try:
                    parsed = parser(response)
                    return {
                        "gender": parsed.get("gender", "unknown"),
                        "age_group": parsed.get("age_group", "unknown"),
                        "emotion": parsed.get("emotion", "unknown"),
                    }
                except Exception:
                    continue

            return {"gender": None, "age_group": None, "emotion": None, "raw_output": response}

        except Exception as e:
            print(f"[AudioGPT] Error during infer_speaker: {e}")
            return {"gender": None, "age_group": None, "emotion": None, "raw_output": f"Error: {e}"}

    def generate_multiturn(self, audio_root: Path, chat: list, system_instruction=None) -> str:
        """
        Multi-turn AudioGPT flow.

        All user audio turns are transcribed locally with Whisper up-front.
        The full conversation (with transcripts) is then sent to GPT in one call
        so it can answer the final user question with proper context — mirroring
        the improvement made to generate() where GPT is queried directly with
        the transcript rather than echoing it back.
        """
        messages: List[Dict] = [
            {"role": "system", "content": SYSTEM_INSTRUCTION}
        ]
        last_user_content: Optional[str] = None

        for turn in chat:
            role = turn.get("role", "")
            text = turn.get("text", "") or ""
            audio_file = turn.get("path") or turn.get("audio_path")

            if role == "user":
                if audio_file:
                    ap = Path(os.path.join(audio_root, audio_file))
                    if not ap.exists():
                        raise FileNotFoundError(f"Audio file not found: {ap}")
                    transcript = self._run_tool_transcribe_whisper(str(ap))
                    content = f"{transcript} {text}".strip()
                else:
                    content = text
                messages.append({"role": "user", "content": content})
                last_user_content = content
            else:
                messages.append({"role": "assistant", "content": text})

        if last_user_content is None:
            return ""

        try:
            return self._call_gpt(messages, max_tokens=MAX_NEW_TOKENS)
        except Exception as e:
            print(f"[AudioGPT] Error during generate_multiturn: {e}")
            return ""
    

class HuggingGPTAudioModel(BaseAudioModel):
    """
    HuggingGPT (JARVIS) audio model — full four-stage pipeline with local ASR.

    Workflow:
      Stage 1 – Task Planning   : GPT-4 parses the request into a structured task list.
      Stage 2 – Model Selection : GPT-4 selects between two locally-loaded ASR models.
      Stage 3 – Task Execution  : Runs the selected local model (Whisper or SpeechT5).
      Stage 4 – Response Gen    : GPT-4 synthesises the final answer from execution logs.

    Local ASR models (loaded at init):
      • openai/whisper-base      (via openai-whisper)
      • microsoft/speecht5_asr   (via transformers pipeline)

    Reference: https://github.com/AI-Chef/HuggingGPT
    Paper: https://arxiv.org/abs/2303.17580

    Installation:
        pip install openai openai-whisper transformers datasets soundfile
    """

    ASR_CANDIDATES = [
        {
            "id": "openai/whisper-base",
            "description": (
                "Whisper is a pre-trained model for automatic speech recognition (ASR) and "
                "speech translation. Achieves strong zero-shot performance across many languages "
                "and accents. Best for multilingual, accented, or noisy speech."
            ),
        },
        {
            "id": "microsoft/speecht5_asr",
            "description": (
                "SpeechT5 model fine-tuned on LibriSpeech for automatic speech recognition. "
                "Optimised for clean, standard English speech. Faster on short clips."
            ),
        },
    ]

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        gpt_model: str = "gpt-4",
        max_retries: int = 3,
    ):
        try:
            import whisper as _whisper
        except ImportError as e:
            raise ImportError("Install openai-whisper: pip install openai-whisper") from e

        self.name = "hugginggpt"
        self.gpt_model = gpt_model
        self.max_retries = max_retries

        self.openai_api_key = openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key must be provided as parameter or via OPENAI_API_KEY env var."
            )

        self.hf_token = hf_token if hf_token is not None else os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # ---- Load local ASR models ----
        print("[HuggingGPT] Loading local Whisper (openai/whisper-base)...")
        self._whisper = _whisper.load_model("base")

        print("[HuggingGPT] Loading local SpeechT5 (microsoft/speecht5_asr)...")
        from transformers import pipeline as hf_pipeline
        self._speecht5_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model="microsoft/speecht5_asr",
            device=0 if (torch.cuda.is_available() and not USE_CPU_ONLY) else -1,
        )

        print(f"[HuggingGPT] Ready. GPT model: {gpt_model}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retry(self, fn, *args, **kwargs):
        last_e = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_e = e
                if attempt < self.max_retries:
                    wait = min(2.0 * attempt, 8.0)
                    time.sleep(wait)
        raise last_e

    def _call_gpt(self, messages: List[Dict], max_tokens: int = MAX_NEW_TOKENS) -> str:
        resp = self._retry(
            self.openai_client.chat.completions.create,
            model=self.gpt_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    def _validate_audio(self, audio_path: str) -> None:
        ap = Path(audio_path)
        if not ap.exists():
            raise FileNotFoundError(f"Audio file not found: {ap}")
        wav, sr = librosa.load(str(ap), sr=None)
        if wav.size == 0:
            raise ValueError(f"Empty audio file: {ap}")
        dur = len(wav) / sr
        if dur < MIN_AUDIO_SEC:
            raise ValueError(f"Audio too short ({dur:.3f}s < {MIN_AUDIO_SEC}s): {ap}")

    # ------------------------------------------------------------------
    # Stage 1 – Task Planning
    # ------------------------------------------------------------------

    def _plan_tasks(self, user_input: str, context: str = "") -> list:
        """Stage 1: GPT-4 decomposes the user request into a structured task list."""
        system_prompt = (
            "#1 Task Planning Stage: The AI assistant can parse user input to several tasks: "
            '[{"task": task, "id": task_id, "dep": dependency_task_id, "args": {"text": text or '
            '"<GENERATED>-dep_id", "image": image_url or "<GENERATED>-dep_id", "audio": audio_url or "<GENERATED>-dep_id"}}]. '
            'The special tag "<GENERATED>-dep_id" refers to the generated text/image/audio in the '
            "dependency task and dep_id must be in the dep list. The dep field denotes ids of "
            "prerequisite tasks which generate a new resource the current task relies on. "
            'The args field must be in ["text", "image", "audio"], nothing else. '
            "The task MUST be selected from: "
            '"token-classification", "text2text-generation", "summarization", "translation", '
            '"question-answering", "conversational", "text-generation", "sentence-similarity", '
            '"tabular-classification", "object-detection", "image-classification", '
            '"image-to-image", "image-to-text", "text-to-image", "text-to-video", '
            '"visual-question-answering", "document-question-answering", "image-segmentation", '
            '"depth-estimation", "text-to-speech", "automatic-speech-recognition", '
            '"audio-to-audio", "audio-classification". '
            "Parse as few tasks as possible while resolving the request. "
            "If the input cannot be parsed, reply with empty JSON []."
        )
        user_prompt = (
            f"The chat log [ {context} ] may contain the resources I mentioned. "
            f"Now I input {{ {user_input} }}. "
            "Pay attention to the input and output types of tasks and the dependencies between tasks."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self._call_gpt(messages, max_tokens=512)
        start, end = raw.find("["), raw.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return []

    # ------------------------------------------------------------------
    # Stage 2 – Model Selection
    # ------------------------------------------------------------------

    def _select_model(self, task: dict) -> str:
        """Stage 2: GPT-4 selects the best local ASR model for the given task."""
        system_prompt = (
            "#2 Model Selection Stage: Given the user request and the parsed tasks, the AI "
            "assistant selects the most suitable model from a list of models. Focus on the model "
            "description to find the one with the most potential to solve the task. Prefer models "
            "that match the audio characteristics (language, accent, noise level)."
        )
        user_prompt = (
            f"Please choose the most suitable model from {json.dumps(self.ASR_CANDIDATES)} "
            f"for the task {json.dumps(task)}. "
            'The output must be in a strict JSON format: {"id": "id", "reason": "your detail reasons for the choice"}.'
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self._call_gpt(messages, max_tokens=256)
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end])
                chosen_id = parsed.get("id", "")
                valid_ids = {c["id"] for c in self.ASR_CANDIDATES}
                if chosen_id in valid_ids:
                    return chosen_id
            except json.JSONDecodeError:
                pass
        # Default fallback
        return "openai/whisper-base"

    # ------------------------------------------------------------------
    # Stage 3 – Task Execution (local models)
    # ------------------------------------------------------------------

    def _run_asr(self, model_id: str, audio_path: str) -> str:
        """Stage 3: Run the locally-loaded ASR model selected in Stage 2."""
        if model_id == "openai/whisper-base":
            result = self._whisper.transcribe(str(audio_path))
            return result.get("text", "").strip()
        elif model_id == "microsoft/speecht5_asr":
            result = self._speecht5_pipeline(str(audio_path))
            if isinstance(result, dict):
                return result.get("text", "").strip()
            if isinstance(result, list) and result:
                return result[0].get("text", "").strip()
            return str(result).strip()
        else:
            raise ValueError(f"Unknown local ASR model: {model_id}")

    # ------------------------------------------------------------------
    # Stage 4 – Response Generation
    # ------------------------------------------------------------------

    def _generate_response(self, user_input: str, execution_logs: list) -> str:
        """Stage 4: GPT-4 synthesises the final answer from execution logs."""
        system_prompt = (
            "#4 Response Generation Stage: With the task execution logs, the AI assistant "
            "needs to describe the process and inference results."
        )
        user_content = (
            f"User request: {user_input}\n\n"
            f"Execution logs: {json.dumps(execution_logs, ensure_ascii=False)}"
        )
        final_instruction = (
            "Yes. Please first think carefully and directly answer my request based on the "
            "inference results. Some inferences may not always be correct — apply careful "
            "consideration. Then detail your workflow including the used models and inference "
            "results in a friendly tone. Filter out information not relevant to my request. "
            "If there is nothing useful in the results, tell me you could not complete the task."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "user", "content": final_instruction},
        ]
        return self._call_gpt(messages, max_tokens=MAX_NEW_TOKENS)

    # ------------------------------------------------------------------
    # BaseAudioModel interface
    # ------------------------------------------------------------------

    def generate(
        self,
        audio_path: Optional[str] = None,
        text_prompt: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Full four-stage HuggingGPT pipeline.

        Stage 1: GPT-4 identifies an automatic-speech-recognition task from the input.
        Stage 2: GPT-4 picks openai/whisper-base or microsoft/speecht5_asr.
        Stage 3: Selected local model transcribes the audio.
        Stage 4: GPT-4 generates the final response from the transcription logs.

        Args:
            audio_path: Path to audio file (optional).
            text_prompt: Additional text query (optional).
            system_instruction: Ignored; HuggingGPT uses its own stage prompts.
        """
        if audio_path is not None:
            self._validate_audio(audio_path)
            user_input = f"Audio file: {audio_path}"
            if text_prompt:
                user_input = f"{user_input}. {text_prompt}"
        else:
            user_input = text_prompt or ""

        if not user_input:
            return ""

        try:
            # Stage 1 – Task Planning: identify whether ASR is needed
            tasks = self._plan_tasks(user_input)

            # Resolve transcript: if any task is ASR and we have audio, run it locally
            transcript: Optional[str] = None
            if audio_path is not None:
                asr_tasks = [t for t in tasks if t.get("task") == "automatic-speech-recognition"]
                asr_task = asr_tasks[0] if asr_tasks else {
                    "task": "automatic-speech-recognition", "id": 0, "dep": [-1], "args": {}
                }
                # Stage 2 – Model Selection
                model_id = self._select_model(asr_task)
                # Stage 3 – Task Execution (local ASR)
                transcript = self._run_asr(model_id, audio_path)

            # Stage 4 – Direct answer via GPT-4
            # Build a clean message: transcript (if any) + optional text prompt → GPT answers
            if transcript is not None:
                question = transcript
                if text_prompt:
                    question = f"{question}\n\n{text_prompt}"
            else:
                question = text_prompt or ""

            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": question},
            ]
            return self._call_gpt(messages)

        except Exception as e:
            print(f"[HuggingGPT] Error during generate: {e}")
            return ""

    def transcribe(self, audio_path: str, text_prompt: Optional[str] = None) -> str:
        """
        Transcribe audio using Stage 2 (GPT-4 model selection) +
        Stage 3 (local ASR execution).

        Args:
            audio_path: Path to audio file.
            text_prompt: Ignored (kept for interface consistency).
        """
        try:
            self._validate_audio(audio_path)
            asr_task = {
                "task": "automatic-speech-recognition",
                "id": 0,
                "dep": [-1],
                "args": {"audio": audio_path},
            }
            model_id = self._select_model(asr_task)
            return self._run_asr(model_id, audio_path)
        except Exception as e:
            print(f"[HuggingGPT] Error during transcribe: {e}")
            return ""

    def infer_speaker(
        self,
        audio_path: str,
        text_prompt: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> dict:
        """
        Infer speaker characteristics.

        Stages 2+3 transcribe the audio with the GPT-selected local model.
        Stage 4 asks GPT-4 to infer gender, age_group, and emotion from the transcript.

        Args:
            audio_path: Path to audio file.
            text_prompt: Speaker-profiling prompt (defaults to INFER_SPEAKER_ID).
            system_instruction: Ignored; HuggingGPT uses its own stage prompts.
        """
        prompt = text_prompt if text_prompt is not None else INFER_SPEAKER_ID

        try:
            self._validate_audio(audio_path)
            asr_task = {
                "task": "automatic-speech-recognition",
                "id": 0,
                "dep": [-1],
                "args": {"audio": audio_path},
            }
            model_id = self._select_model(asr_task)
            transcript = self._run_asr(model_id, audio_path)

            messages = [
                {"role": "system", "content": INFER_SPEAKER_ID},
                {"role": "user", "content": f'Audio transcription: "{transcript}"'},
            ]
            response = self._call_gpt(messages)

            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(response)
                    return {
                        "gender": parsed.get("gender", "unknown"),
                        "age_group": parsed.get("age_group", "unknown"),
                        "emotion": parsed.get("emotion", "unknown"),
                    }
                except Exception:
                    continue

            return {"gender": None, "age_group": None, "emotion": None, "raw_output": response}

        except Exception as e:
            print(f"[HuggingGPT] Error during infer_speaker: {e}")
            return {
                "gender": None,
                "age_group": None,
                "emotion": None,
                "raw_output": f"Error: {str(e)}",
            }

    def generate_multiturn(self, audio_root: Path, chat: list, system_instruction=None) -> str:
        """
        Multi-turn pipeline.

        GPT-4 selects the local ASR model once (Stage 2) then transcribes each
        user audio turn locally (Stage 3). The full conversation history is
        forwarded to GPT-4 for the final response.

        Args:
            audio_root: Root directory for resolving relative audio paths.
            chat: List of turn dicts with "role", "text", "path"/"audio_path".
            system_instruction: Optional system prompt override.
        """
        # Stage 2 – select ASR model once for all turns
        asr_task = {"task": "automatic-speech-recognition", "id": 0, "dep": [-1], "args": {}}
        model_id = self._select_model(asr_task)

        conversation: List[Dict] = [
            {"role": "system", "content": system_instruction or SYSTEM_INSTRUCTION}
        ]

        for turn in chat:
            role = turn["role"]
            text = turn.get("text", "") or ""
            audio_file = turn.get("path") or turn.get("audio_path")

            if role == "user" and audio_file:
                ap = Path(os.path.join(audio_root, audio_file))
                if not ap.exists():
                    raise FileNotFoundError(f"Audio file not found: {ap}")
                # Stage 3 – local transcription
                transcript = self._run_asr(model_id, str(ap))
                content = f"(Transcribed): {transcript}"
                if text:
                    content = f"{content}\n{text}"
                conversation.append({"role": "user", "content": content})
            else:
                conversation.append({"role": role, "content": text})

        try:
            resp = self._retry(
                self.openai_client.chat.completions.create,
                model=self.gpt_model,
                messages=conversation,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[HuggingGPT] Error during generate_multiturn: {e}")
            return ""

# ----------------------------
# SALMONN
# ----------------------------

# class SALMONNModel(BaseAudioModel):
#     def __init__(self):
#         from transformers import AutoProcessor, AutoModelForSeq2SeqLM

#         self.name = "salmonn"
#         if USE_CPU_ONLY:
#             device_map = {"": "cpu"}
#             dtype = torch.float32
#         else:
#             device_map = "auto"
#             dtype = torch.float16 if DEVICE == "cuda" else torch.float32

#         self.processor = AutoProcessor.from_pretrained("SALMONN/salmonn-audio", trust_remote_code=True)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(
#             "SALMONN/salmonn-audio", torch_dtype=dtype, device_map=device_map, trust_remote_code=True
#         )
#         self.model.eval()
#         print(f"Model loaded on: {self.model.device}")

#     def generate(self, audio_path=None, text_prompt: Optional[str] = None, system_instruction=None):
#         if system_instruction is None:
#             system_instruction = SYSTEM_INSTRUCTION

#         if audio_path is None:
#             # Text-only mode
#             inputs = self.processor(text=text_prompt or "", return_tensors="pt")
#         else:
#             wav, sr = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
#             if len(wav) == 0:
#                 raise ValueError(f"Empty audio loaded: {audio_path}")
#             if len(wav) < sr * MIN_AUDIO_SEC:
#                 raise ValueError(f"Audio too short for SALMONN ({len(wav)/sr:.3f}s) at {audio_path}")

#             inputs = self.processor(
#                 text=text_prompt or "",
#                 audio=wav,
#                 sampling_rate=sr,
#                 return_tensors="pt"
#             )

#         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             output_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

#         response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#         return response  

    
# class StepAudioChatModel(BaseAudioModel):
#     def __init__(self):
#         from transformers import AutoProcessor, AutoModelForSeq2SeqLM

#         self.name = "step-audio-chat"

#         if USE_CPU_ONLY:
#             device_map = {"": "cpu"}
#             dtype = torch.float32
#         else:
#             device_map = "auto"
#             dtype = torch.float16 if DEVICE == "cuda" else torch.float32

#         # Load processor and model from Hugging Face
#         self.processor = AutoProcessor.from_pretrained(
#             "stepfun-ai/Step-Audio-Chat",
#             trust_remote_code=True
#         )
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(
#             "stepfun-ai/Step-Audio-Chat",
#             torch_dtype=dtype,
#             device_map=device_map,
#             trust_remote_code=True
#         )
#         self.model.eval()
#         print(f"Step-Audio-Chat model loaded on: {self.model.device}")

#     def generate(self, audio_path=None, text_prompt: Optional[str] = None, system_instruction=None):
#         if system_instruction is None:
#             system_instruction = SYSTEM_INSTRUCTION

#         if audio_path is None:
#             # Text-only mode (optional)
#             inputs = self.processor(text=text_prompt or "", return_tensors="pt")
#         else:
#             wav, sr = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
#             if len(wav) == 0:
#                 raise ValueError(f"Empty audio loaded from {audio_path}")
#             if len(wav) < sr * MIN_AUDIO_SEC:
#                 raise ValueError(f"Audio too short for Step-Audio-Chat ({len(wav)/sr:.3f}s) at {audio_path}")

#             # Prepare audio + optional text prompt for generation
#             conversation = [
#                 {"role": "system", "content": system_instruction},
#                 {"role": "user", "content": [{"type": "audio", "audio_url": str(audio_path)}]},
#             ]

#             if text_prompt:
#                 conversation[1]["content"].append({"type": "text", "text": text_prompt})

#             # Apply chat template
#             text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
#             inputs = self.processor(text=text, audios=[wav], sampling_rate=sr, return_tensors="pt")

#         # Move tensors to model device
#         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

#         # Generate response
#         with torch.no_grad():
#             output_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

#         response = self.processor.batch_decode(
#             output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )[0].strip()


# class MERaLiONAudioModel(BaseAudioModel):
#     """
#     MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION
#     - Uses ONLY Script-1 SYSTEM_INSTRUCTION
#     - No transcription / translation logic
#     - Compatible with FSS evaluation pipeline
#     """

#     def __init__(self, repo_id: str = "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION"):
#         from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

#         self.name = "meralion-audiollm"
#         self.repo_id = repo_id

#         if USE_CPU_ONLY:
#             self.device = torch.device("cpu")
#             dtype = torch.float32
#         else:
#             self.device = torch.device(DEVICE)
#             dtype = torch.float16 if self.device.type == "cuda" else torch.float32

#         self.processor = AutoProcessor.from_pretrained(
#             self.repo_id,
#             trust_remote_code=True,
#             use_fast=False,
#         )

#         self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             self.repo_id,
#             trust_remote_code=True,
#             use_safetensors=True,
#             torch_dtype=dtype,
#         ).to(self.device)

#         self.model.eval()

#         # Model card recommends <= 30s, 16kHz
#         self.target_sr = 16000
#         self.max_audio_sec = 30.0

#         print(f"[MERaLiON] Loaded on {self.device} (dtype={dtype})")

#     def _build_prompt(self, user_prompt: str) -> str:
#         # Inject ONLY Script-1 system instruction, then user prompt (no task injection).
#         content = f"{SYSTEM_INSTRUCTION}\n\n{user_prompt}"

#         conversation = [[{"role": "user", "content": content}]]

#         return self.processor.tokenizer.apply_chat_template(
#             conversation=conversation,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#     def _load_audio(self, audio_path: str):
#         wav, sr = librosa.load(audio_path, sr=self.target_sr)

#         if wav.size == 0:
#             raise ValueError(f"Empty audio: {audio_path}")

#         if len(wav) < int(self.target_sr * MIN_AUDIO_SEC):
#             raise ValueError(
#                 f"Audio too short ({len(wav)/self.target_sr:.3f}s < {MIN_AUDIO_SEC}s)"
#             )

#         max_len = int(self.target_sr * self.max_audio_sec)
#         if len(wav) > max_len:
#             wav = wav[:max_len]

#         return wav, self.target_sr

#     def generate(self, audio_path: Optional[str], text_prompt: Optional[str] = None) -> str:
#         # Text-only path: MERaLiON is audio-first; keep pipeline stable.
#         if audio_path is None:
#             return ""

#         prompt = text_prompt or ""
#         chat_prompt = self._build_prompt(prompt)

#         wav, sr = self._load_audio(str(audio_path))

#         inputs = self.processor(
#             text=chat_prompt,
#             audios=[wav],
#             sampling_rate=sr,
#             return_tensors="pt",
#             padding=True,
#         )

#         inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

#         input_len = inputs["input_ids"].size(1)
#         gen_ids = outputs[:, input_len:]

#         resp = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
#         return (resp[0] if resp else "").strip()


# # --------------------
# # Closed-source
# # --------------------

# class GPT4oMiniAudioModel(BaseAudioModel):
#     def __init__(self, api_key: str):
#         self.name = "gpt-4o-mini"
#         self.model_name = "gpt-4o-mini-audio-preview"
#         self.client = OpenAI(api_key=api_key)

#     def _encode_audio(self, audio_path):
#         with open(audio_path, "rb") as f:
#             return base64.b64encode(f.read()).decode("utf-8")

#     def generate(self, audio_path=None, text_prompt: Optional[str] = None, system_instruction=None):
#         # if audio_path is None:
#         #     raise ValueError("GPT-4o Mini Audio requires audio input.")

#         if system_instruction is None:
#             system_instruction = SYSTEM_INSTRUCTION

#         if audio_path is None:
#             t0 = time.time()
#             completion = self.client.chat.completions.create(
#                 model="gpt-4o-mini-audio-preview",
#                 modalities=["text"],
#                 messages=[
#                     {"role": "user", "content": text_prompt}
#                 ]
#             )
#         else:
#             encoded_audio = self._encode_audio(audio_path)
#             messages = [
#                 {
#                     "role": "system",
#                     "content": system_instruction
#                 },
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "input_audio",
#                             "input_audio": {
#                                 "data": encoded_audio,
#                                 "format": "wav"
#                             }
#                         }
#                     ]
#                 }
#             ]


#         completion = self.client.chat.completions.create(
#             model= self.model_name,
#             modalities=[ "text"],
#             messages=messages
#         )

#         output_text = completion.choices[0].message.content
#         return output_text


# class GPT4oAudioPreviewModel(BaseAudioModel):
#     """
#     GPT-4o Audio model (native audio input, no ASR preprocessing).
#     Compatible with FSS evaluation pipeline.
#     """

#     def __init__(
#         self,
#         api_key: str,
#         model_name: str = "gpt-4o-audio-preview",
#         max_retries: int = 3,
#     ):
#         self.name = self.name = "gpt-4o-mini-audio"
#         self.model_name = model_name
#         self.max_retries = max_retries

#         self.client = OpenAI(api_key=api_key)

#     def _retry(self, fn, *args, **kwargs):
#         last_e = None
#         for attempt in range(1, self.max_retries + 1):
#             try:
#                 return fn(*args, **kwargs)
#             except Exception as e:
#                 last_e = e
#                 time.sleep(min(2 * attempt, 8))
#         raise last_e

#     def _load_audio_base64(self, audio_path: str) -> str:
#         ap = Path(audio_path)
#         if not ap.exists():
#             raise FileNotFoundError(ap)

#         with open(ap, "rb") as f:
            
#             return base64.b64encode(f.read()).decode("utf-8")

#     def generate(
#         self,
#         audio_path: Optional[str],
#         text_prompt: Optional[str] = None,
#     ) -> str:

#         content = []

#         if audio_path is None:
#             content.append({
#                 "type": "input_text",
#                 "text": text_prompt,
#             })

#         else:
#             audio_b64 = self._load_audio_base64(audio_path)
#             ap = Path(audio_path)
#             if not ap.exists():
#                 raise FileNotFoundError(ap)
        

#             with open(ap, "rb") as f:
#                 base64.b64encode(f.read()).decode("utf-8")
            
#             content.append({
#                 "type": "input_audio",
#                 "input_audio": {
#                     "data": audio_b64,
#                     "format": "wav",
#                 },
#             })

#         if not content:
#             raise ValueError("Must provide audio_path or text_prompt")

#         response = self._retry(
#             self.client.responses.create,
#             model=self.model_name,
#             input=[{
#                 "role": "user",
#                 "content": content,
#             }],
#         )

#         return (response.output_text or "").strip()



# ---------- Closed-source: Gemini 2.5 ----------
class GeminiAudioModel(BaseAudioModel):
    """
    Google Gemini Audio Model with native audio understanding capabilities.
    Uses the google.genai SDK for audio and text processing.
    
    Reference: https://ai.google.dev/gemini-api/docs
    
    Installation:
        pip install google-genai
    
    Features:
    - Native audio understanding (no ASR preprocessing)
    - Text and audio inputs
    - Supports various Gemini models (gemini-2.0-flash, etc.)
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3-flash-preview", max_retries: int = 3):        
        self.name = "gemini3-audio"
        self.model_name = model_name
        self.max_retries = max_retries
        
        # Get API key from parameter or environment
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key must be provided either as parameter or via GEMINI_API_KEY environment variable.\n"
            )
        
        # Initialize client
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=self.api_key)
        
        print(f"Initialized Gemini Audio Model ({model_name})")

    def _retry(self, fn, *args, **kwargs):
        """Retry logic for API calls with exponential backoff."""
        last_e = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_e = e
                if attempt < self.max_retries:
                    wait_time = min(2.0 * attempt, 8.0)
                    print(f"API call failed (attempt {attempt}/{self.max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
        raise last_e

    def generate(self, audio_path: Optional[str] = None, text_prompt: Optional[str] = None, system_instruction: Optional[str] = None) -> str:
        """
        Generate response from Gemini model.
        
        Args:
            audio_path: Path to audio file (optional)
            text_prompt: Text prompt (optional)
            system_instruction: System instruction (optional, defaults to SYSTEM_INSTRUCTION)
            
        Returns:
            Generated text response
        """
        if system_instruction is None:
            system_instruction = SYSTEM_INSTRUCTION
        
        # TEXT-ONLY MODE
        if audio_path is None:
            prompt = text_prompt or ""
            config = self.types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
            resp = self._retry(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return (resp.text or "").strip()

        # AUDIO MODE
        ap = Path(str(audio_path))
        if not ap.exists():
            raise FileNotFoundError(f"Audio file not found: {ap}")

        # Validate audio duration
        try:
            wav, sr = librosa.load(str(ap), sr=None)
            if wav.size == 0:
                raise ValueError(f"Empty audio loaded from {ap}")
            dur = len(wav) / sr
            if dur < MIN_AUDIO_SEC:
                raise ValueError(f"Audio too short ({dur:.3f}s < {MIN_AUDIO_SEC}s) at {ap}")
            if dur > MAX_AUDIO_SEC:
                print(f"Warning: Audio duration ({dur:.1f}s) exceeds {MAX_AUDIO_SEC}s, may affect performance")
        except Exception as e:
            # If librosa fails on some codecs, still let Gemini try to upload it
            print(f"Warning: Could not validate audio: {e}")

        # Upload file - try different API methods
        uploaded = None
        upload_errors = []
        
        # Method 1: Try using genai.upload_file() standalone function
        try:
            uploaded = self._retry(self.genai.upload_file, str(ap))
            print(f"[Gemini] Successfully uploaded file using genai.upload_file()")
        except (AttributeError, TypeError, Exception) as e1:
            upload_errors.append(f"Method 1 (genai.upload_file): {e1}")
            
            # Method 2: Try client.files.upload with file parameter
            try:
                uploaded = self._retry(self.client.files.upload, file=str(ap))
                print(f"[Gemini] Successfully uploaded file using client.files.upload(file=...)")
            except (AttributeError, TypeError, Exception) as e2:
                upload_errors.append(f"Method 2 (client.files.upload with file=): {e2}")
                
                # Method 3: Try reading file and passing content
                try:
                    with open(str(ap), 'rb') as f:
                        file_content = f.read()
                    
                    # Determine MIME type from file extension
                    ext = ap.suffix.lower()
                    mime_type = {
                        '.wav': 'audio/wav',
                        '.mp3': 'audio/mpeg',
                        '.flac': 'audio/flac',
                        '.ogg': 'audio/ogg',
                        '.m4a': 'audio/mp4'
                    }.get(ext, 'audio/wav')
                    
                    uploaded = self._retry(
                        self.client.files.upload,
                        file=file_content,
                        mime_type=mime_type
                    )
                    print(f"[Gemini] Successfully uploaded file using client.files.upload(file=content)")
                except Exception as e3:
                    upload_errors.append(f"Method 3 (client.files.upload with content): {e3}")
        
        # Check if any method succeeded
        if uploaded is None:
            raise RuntimeError(
                f"Failed to upload audio file to Gemini. Tried {len(upload_errors)} methods:\n" +
                "\n".join(f"  {err}" for err in upload_errors) +
                "\n\nPlease update google-genai: pip install --upgrade google-genai"
            )
        
        # Prepare contents
        contents = [uploaded]
        if text_prompt:
            contents.append(text_prompt)
        
        # Generate content
        config = self.types.GenerateContentConfig(
            system_instruction=system_instruction,
        )
        
        resp = self._retry(
            self.client.models.generate_content,
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return (resp.text or "").strip()
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio using Gemini.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        return self.generate(
            audio_path=audio_path,
            text_prompt=ASR_PROMPT,
            system_instruction="You are a speech recognition system."
        )
    
    def infer_speaker(self, audio_path: str, text_prompt: Optional[str] = None, system_instruction: Optional[str] = None) -> dict:
        """
        Infer speaker characteristics from audio.
        
        Args:
            audio_path: Path to audio file
            text_prompt: Custom prompt (optional, defaults to INFER_SPEAKER_ID)
            system_instruction: System instruction (optional)
            
        Returns:
            Dictionary with gender, age_group, and emotion
        """
        if text_prompt is None:
            text_prompt = INFER_SPEAKER_ID
        
        if system_instruction is None:
            system_instruction = "You are an expert in speaker profiling."
        
        response = self.generate(
            audio_path=audio_path,
            text_prompt=text_prompt,
            system_instruction=system_instruction
        )
        
        # Try to parse JSON from response
        try:
            parsed = json.loads(response)
            return {
                "gender": parsed.get("gender", "unknown"),
                "age_group": parsed.get("age_group", "unknown"),
                "emotion": parsed.get("emotion", "unknown")
            }
        except json.JSONDecodeError:
            # Try ast.literal_eval as fallback
            try:
                parsed = ast.literal_eval(response)
                return {
                    "gender": parsed.get("gender", "unknown"),
                    "age_group": parsed.get("age_group", "unknown"),
                    "emotion": parsed.get("emotion", "unknown")
                }
            except:
                return {
                    "gender": None,
                    "age_group": None,
                    "emotion": None,
                    "raw_output": response
                }

    def generate_multiturn(self, audio_root: Path, chat: list, system_instruction=None) -> str:
        """
        Generate a response for a multi-turn conversation.

        User turns are uploaded to Gemini as audio files; assistant turns are
        passed as text strings so the model has accurate conversation context.

        Args:
            audio_root: Root directory that the relative "path" fields in chat
                        are resolved against.
            chat: The dataset's generated["turns"] list. Each element must have
                  "role" ("user"/"assistant"), "text", and "path" for user turns.
            system_instruction: Optional system prompt override.

        Returns:
            Model response string for the final user turn.

        Example:
            response = model.generate_multiturn(
                audio_root=Path("Fairness_TTS_subset"),
                chat=record["generated"]["turns"],
            )
        """
        if system_instruction is None:
            system_instruction = SYSTEM_INSTRUCTION

        # Gemini uses a flat contents list: interleave uploaded audio + text
        contents = []

        for turn in chat:
            role = turn["role"]
            text = turn.get("text", "") or ""
            audio_file = turn.get("path") or turn.get("audio_path")

            if role == "user" and audio_file:
                ap = Path(os.path.join(audio_root, audio_file))
                if not ap.exists():
                    raise FileNotFoundError(f"Audio file not found: {ap}")

                # Validate duration
                try:
                    wav, sr = librosa.load(str(ap), sr=None)
                    dur = len(wav) / sr
                    if dur < MIN_AUDIO_SEC:
                        raise ValueError(f"Audio too short ({dur:.3f}s < {MIN_AUDIO_SEC}s) at {ap}")
                except Exception as e:
                    print(f"Warning: Could not validate audio: {e}")

                uploaded = None
                try:
                    uploaded = self._retry(self.genai.upload_file, str(ap))
                except Exception:
                    try:
                        uploaded = self._retry(self.client.files.upload, file=str(ap))
                    except Exception:
                        with open(str(ap), "rb") as f:
                            file_content = f.read()
                        uploaded = self._retry(
                            self.client.files.upload,
                            file=file_content,
                            mime_type="audio/wav",
                        )

                contents.append(uploaded)
                if text:
                    contents.append(f"User: {text}")
            else:
                # Assistant turns: inject as plain text so model sees prior context
                if text:
                    contents.append(f"Assistant: {text}")

        if not contents:
            return ""

        config = self.types.GenerateContentConfig(
            system_instruction=system_instruction,
        )

        resp = self._retry(
            self.client.models.generate_content,
            model=self.model_name,
            contents=contents,
            config=config,
        )

        print(f"[Gemini3] Response:\n{resp.text}")

        return (resp.text or "").strip()

    # ========== ASYNC METHODS FOR CONCURRENT PROCESSING ==========
    
    def _generate_sync_wrapper(self, audio_path: Optional[str], text_prompt: Optional[str], system_instruction: Optional[str]) -> str:
        """Wrapper for generate() to use in ThreadPoolExecutor."""
        return self.generate(audio_path=audio_path, text_prompt=text_prompt, system_instruction=system_instruction)
    
    def _transcribe_sync_wrapper(self, audio_path: str) -> str:
        """Wrapper for transcribe() to use in ThreadPoolExecutor."""
        return self.transcribe(audio_path)
    
    def _infer_speaker_sync_wrapper(self, audio_path: str) -> dict:
        """Wrapper for infer_speaker() to use in ThreadPoolExecutor."""
        return self.infer_speaker(audio_path)
    
    async def generate_async(self, audio_path: Optional[str] = None, text_prompt: Optional[str] = None, 
                            system_instruction: Optional[str] = None) -> str:
        """
        Async version of generate() for concurrent processing.
        
        Args:
            audio_path: Path to audio file (optional)
            text_prompt: Text prompt (optional)
            system_instruction: System instruction (optional)
            
        Returns:
            Generated text response
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self._generate_sync_wrapper,
                audio_path,
                text_prompt,
                system_instruction
            )
        return result
    
    async def transcribe_async(self, audio_path: str) -> str:
        """
        Async version of transcribe() for concurrent processing.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self._transcribe_sync_wrapper,
                audio_path
            )
        return result
    
    async def infer_speaker_async(self, audio_path: str) -> dict:
        """
        Async version of infer_speaker() for concurrent processing.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with gender, age_group, and emotion
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self._infer_speaker_sync_wrapper,
                audio_path
            )
        return result
    
    def process_batch(self, audio_paths: List[str], text_prompts: Optional[List[str]] = None,
                     max_concurrent: int = 5) -> List[str]:
        """
        Process multiple audio files concurrently.
        
        Args:
            audio_paths: List of audio file paths
            text_prompts: Optional list of text prompts (one per audio file)
            max_concurrent: Maximum number of concurrent API calls (default: 5)
            
        Returns:
            List of generated responses (same order as input)
            
        Example:
            >>> model = GeminiAudioModel()
            >>> audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
            >>> results = model.process_batch(audio_files, max_concurrent=3)
        """
        if text_prompts is None:
            text_prompts = [None] * len(audio_paths)
        
        if len(text_prompts) != len(audio_paths):
            raise ValueError(f"Length mismatch: {len(audio_paths)} audio files but {len(text_prompts)} prompts")
        
        async def _process_all():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def _process_one(audio_path, text_prompt):
                async with semaphore:
                    try:
                        return await self.generate_async(audio_path=audio_path, text_prompt=text_prompt)
                    except Exception as e:
                        print(f"[ERROR] Failed to process {audio_path}: {e}")
                        return f"ERROR: {str(e)}"
            
            tasks = [_process_one(ap, tp) for ap, tp in zip(audio_paths, text_prompts)]
            return await asyncio.gather(*tasks)
        
        # Run the async function
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new one
                import nest_asyncio
                nest_asyncio.apply()
                results = loop.run_until_complete(_process_all())
            else:
                results = loop.run_until_complete(_process_all())
        except RuntimeError:
            # No event loop in current thread
            results = asyncio.run(_process_all())
        
        return results
    
    def process_batch_simple(self, audio_paths: List[str], text_prompts: Optional[List[str]] = None,
                            max_workers: int = 5) -> List[str]:
        """
        Process multiple audio files using ThreadPoolExecutor (simpler alternative to async).
        
        Args:
            audio_paths: List of audio file paths
            text_prompts: Optional list of text prompts (one per audio file)
            max_workers: Maximum number of concurrent workers (default: 5)
            
        Returns:
            List of generated responses (same order as input)
            
        Example:
            >>> model = GeminiAudioModel()
            >>> audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
            >>> results = model.process_batch_simple(audio_files, max_workers=3)
        """
        if text_prompts is None:
            text_prompts = [None] * len(audio_paths)
        
        if len(text_prompts) != len(audio_paths):
            raise ValueError(f"Length mismatch: {len(audio_paths)} audio files but {len(text_prompts)} prompts")
        
        def _process_one(args):
            audio_path, text_prompt = args
            try:
                return self.generate(audio_path=audio_path, text_prompt=text_prompt)
            except Exception as e:
                print(f"[ERROR] Failed to process {audio_path}: {e}")
                return f"ERROR: {str(e)}"
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_process_one, zip(audio_paths, text_prompts)))
        
        return results

# ---------- Opened-source: Phi4 ----------
class Phi4MultimodalAudioModel(BaseAudioModel):
    """
    microsoft/Phi-4-multimodal-instruct
    - Same interface as other wrappers: generate(audio_path, text_prompt) -> str
    - Uses Script SYSTEM_INSTRUCTION (optionally + user text_prompt)
    - Audio path required for audio mode; text-only supported
    - Important: sets num_logits_to_keep=1 to avoid NoneType slicing error
    
    Note: This model requires prepare_inputs_for_generation to be added to Phi4MMModel class
    in the cached transformers module. The method has been patched in:
    ~/.cache/huggingface/modules/transformers_modules/microsoft/Phi_hyphen_4_hyphen_multimodal_hyphen_instruct/.../modeling_phi4mm.py
    """

    def __init__(self, repo_id: str = "microsoft/Phi-4-multimodal-instruct"):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        self.name = "phi-4-multimodal"
        self.repo_id = repo_id

        # Device / dtype policy consistent with your other wrappers
        if USE_CPU_ONLY:
            self.device = torch.device("cpu")
            self.device_map = {"": "cpu"}
            self.dtype = torch.float32
        else:
            self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
            # device_map="auto" is safer than forcing cuda, and avoids double-moves
            self.device_map = "auto" if self.device.type == "cuda" else {"": "cpu"}
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print("[Phi-4-MM] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.repo_id,
            trust_remote_code=True,
        )

        print("[Phi-4-MM] Loading generation config...")
        self.generation_config = GenerationConfig.from_pretrained(self.repo_id)

        print("[Phi-4-MM] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.repo_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            _attn_implementation="eager",
        )
        self.model.eval()

        # Audio constraints similar to other wrappers
        self.target_sr = TARGET_SR
        self.max_audio_sec = MAX_AUDIO_SEC

        # Determine model device for moving inputs (accelerate models may not expose .device cleanly)
        try:
            self.model_device = next(self.model.parameters()).device
        except StopIteration:
            self.model_device = self.device

        print(f"[Phi-4-MM] Loaded (dtype={self.dtype}) on model_device={self.model_device}")

    def _load_audio(self, audio_path: str):
        wav, sr = librosa.load(audio_path, sr=self.target_sr)

        if wav.size == 0:
            raise ValueError(f"Empty audio: {audio_path}")

        dur = len(wav) / float(self.target_sr)
        if dur < MIN_AUDIO_SEC:
            raise ValueError(f"Audio too short ({dur:.3f}s < {MIN_AUDIO_SEC}s) at {audio_path}")

        if dur > self.max_audio_sec:
            wav = wav[: int(self.max_audio_sec * self.target_sr)]

        return wav, self.target_sr

    def _build_prompt(self, user_text: str, system_instruction: Optional[str] = None) -> str:
        """
        Phi-4 MM example uses special tokens:
          <|user|> ... <|assistant|>
        We keep it minimal and consistent:
          - include audio token if audio is present
          - include system instruction and optional user_text
        """
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"

        # Include system instruction and optional user question
        if system_instruction is None:
            system_instruction = SYSTEM_INSTRUCTION
        
        combined = system_instruction
        if user_text:
            combined = f"{combined}\n\n{user_text}"

        # For audio mode, Phi-4 MM expects <|audio_1|> placeholder
        # (processor aligns it with audios=[...])
        prompt = f"{user_prompt}<|audio_1|>{combined}{prompt_suffix}{assistant_prompt}"
        return prompt

    def generate(self, audio_path: Optional[str] = None, text_prompt: Optional[str] = None, system_instruction: Optional[str] = None) -> str:
        """
        Generate response from Phi-4 Multimodal model.
        
        Args:
            audio_path: Path to audio file (optional)
            text_prompt: Text prompt (optional)
            system_instruction: System instruction (optional, defaults to SYSTEM_INSTRUCTION)
            
        Returns:
            Generated text response
        """
        if system_instruction is None:
            system_instruction = SYSTEM_INSTRUCTION
        
        user_text = text_prompt or ""

        # Text-only mode (supported, but Phi-4 MM is multimodal; keep stable)
        if audio_path is None:
            # No <|audio_1|> in prompt for text-only
            prompt = f"<|user|>{system_instruction}\n\n{user_text}<|end|><|assistant|>"
            inputs = self.processor(text=prompt, return_tensors="pt")
        else:
            wav, sr = self._load_audio(str(audio_path))
            prompt = self._build_prompt(user_text, system_instruction)

            # Use the tuple format audios=[(audio, sr)] as in your working snippet
            inputs = self.processor(
                text=prompt,
                audios=[(wav, sr)],
                return_tensors="pt",
                padding=True,
            )

        # Move tensors to model device (ignore non-tensors)
        inputs = {k: v.to(self.model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Inference best practice (and reduces VRAM)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(self.model_device.type == "cuda")):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                generation_config=self.generation_config,
                num_logits_to_keep=1,   # critical fix for Phi-4 MM remote code path
                use_cache=False,  # Disabled due to DynamicCache compatibility issues
            )

        # Remove prompt part (same as your example)
        if "input_ids" in inputs:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return response
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio using Phi-4 Multimodal model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        wav, sr = self._load_audio(audio_path)
        
        # Build transcription prompt
        prompt = f"<|user|><|audio_1|>You are a speech recognition system.\n\n{ASR_PROMPT}<|end|><|assistant|>"
        
        inputs = self.processor(
            text=prompt,
            audios=[(wav, sr)],
            return_tensors="pt",
            padding=True,
        )
        
        inputs = {k: v.to(self.model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(self.model_device.type == "cuda")):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                generation_config=self.generation_config,
                num_logits_to_keep=1,
                use_cache=False,  # Disabled due to DynamicCache compatibility issues
            )
        
        if "input_ids" in inputs:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        
        return response
    
    def infer_speaker(self, audio_path: str, text_prompt: Optional[str] = None, system_instruction: Optional[str] = None) -> dict:
        """
        Infer speaker characteristics from audio using Phi-4 Multimodal model.
        
        Args:
            audio_path: Path to audio file
            text_prompt: Custom prompt (optional, defaults to INFER_SPEAKER_ID)
            system_instruction: System instruction (optional)
            
        Returns:
            Dictionary with gender, age_group, and emotion
        """
        if text_prompt is None:
            text_prompt = INFER_SPEAKER_ID
        
        if system_instruction is None:
            system_instruction = "You are an expert in speaker profiling."
        
        wav, sr = self._load_audio(str(audio_path))
        
        # Build speaker inference prompt
        combined = f"{system_instruction}\n\n{text_prompt}"
        prompt = f"<|user|><|audio_1|>{combined}<|end|><|assistant|>"
        
        inputs = self.processor(
            text=prompt,
            audios=[(wav, sr)],
            return_tensors="pt",
            padding=True,
        )
        
        inputs = {k: v.to(self.model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(self.model_device.type == "cuda")):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                generation_config=self.generation_config,
                num_logits_to_keep=1,
                use_cache=False,  # Disabled due to DynamicCache compatibility issues
            )
        
        if "input_ids" in inputs:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        
        # Try to parse JSON from response (same logic as Qwen2)
        try:
            parsed = ast.literal_eval(response)
            return {
                "gender": parsed.get("gender", "unknown"),
                "age_group": parsed.get("age_group", "unknown"),
                "emotion": parsed.get("emotion", "unknown")
            }
        except:
            # Try json.loads as fallback
            try:
                parsed = json.loads(response)
                return {
                    "gender": parsed.get("gender", "unknown"),
                    "age_group": parsed.get("age_group", "unknown"),
                    "emotion": parsed.get("emotion", "unknown")
                }
            except:
                return {
                    "gender": None,
                    "age_group": None,
                    "emotion": None,
                    "raw_output": response
                }

    def generate_multiturn(self, audio_root: Path, chat: list, system_instruction=None) -> str:
        """
        Generate a response for a multi-turn conversation.

        User turns are fed as audio with indexed <|audio_N|> placeholders;
        assistant turns are passed as text for accurate conversation context.

        Args:
            audio_root: Root directory that the relative "path" fields in chat
                        are resolved against.
            chat: The dataset's generated["turns"] list. Each element must have
                  "role" ("user"/"assistant"), "text", and "path" for user turns.
            system_instruction: Optional system prompt override.

        Returns:
            Model response string for the final user turn.

        Example:
            response = model.generate_multiturn(
                audio_root=Path("Fairness_TTS_subset"),
                chat=record["generated"]["turns"],
            )
        """
        if system_instruction is None:
            system_instruction = SYSTEM_INSTRUCTION

        user_prompt_tok = "<|user|>"
        asst_prompt_tok = "<|assistant|>"
        end_tok = "<|end|>"

        audio_list = []   # list of (wav, sr) tuples, one per user turn
        prompt_parts = [] # string segments that form the full prompt
        audio_idx = 0     # <|audio_N|> counter (1-based)

        for turn in chat:
            role = turn["role"]
            text = turn.get("text", "") or ""
            audio_file = turn.get("path") or turn.get("audio_path")

            if role == "user" and audio_file:
                wav, sr = self._load_audio(str(os.path.join(audio_root, audio_file)))
                audio_list.append((wav, sr))
                audio_idx += 1
                # Each user audio turn gets its own numbered placeholder
                combined = f"{system_instruction}\n\n{text}" if text else system_instruction
                prompt_parts.append(
                    f"{user_prompt_tok}<|audio_{audio_idx}|>{combined}{end_tok}"
                )
            else:
                # Assistant turn: emit text only so model sees full prior context
                prompt_parts.append(
                    f"{asst_prompt_tok}{text}{end_tok}"
                )

        # Append the final generation prompt
        prompt_parts.append(asst_prompt_tok)
        prompt = "".join(prompt_parts)

        inputs = self.processor(
            text=prompt,
            audios=audio_list,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(self.model_device.type == "cuda")):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                generation_config=self.generation_config,
                num_logits_to_keep=1,
                use_cache=False,
            )

        if "input_ids" in inputs:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return response


# ---------- Open-source: Covo-Audio ----------
class CovoAudioModel(BaseAudioModel):
    """
    Tencent Covo-Audio — 7B end-to-end audio language model.

    Architecture: Whisper-large-v3 encoder + Qwen2.5-7B LLM backbone with an
    AudioAdapter in between.  The model generates interleaved text + audio tokens
    (a2ta mode, matching the original example.py default).  Audio tokens are
    filtered out by _decode_text, so the wrapper always returns plain text,
    compatible with the FSS evaluation pipeline.

    Reference: https://arxiv.org/abs/2602.09823
    GitHub:    https://github.com/Tencent/Covo-Audio
    HuggingFace: tencent/Covo-Audio-Chat

    Local model directory: /home/soumya/Covo-Audio/covoaudio

    Installation:
        pip install -r /home/soumya/Covo-Audio/requirements.txt

    Notes:
      - Text-only input (audio_path=None) returns "" — model is audio-first.
      - system_instruction / text_prompt are ignored; Covo-Audio has a fixed
        Chinese system prompt baked into the dialog template.
      - generate_multiturn feeds all turns sequentially, reusing KV-cache.
    """

    _MODEL_DIR = "/home/soumya/Covo-Audio/covoaudio"
    # Spoken system-prompt audio files from audio_system_prompts/.
    # Each method concatenates the appropriate file before the user audio.

    def __init__(self, model_dir: Optional[str] = None):
        import sys
        import numpy as np
        # Ensure the Covo-Audio repo is importable
        _covo_root = str(Path("/home/soumya/Covo-Audio"))
        if _covo_root not in sys.path:
            sys.path.insert(0, _covo_root)

        from transformers import AutoConfig, AutoModel, LogitsProcessorList
        from transformers.models.qwen2 import Qwen2TokenizerFast
        from covoaudio.configuration_covo_audio import CovoAudioConfig
        from covoaudio.modeling_covo_audio import (
            CovoAudioForCausalLM,
            get_dialog_prompt,
            WindowedRepetitionPenaltyLogitsProcessor,
        )

        self.name = "covo-audio"
        self._model_dir = model_dir or self._MODEL_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU_ONLY else "cpu")

        self._SYS_PROMPT_AUDIO     = "/home/soumya/AudioLLM/audio_system_prompts/SYSTEM_INSTRUCTION.wav"
        self._ASR_PROMPT_AUDIO     = "/home/soumya/AudioLLM/audio_system_prompts/ASR_PROMPT.wav"
        self._SPEAKER_PROMPT_AUDIO = "/home/soumya/AudioLLM/audio_system_prompts/INFER_SPEAKER_ID.wav"

        # Store imports for later use in methods
        self._get_dialog_prompt = get_dialog_prompt
        self._WindowedRepetitionPenaltyLogitsProcessor = WindowedRepetitionPenaltyLogitsProcessor
        self._LogitsProcessorList = LogitsProcessorList

        # Register custom config/model classes with HuggingFace auto
        AutoConfig.register("covo_audio", CovoAudioConfig)
        AutoModel.register(CovoAudioConfig, CovoAudioForCausalLM)
        CovoAudioConfig.register_for_auto_class("AutoConfig")
        CovoAudioForCausalLM.register_for_auto_class("AutoModel")

        print(f"[CovoAudio] Loading model from {self._model_dir} ...")
        config = CovoAudioConfig.from_pretrained(self._model_dir)
        self.model = AutoModel.from_config(config)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load safetensors weights
        from safetensors.torch import load_file as safetensors_load
        import os as _os
        weight_files = sorted(
            f for f in _os.listdir(self._model_dir) if f.endswith(".safetensors")
        )
        for wf in weight_files:
            state_dict = safetensors_load(_os.path.join(self._model_dir, wf))
            self.model.load_state_dict(state_dict, strict=False)
        print(f"[CovoAudio] Loaded {len(weight_files)} weight shard(s).")

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(self._model_dir)

        # Use a2t mode (matching example.py/example.sh): model generates plain text only.
        # <|endoftext|> is the EOS for this mode.
        self._eos_token_id = self.tokenizer.encode("<|endoftext|>")[0]

        # Fixed reusable path where the concatenated audio is written before
        # each inference call and deleted immediately after.  Using a fixed name
        # (rather than a fresh tempfile each time) means only one scratch file
        # ever exists on disk, avoiding accumulation.
        self._concat_audio_path: str = str(Path(self._model_dir) / "_covo_concat_tmp.wav")

        print(f"[CovoAudio] Ready on {self.device}.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_text(self, tokens: torch.Tensor) -> str:
        """Decode only the text tokens from a generation output tensor."""
        filtered = tokens[tokens < len(self.tokenizer)]
        decoded = self.tokenizer.decode(filtered, skip_special_tokens=True)
        return decoded

    def _concat_with_sys_prompt(self, audio_path: str, sys_prompt_path: Optional[str]) -> str:
        """
        Concatenate sys_prompt_path audio + audio_path audio, write the result
        to self._concat_audio_path (overwriting any previous version), and
        return that path.  The caller is responsible for deleting the file
        after use via _cleanup_concat().

        Covo-Audio uses 24 kHz internally; both files are resampled to 24 kHz
        before concatenation so the model sees a consistent sample rate.
        """
        import numpy as np
        _SR = 24_000  # Covo-Audio native sample rate
        prompt_wav, _ = librosa.load(str(sys_prompt_path), sr=_SR, mono=True)
        user_wav, _ = librosa.load(str(audio_path), sr=_SR, mono=True)
        combined = np.concatenate([prompt_wav, user_wav])
        sf.write(self._concat_audio_path, combined, _SR)
        return self._concat_audio_path

    def _cleanup_concat(self) -> None:
        """Delete the scratch concatenation file if it exists."""
        try:
            Path(self._concat_audio_path).unlink(missing_ok=True)
        except Exception:
            pass

    def _run_inference(self, audio_path: str, sys_prompt_path: Optional[str] = None,
                       first_round: bool = True,
                       past_key_values=None, attention_mask=None) -> Tuple[str, object]:
        """
        Run a single inference step.

        Prepends sys_prompt_path audio (defaults to self._sys_prompt_path) to
        audio_path, writes the result to a fixed scratch file, runs generation,
        then immediately deletes the scratch file.

        Returns:
            (decoded_text, past_key_values) so the caller can chain multi-turn.
        """
        if sys_prompt_path is not None:
            _sys = sys_prompt_path
            audio_path = self._concat_with_sys_prompt(audio_path, _sys)
        try:
            wavs, input_ids, attn_mask = self._get_dialog_prompt(
                audio=audio_path,
                tokenizer=self.tokenizer,
                device=self.device,
                first_round=first_round,
            )
        finally:
            self._cleanup_concat()

        # If multi-turn, extend attention mask to cover history
        if past_key_values is not None and attention_mask is not None:
            history_len = past_key_values.get_seq_length()
            current_len = attn_mask.shape[1]
            attn_mask = torch.ones(
                (1, history_len + current_len), device=self.device
            )

        logits_processor = self._LogitsProcessorList([
            self._WindowedRepetitionPenaltyLogitsProcessor(penalty=1.05, window_size=8)
        ])

        input_len = input_ids.shape[1]
        with torch.no_grad():
            result = self.model.generate(
                input_ids=input_ids,
                wavs=wavs,
                attention_mask=attn_mask,
                max_new_tokens=2048,
                eos_token_id=self._eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                past_key_values=past_key_values,
                return_dict_in_generate=True,
                logits_processor=logits_processor,
                repetition_penalty=1.05,
            )

        seq = result.sequences
        tokens = seq.squeeze(0)[input_len:]
        text = self._decode_text(tokens)
        return text, result.past_key_values

    # ------------------------------------------------------------------
    # BaseAudioModel interface
    # ------------------------------------------------------------------

    def generate(
        self,
        audio_path: Optional[str] = None,
        text_prompt: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Return Covo-Audio's text response to the input audio.

        Args:
            audio_path: Path to input audio file (required; returns "" if None).
            text_prompt: Not used by Covo-Audio (silently ignored).
            system_instruction: Not used by Covo-Audio (silently ignored).
        """
        if audio_path is None:
            return ""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        try:
            text, _ = self._run_inference(
                audio_path,
                sys_prompt_path=self._SYS_PROMPT_AUDIO,
                first_round=True,
            )
            return text
        except Exception as e:
            print(f"[CovoAudio] generate() error: {e}")
            return ""

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using Covo-Audio's text output."""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        try:
            text, _ = self._run_inference(
                audio_path,
                sys_prompt_path=self._ASR_PROMPT_AUDIO,
                first_round=True,
            )
            return text
        except Exception as e:
            print(f"[CovoAudio] transcribe() error: {e}")
            return ""

    def infer_speaker(
        self,
        audio_path: str,
        text_prompt: Optional[str] = None,
    ) -> dict:
        """Attempt speaker profiling from Covo-Audio's text output.

        Covo-Audio is not designed for structured speaker analysis, so the raw
        response is returned under 'raw_output' alongside None values for the
        structured fields.
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        try:
            response, _ = self._run_inference(
                audio_path,
                sys_prompt_path=self._SPEAKER_PROMPT_AUDIO,
                first_round=True,
            )
            parsed = extract_json(response)
            if parsed and all(k in parsed for k in ("gender", "age_group", "emotion")):
                return parsed
            return {
                "gender": None,
                "age_group": None,
                "emotion": None,
                "raw_output": response,
            }
        except Exception as e:
            print(f"[CovoAudio] infer_speaker() error: {e}")
            return {"gender": None, "age_group": None, "emotion": None, "raw_output": "No output"}

    def generate_multiturn(
        self,
        audio_root: Path,
        chat: list,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate a response for a multi-turn conversation.

        Processes each user audio turn sequentially, reusing the KV-cache so
        the model retains context across rounds.

        Args:
            audio_root: Root directory for resolving relative audio paths.
            chat: List of turn dicts with keys 'role', 'text', and 'path'.
            system_instruction: Ignored (Covo-Audio uses a fixed system prompt).

        Returns:
            Model text response for the final user turn.
        """
        past_key_values = None
        attn_mask = None
        response = ""
        first_round = True

        for turn in chat:
            role = turn.get("role", "")
            audio_file = turn.get("path") or turn.get("audio_path")

            if role != "user" or not audio_file:
                # Skip assistant turns — history is preserved in KV-cache
                continue

            full_audio_path = Path(audio_root) / audio_file
            if not full_audio_path.exists():
                print(f"[CovoAudio] Audio not found, skipping turn: {full_audio_path}")
                continue

            try:
                response, past_key_values = self._run_inference(
                    audio_path=str(full_audio_path),
                    first_round=first_round,
                    past_key_values=past_key_values,
                    attention_mask=attn_mask,
                )
            except Exception as e:
                print(f"[CovoAudio] generate_multiturn() turn error: {e}")

            first_round = False  # subsequent turns reuse KV-cache; sys-prompt already prepended

        return response


# # ---------- Opened-source: IBM Granite ----------
# class GraniteSpeechAudioModel(BaseAudioModel):
#     """
#     ibm-granite/granite-speech-3.3-8b

#     Fixes:
#       - Does NOT pass sampling_rate to processor (tokenizer path doesn't accept it)
#       - Avoids meta device issues (never .to('meta'); chooses a real device)
#     """

#     def __init__(
#         self,
#         repo_id: str = "ibm-granite/granite-speech-3.3-8b",
#         force_single_device: bool = True,   # recommended for stable batch eval
#     ):
#         from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

#         self.name = "granite-speech-3.3"
#         self.repo_id = repo_id

#         # Granite expects mono 16kHz audio
#         self.target_sr = 16000
#         self.max_audio_sec = MAX_AUDIO_SEC

#         # Decide runtime device
#         if USE_CPU_ONLY or not torch.cuda.is_available():
#             self.runtime_device = torch.device("cpu")
#             self.dtype = torch.float32
#         else:
#             self.runtime_device = torch.device("cuda:0")
#             # bf16 typically works well; if your GPU doesn't support bf16, switch to float16
#             self.dtype = torch.bfloat16

#         self.processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=True)
#         self.tokenizer = self.processor.tokenizer

#         # ---- Model loading strategy ----
#         # For evaluation, stability > clever sharding; single device avoids meta tensors entirely.
#         if force_single_device:
#             self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
#                 self.repo_id,
#                 torch_dtype=self.dtype,
#             ).to(self.runtime_device)
#             self.model.eval()
#             self.model_device = self.runtime_device
#             print(f"[Granite] Loaded on {self.model_device} (dtype={self.dtype}, single_device=True)")
#         else:
#             # Sharded / accelerate path (safe device selection added)
#             device_map = {"": "cpu"} if self.runtime_device.type == "cpu" else "auto"
#             self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
#                 self.repo_id,
#                 device_map=device_map,
#                 torch_dtype=self.dtype,
#             )
#             self.model.eval()
#             self.model_device = self._pick_real_device()
#             print(f"[Granite] Loaded with device_map={device_map}; using input device {self.model_device} (dtype={self.dtype})")

#     def _pick_real_device(self) -> torch.device:
#         """
#         For accelerate-sharded models, ensure we pick a concrete device for inputs.
#         Never return meta/disk.
#         """
#         hf_map = getattr(self.model, "hf_device_map", None)
#         if isinstance(hf_map, dict) and len(hf_map) > 0:
#             for _, dev in hf_map.items():
#                 if dev not in ("meta", "disk"):
#                     return torch.device(dev)

#         for p in self.model.parameters():
#             if p.device.type != "meta":
#                 return p.device

#         # last resort
#         return torch.device("cuda:0" if (torch.cuda.is_available() and not USE_CPU_ONLY) else "cpu")

#     def _load_audio(self, audio_path: str):
#         import torchaudio

#         wav, sr = torchaudio.load(audio_path, normalize=True)

#         if wav.numel() == 0:
#             raise ValueError(f"Empty audio: {audio_path}")

#         # Convert to mono
#         if wav.shape[0] != 1:
#             wav = wav.mean(dim=0, keepdim=True)

#         # Resample to 16kHz
#         if sr != self.target_sr:
#             wav = torchaudio.functional.resample(wav, sr, self.target_sr)
#             sr = self.target_sr

#         dur = wav.shape[-1] / float(sr)
#         if dur < MIN_AUDIO_SEC:
#             raise ValueError(f"Audio too short ({dur:.3f}s < {MIN_AUDIO_SEC}s) at {audio_path}")

#         if dur > self.max_audio_sec:
#             wav = wav[:, : int(self.max_audio_sec * sr)]

#         # Return 1D waveform as Granite examples do
#         return wav.squeeze(0), sr

#     def _build_prompt(self, user_prompt: str) -> str:
#         # Granite requires the <|audio|> placeholder in the user message.
#         # Keep system instruction aligned with your script.
#         chat = [
#             {"role": "system", "content": SYSTEM_INSTRUCTION},
#             {"role": "user", "content": f"<|audio|>{user_prompt}"},
#         ]
#         return self.tokenizer.apply_chat_template(
#             chat,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#     def generate(self, audio_path: Optional[str], text_prompt: Optional[str] = None) -> str:
#         # Audio-first model; for text-only evaluations return empty string (keeps your pipeline stable)
#         if audio_path is None:
#             return ""

#         user_prompt = (text_prompt or "").strip()
#         if not user_prompt:
#             # default behavior for safety/fairness runs if prompt is missing
#             user_prompt = "Please transcribe the speech."

#         prompt = self._build_prompt(user_prompt)
#         wav, _sr = self._load_audio(str(audio_path))

#         # IMPORTANT: Granite processor DOES NOT accept sampling_rate kwarg here.
#         # Also keep call signature close to IBM snippet.
#         inputs = self.processor(
#             prompt,
#             wav,
#             return_tensors="pt",
#         )

#         # Move tensor inputs to a REAL device (never meta)
#         inputs = {k: v.to(self.model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 do_sample=False,
#                 num_beams=1,
#                 use_cache=False,  # reduces cache-related surprises across models
#             )

#         # Granite includes prompt tokens in output; slice them out
#         num_input_tokens = inputs["input_ids"].shape[-1]
#         gen_ids = output_ids[:, num_input_tokens:]

#         text = self.tokenizer.batch_decode(
#             gen_ids,
#             skip_special_tokens=True,
#             add_special_tokens=False,
#         )[0].strip()

#         return text

