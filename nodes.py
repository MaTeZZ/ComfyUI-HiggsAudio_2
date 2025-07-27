from .boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from .boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio
import time
import click
import os
import base64
import io
import json
import sys
import re
from typing import List

try:
    import soundfile as sf
except ImportError:
    print("Warning: soundfile not installed. Install with: pip install soundfile")

_engine_cache = {}

class HiggsAudioTextChunker:
    """Text chunker optimized for HiggsAudio2 with 2000 token limit"""
    @staticmethod
    def count_tokens_approximate(text: str) -> int:
        return len(text) // 4

    @staticmethod
    def split_into_sentences(text: str, max_tokens: int = 2000) -> List[str]:
        print(f"🧩 [Chunker] Input: {len(text)} chars / {HiggsAudioTextChunker.count_tokens_approximate(text)} tokens (max {max_tokens} tokens/chunk)")
        if not text.strip():
            print("🧩 [Chunker] Empty input, returning []")
            return []
        text = re.sub(r'\s+', ' ', text.strip())
        if HiggsAudioTextChunker.count_tokens_approximate(text) <= max_tokens:
            print("🧩 [Chunker] Single chunk is enough.")
            return [text]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence: continue
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            test_tokens = HiggsAudioTextChunker.count_tokens_approximate(test_chunk)
            if test_tokens <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    print(f"🧩 [Chunker] Finalizing chunk of {HiggsAudioTextChunker.count_tokens_approximate(current_chunk)} tokens")
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if HiggsAudioTextChunker.count_tokens_approximate(sentence) > max_tokens:
                        parts = re.split(r'(?<=[,;:])\s+', sentence)
                        sub_chunk = ""
                        for part in parts:
                            test_part = sub_chunk + " " + part if sub_chunk else part
                            if HiggsAudioTextChunker.count_tokens_approximate(test_part) <= max_tokens:
                                sub_chunk = test_part
                            else:
                                if sub_chunk:
                                    print(f"🧩 [Chunker] Finalizing sub-chunk of {HiggsAudioTextChunker.count_tokens_approximate(sub_chunk)} tokens")
                                    chunks.append(sub_chunk.strip())
                                sub_chunk = part
                        if sub_chunk:
                            current_chunk = sub_chunk
                    else:
                        current_chunk = sentence
        if current_chunk.strip():
            print(f"🧩 [Chunker] Finalizing last chunk of {HiggsAudioTextChunker.count_tokens_approximate(current_chunk)} tokens")
            chunks.append(current_chunk.strip())
        print(f"🧩 [Chunker] Returning {len(chunks)} chunks.")
        return chunks

    @staticmethod
    def combine_audio_chunks(audio_segments: List[dict], silence_ms: int = 100) -> dict:
        print(f"🔗 [Chunker] Combining {len(audio_segments)} chunks with {silence_ms}ms silence")
        if len(audio_segments) == 1:
            return audio_segments[0]
        combined_waveform = audio_segments[0]["waveform"]
        sample_rate = audio_segments[0]["sample_rate"]
        silence_samples = int(silence_ms * sample_rate / 1000)
        for i in range(1, len(audio_segments)):
            if silence_ms > 0:
                silence_shape = list(combined_waveform.shape)
                silence_shape[-1] = silence_samples
                silence = torch.zeros(*silence_shape)
                combined_waveform = torch.cat([combined_waveform, silence], dim=-1)
            next_waveform = audio_segments[i]["waveform"]
            combined_waveform = torch.cat([combined_waveform, next_waveform], dim=-1)
        print(f"🔗 [Chunker] Combined waveform shape: {combined_waveform.shape}")
        return {"waveform": combined_waveform, "sample_rate": sample_rate}

def load_voice_presets():
    try:
        voice_examples_dir = os.path.join(os.path.dirname(__file__), "voice_examples")
        config_path = os.path.join(voice_examples_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            voice_dict = json.load(f)
        voice_presets = {}
        for k, v in voice_dict.items():
            voice_presets[k] = v["transcript"]
        voice_presets["voice_clone"] = "No reference voice (use custom audio)"
        return voice_presets, voice_dict
    except FileNotFoundError:
        print("ERROR: Voice examples config file not found. Using empty voice presets.")
        return {"voice_clone": "No reference voice (use custom audio)"}, {}
    except Exception as e:
        print(f"ERROR: Error loading voice presets: {e}")
        return {"voice_clone": "No reference voice (use custom audio)"}, {}

def get_voice_preset_path(voice_preset):
    if voice_preset == "voice_clone":
        return None
    voice_examples_dir = os.path.join(os.path.dirname(__file__), "voice_examples")
    voice_path = os.path.join(voice_examples_dir, f"{voice_preset}.wav")
    if os.path.exists(voice_path):
        return voice_path
    return None

try:
    VOICE_PRESETS, VOICE_DICT = load_voice_presets()
except Exception as e:
    print(f"ERROR: Failed to load voice presets: {e}")
    VOICE_PRESETS, VOICE_DICT = {"voice_clone": "No reference voice (use custom audio)"}, {}

class LoadHiggsAudioModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_path": ("STRING", {"default": "bosonai/higgs-audio-v2-generation-3B-base"}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL_PATH",)
    FUNCTION = "load_model"
    CATEGORY = "Higgs Audio"
    def load_model(self, model_path): return (model_path,)

class LoadHiggsAudioTokenizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_path": ("STRING", {"default": "bosonai/higgs-audio-v2-tokenizer"}),
        }}
    RETURN_TYPES = ("AUDIOTOKENIZER",)
    RETURN_NAMES = ("AUDIO_TOKENIZER_PATH",)
    FUNCTION = "load_model"
    CATEGORY = "Higgs Audio"
    def load_model(self, model_path): return (model_path,)

class LoadHiggsAudioSystemPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {
                "default": "Generate audio following instruction.",
                "multiline": True
            }),
        }}
    RETURN_TYPES = ("SYSTEMPROMPT",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Higgs Audio"
    def load_prompt(self, text): return (text,)

class LoadHiggsAudioPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {
                "default": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
                "multiline": True
            }),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_prompt"
    CATEGORY = "Higgs Audio"
    def load_prompt(self, text): return (text,)

class HiggsAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL_PATH": ("MODEL",),
                "AUDIO_TOKENIZER_PATH": ("AUDIOTOKENIZER",),
                "system_prompt": ("SYSTEMPROMPT",),
                "prompt": ("STRING",),
                "max_new_tokens": ("INT", {"default": 2048, "min": 128, "max": 4096}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": -1, "max": 100}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "voice_preset": (list(VOICE_PRESETS.keys()), {"default": "voice_clone"}),
                "reference_audio": ("AUDIO",),
                "reference_text": ("STRING", {"default": "", "multiline": True}),
                "audio_priority": (["preset_dropdown", "reference_input", "auto", "force_preset"], {"default": "auto"}),
                "enable_chunking": ("BOOLEAN", {"default": True}),
                "max_tokens_per_chunk": ("INT", {"default": 225, "min": 200, "max": 4096, "step": 25}),
                "silence_between_chunks_ms": ("INT", {"default": 100, "min": 0, "max": 2000, "step": 25}),
            }
        }
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("output", "generation_info")
    FUNCTION = "generate"
    CATEGORY = "Higgs Audio"

    def __init__(self):
        self.chunker = HiggsAudioTextChunker()

    def process_single_chunk(self, chunk_text, MODEL_PATH, AUDIO_TOKENIZER_PATH, system_prompt,
                            max_new_tokens, temperature, top_p, top_k, device,
                            voice_preset, reference_audio, reference_text, audio_priority):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = f"{MODEL_PATH}_{AUDIO_TOKENIZER_PATH}_{device}"
        if cache_key not in _engine_cache:
            print(f"Loading HiggsAudio engine: {MODEL_PATH}")
            _engine_cache[cache_key] = HiggsAudioServeEngine(
                MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)
        serve_engine = _engine_cache[cache_key]
        messages = []
        if len(system_prompt.strip()) > 0:
            messages.append(Message(role="system", content=system_prompt))
        audio_for_cloning = None
        text_for_cloning = ""
        used_voice_info = "No voice cloning"
        has_valid_reference_audio = False
        if reference_audio is not None:
            try:
                if isinstance(reference_audio, dict) and "waveform" in reference_audio:
                    waveform = reference_audio["waveform"]
                    if hasattr(waveform, 'shape') and waveform.numel() > 0:
                        has_valid_reference_audio = True
            except Exception:
                pass
        if audio_priority == "preset_dropdown":
            use_preset = voice_preset != "voice_clone"
            use_input = not use_preset and has_valid_reference_audio
        elif audio_priority == "reference_input":
            use_input = has_valid_reference_audio
            use_preset = not use_input and voice_preset != "voice_clone"
        elif audio_priority == "force_preset":
            use_preset = voice_preset != "voice_clone"
            use_input = False
        else:
            if voice_preset != "voice_clone":
                use_preset = True
                use_input = False
            else:
                use_preset = False
                use_input = has_valid_reference_audio
        if use_preset:
            voice_path = get_voice_preset_path(voice_preset)
            if voice_path and os.path.exists(voice_path):
                try:
                    waveform, sample_rate = torchaudio.load(voice_path)
                    if waveform.dim() == 2:
                        waveform = waveform.unsqueeze(0)
                    audio_for_cloning = {"waveform": waveform.float(), "sample_rate": sample_rate}
                    text_for_cloning = VOICE_PRESETS.get(voice_preset, "")
                    used_voice_info = f"Voice Preset: {voice_preset}"
                except Exception as e:
                    print(f"Error loading voice preset {voice_preset}: {e}")
                    audio_for_cloning = None
        elif use_input:
            audio_for_cloning = reference_audio
            text_for_cloning = reference_text.strip()
            used_voice_info = "Reference Audio Input"
            if not text_for_cloning:
                text_for_cloning = "Reference audio for voice cloning."
        if audio_for_cloning is not None:
            try:
                audio_base64 = self._audio_to_base64(audio_for_cloning)
                if audio_base64:
                    if text_for_cloning:
                        messages.append(Message(role="system", content=text_for_cloning))
                    else:
                        messages.append(Message(role="system", content="Reference audio for voice cloning."))
                    audio_content = AudioContent(raw_audio=audio_base64, audio_url="")
                    messages.append(Message(role="assistant", content=[audio_content]))
                else:
                    used_voice_info = "Audio encoding failed - using basic TTS"
            except Exception as e:
                print(f"Error in audio processing: {e}")
                used_voice_info = f"Audio processing error: {str(e)}"
        messages.append(Message(role="user", content=chunk_text))
        print(f"🗣️ [HiggsAudio] Generating audio for chunk of {len(chunk_text)} chars / ~{self.chunker.count_tokens_approximate(chunk_text)} tokens.")
        try:
            output: HiggsAudioResponse = serve_engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )
        except Exception as e:
            print(f"Error during audio generation: {e}")
            raise e
        if hasattr(output, 'audio') and hasattr(output, 'sampling_rate'):
            audio_np = output.audio
            if len(audio_np.shape) == 1:
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).float()
            elif len(audio_np.shape) == 2:
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()
            else:
                audio_tensor = torch.from_numpy(audio_np).float()
            chunk_audio = {
                "waveform": audio_tensor,
                "sample_rate": output.sampling_rate
            }
            print(f"🗣️ [HiggsAudio] Finished chunk, audio shape: {audio_tensor.shape}, sample_rate: {output.sampling_rate}")
            return chunk_audio, used_voice_info
        else:
            raise ValueError("Invalid audio output from HiggsAudio engine")

    def generate(self, MODEL_PATH, AUDIO_TOKENIZER_PATH, system_prompt, prompt, max_new_tokens, 
                 temperature, top_p, top_k, device, voice_preset="voice_clone", reference_audio=None, 
                 reference_text="", audio_priority="auto", enable_chunking=True, 
                 max_tokens_per_chunk=250, silence_between_chunks_ms=100):

        print("="*50)
        print(f"🚨 [HiggsAudioNode] Full prompt length: {len(prompt)} chars")
        approx_tokens = self.chunker.count_tokens_approximate(prompt)
        print(f"🚨 [HiggsAudioNode] Full prompt approx tokens: {approx_tokens}")
        print(f"🚨 [HiggsAudioNode] Prompt preview: {prompt[:120]}{'...' if len(prompt)>120 else ''}")
        try:
            with open("DEBUG_higgs_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception as e:
            print(f"[DEBUG] Could not write DEBUG_higgs_prompt.txt ({e})")
        if enable_chunking is None: enable_chunking = True
        if max_tokens_per_chunk is None or max_tokens_per_chunk < 200: max_tokens_per_chunk = 1200
        if silence_between_chunks_ms is None: silence_between_chunks_ms = 100
        start_time = time.time()
        prompt_tokens = self.chunker.count_tokens_approximate(prompt)
        if not enable_chunking or prompt_tokens <= max_tokens_per_chunk:
            print(f"🔊 Processing as single chunk: {prompt_tokens} tokens ~ {len(prompt)} chars")
            chunk_audio, voice_info = self.process_single_chunk(
                prompt, MODEL_PATH, AUDIO_TOKENIZER_PATH, system_prompt,
                max_new_tokens, temperature, top_p, top_k, device,
                voice_preset, reference_audio, reference_text, audio_priority
            )
            duration = chunk_audio['waveform'].size(-1) / chunk_audio['sample_rate']
            info = f"Generated {duration:.1f}s audio from {prompt_tokens} tokens (single chunk)"
            print(f"✅ Final audio duration: {duration:.1f} seconds ({duration/60:.1f} min)")
            print("="*50)
            return (chunk_audio, info)

        print(f"🔎 [HiggsAudioNode] Split start: {prompt_tokens} tokens for chunking, {max_tokens_per_chunk} tokens/chunk allowed.")
        chunks = self.chunker.split_into_sentences(prompt, max_tokens_per_chunk)
        print(f"🔎 [HiggsAudioNode] Got {len(chunks)} chunks from split.")
        print(f"   Max tokens/chunk: {max_tokens_per_chunk}, Max chars/chunk: ~{max_tokens_per_chunk*4}")

        audio_segments = []
        total_chunk_tokens = 0
        with open("DEBUG_chunk_audio_lengths.log", "w", encoding="utf-8") as log_file:
            for i, chunk in enumerate(chunks):
                chunk_tokens = self.chunker.count_tokens_approximate(chunk)
                chunk_chars = len(chunk)
                preview = chunk[:80] + ("..." if len(chunk) > 80 else "")
                print(f"🎤 Chunk {i+1}/{len(chunks)}: {chunk_tokens} tokens / {chunk_chars} chars")
                print(f"   Preview: {preview}")

                chunk_audio, voice_info = self.process_single_chunk(
                    chunk, MODEL_PATH, AUDIO_TOKENIZER_PATH, system_prompt,
                    max_new_tokens, temperature, top_p, top_k, device,
                    voice_preset, reference_audio, reference_text, audio_priority
                )
                audio_segments.append(chunk_audio)
                total_chunk_tokens += chunk_tokens

                samples = chunk_audio['waveform'].size(-1)
                sr = chunk_audio['sample_rate']
                audio_sec = samples / sr
                print(f"   [AUDIO] Chunk {i+1}: {chunk_chars} chars, {chunk_tokens} tokens => audio {audio_sec:.2f}s, shape {chunk_audio['waveform'].shape}")
                log_file.write(f"Chunk {i+1}: {chunk_chars} chars, {chunk_tokens} tokens, audio {audio_sec:.2f} sec\n")

        print(f"🔗 [HiggsAudioNode] Concatenating {len(audio_segments)} audio segments")
        print(f"   Adding {silence_between_chunks_ms}ms silence between chunks")
        combined_audio = self.chunker.combine_audio_chunks(audio_segments, silence_between_chunks_ms)
        duration = combined_audio['waveform'].size(-1) / combined_audio['sample_rate']
        avg_chunk_tokens = total_chunk_tokens // len(chunks)
        word_est = len(prompt.split())
        print(f"✅ Final audio duration: {duration:.1f} seconds ({duration/60:.1f} min)")
        print(f"📐 Estimated input words: {word_est} | Est. min: {round(word_est/150, 1)} min (@150wpm)")
        print(f"📄 All chunk tokens total: {total_chunk_tokens} | Original prompt tokens est: {prompt_tokens}")
        print("="*50)
        info = (f"Generated {duration:.1f}s ({duration/60:.1f}min) audio from "
                f"{prompt_tokens} tokens using {len(chunks)} chunks (avg {avg_chunk_tokens} tokens/chunk)")
        return (combined_audio, info)

    def _audio_to_base64(self, comfy_audio):
        waveform = comfy_audio["waveform"]
        sample_rate = comfy_audio["sample_rate"]
        if waveform.dim() == 3:
            audio_np = waveform[0, 0].numpy()
        elif waveform.dim() == 2:
            audio_np = waveform[0].numpy()
        else:
            audio_np = waveform.numpy()
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return audio_base64

NODE_CLASS_MAPPINGS = {
    "LoadHiggsAudioModel": LoadHiggsAudioModel,
    "LoadHiggsAudioTokenizer": LoadHiggsAudioTokenizer,
    "LoadHiggsAudioSystemPrompt": LoadHiggsAudioSystemPrompt,
    "LoadHiggsAudioPrompt": LoadHiggsAudioPrompt,
    "HiggsAudio": HiggsAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHiggsAudioModel": "Load HiggsAudio Model",
    "LoadHiggsAudioTokenizer": "Load HiggsAudio Tokenizer",
    "LoadHiggsAudioSystemPrompt": "Load HiggsAudio System Prompt",
    "LoadHiggsAudioPrompt": "Load HiggsAudio Prompt",
    "HiggsAudio": "HiggsAudio Text-to-Speech",
}
