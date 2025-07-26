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

try:
    import soundfile as sf
except ImportError:
    print("Warning: soundfile not installed. Install with: pip install soundfile")

# Global engine cache to avoid reloading
_engine_cache = {}

def load_voice_presets():
    """Load the voice presets from the voice_examples directory."""
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
    """Get the voice path for a given voice preset."""
    if voice_preset == "voice_clone":
        return None
    
    voice_examples_dir = os.path.join(os.path.dirname(__file__), "voice_examples")
    voice_path = os.path.join(voice_examples_dir, f"{voice_preset}.wav")
    
    if os.path.exists(voice_path):
        return voice_path
    return None

# Load voice presets at module level
try:
    VOICE_PRESETS, VOICE_DICT = load_voice_presets()
except Exception as e:
    print(f"ERROR: Failed to load voice presets: {e}")
    VOICE_PRESETS, VOICE_DICT = {"voice_clone": "No reference voice (use custom audio)"}, {}

class LoadHiggsAudioModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "bosonai/higgs-audio-v2-generation-3B-base"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL_PATH",)
    FUNCTION = "load_model"
    CATEGORY = "Higgs Audio"

    def load_model(self, model_path):
        MODEL_PATH = model_path
        return (MODEL_PATH,)


class LoadHiggsAudioTokenizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "bosonai/higgs-audio-v2-tokenizer"}),
            }
        }

    RETURN_TYPES = ("AUDIOTOKENIZER",)
    RETURN_NAMES = ("AUDIO_TOKENIZER_PATH",)
    FUNCTION = "load_model"
    CATEGORY = "Higgs Audio"

    def load_model(self, model_path):
        AUDIO_TOKENIZER_PATH = model_path
        return (AUDIO_TOKENIZER_PATH,)


class LoadHiggsAudioSystemPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "Generate audio following instruction.",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("SYSTEMPROMPT",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Higgs Audio"

    def load_prompt(self, text):
        system_prompt = text
        return (system_prompt,)


class LoadHiggsAudioPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_prompt"
    CATEGORY = "Higgs Audio"

    def load_prompt(self, text):
        return (text,)


class HiggsAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL_PATH": ("MODEL",),
                "AUDIO_TOKENIZER_PATH": ("AUDIOTOKENIZER",),
                "system_prompt": ("SYSTEMPROMPT",),
                "prompt": ("STRING",),
                "max_new_tokens": ("INT", {"default": 1024, "min": 128, "max": 4096}),
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
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("output", "used_voice_info")
    FUNCTION = "generate"
    CATEGORY = "Higgs Audio"

    def generate(self, MODEL_PATH, AUDIO_TOKENIZER_PATH, system_prompt, prompt, max_new_tokens, temperature, top_p, top_k, device, voice_preset="voice_clone", reference_audio=None, reference_text="", audio_priority="auto"):
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create cache key
        cache_key = f"{MODEL_PATH}_{AUDIO_TOKENIZER_PATH}_{device}"
        
        # Check if engine is already loaded
        if cache_key not in _engine_cache:
            print(f"Loading HiggsAudio engine: {MODEL_PATH}")
            _engine_cache[cache_key] = HiggsAudioServeEngine(
                MODEL_PATH, 
                AUDIO_TOKENIZER_PATH, 
                device=device
            )
        
        serve_engine = _engine_cache[cache_key]
        
        # Prepare messages
        messages = []
        
        # Add system message if provided
        if len(system_prompt.strip()) > 0:
            messages.append(Message(role="system", content=system_prompt))
        
        # Determine which audio source to use based on priority
        audio_for_cloning = None
        text_for_cloning = ""
        used_voice_info = "No voice cloning"
        
        # Better detection of valid reference audio
        has_valid_reference_audio = False
        if reference_audio is not None:
            try:
                if isinstance(reference_audio, dict) and "waveform" in reference_audio:
                    waveform = reference_audio["waveform"]
                    if hasattr(waveform, 'shape') and waveform.numel() > 0:
                        has_valid_reference_audio = True
            except Exception:
                pass
        
        # Priority logic
        if audio_priority == "preset_dropdown":
            use_preset = voice_preset != "voice_clone"
            use_input = not use_preset and has_valid_reference_audio
        elif audio_priority == "reference_input":
            use_input = has_valid_reference_audio
            use_preset = not use_input and voice_preset != "voice_clone"
        elif audio_priority == "force_preset":
            use_preset = voice_preset != "voice_clone"
            use_input = False
        else:  # auto
            if voice_preset != "voice_clone":
                use_preset = True
                use_input = False
            else:
                use_preset = False
                use_input = has_valid_reference_audio
        
        # Use voice preset from dropdown
        if use_preset:
            voice_path = get_voice_preset_path(voice_preset)
            if voice_path and os.path.exists(voice_path):
                try:
                    waveform, sample_rate = torchaudio.load(voice_path)
                    if waveform.dim() == 2:
                        waveform = waveform.unsqueeze(0)
                    audio_for_cloning = {
                        "waveform": waveform.float(),
                        "sample_rate": sample_rate
                    }
                    text_for_cloning = VOICE_PRESETS.get(voice_preset, "")
                    used_voice_info = f"Voice Preset: {voice_preset}"
                except Exception as e:
                    print(f"Error loading voice preset {voice_preset}: {e}")
                    audio_for_cloning = None
        
        # Use reference audio from input
        elif use_input:
            audio_for_cloning = reference_audio
            text_for_cloning = reference_text.strip()
            used_voice_info = "Reference Audio Input"
            
            if not text_for_cloning:
                text_for_cloning = "Reference audio for voice cloning."
        
        # Add reference audio for voice cloning if available
        if audio_for_cloning is not None:
            try:
                audio_base64 = self._audio_to_base64(audio_for_cloning)
                if audio_base64:
                    # Add user message with reference text
                    if text_for_cloning:
                        messages.append(Message(role="system", content=text_for_cloning))
                    else:
                        messages.append(Message(role="system", content="Reference audio for voice cloning."))
                    
                    # Add assistant message with audio content
                    audio_content = AudioContent(raw_audio=audio_base64, audio_url="")
                    messages.append(Message(role="assistant", content=[audio_content]))
                else:
                    used_voice_info = "Audio encoding failed - using basic TTS"
            except Exception as e:
                print(f"Error in audio processing: {e}")
                used_voice_info = f"Audio processing error: {str(e)}"
        
        # Add the main user message
        messages.append(Message(role="user", content=prompt))
        
        print(f"Generating audio with HiggsAudio...")
        start_time = time.time()
        
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
        
        generation_time = time.time() - start_time
        print(f"Audio generated in {generation_time:.2f} seconds")
        
        # Convert to ComfyUI format
        if hasattr(output, 'audio') and hasattr(output, 'sampling_rate'):
            audio_np = output.audio
            if len(audio_np.shape) == 1:
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).float()
            elif len(audio_np.shape) == 2:
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()
            else:
                audio_tensor = torch.from_numpy(audio_np).float()
            
            comfy_audio = {
                "waveform": audio_tensor,
                "sample_rate": output.sampling_rate
            }
            return (comfy_audio, used_voice_info)
        else:
            raise ValueError("Invalid audio output from HiggsAudio engine")
    
    def _audio_to_base64(self, comfy_audio):
        """Convert ComfyUI audio format to base64 string."""
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