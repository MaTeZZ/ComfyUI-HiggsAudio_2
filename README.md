# ComfyUI-HiggsAudio_Wrapper

A comprehensive ComfyUI wrapper for HiggsAudio v2, enabling high-quality text-to-speech generation with advanced voice cloning capabilities.

<img width="2619" height="1468" alt="image" src="https://github.com/user-attachments/assets/7cfd3e77-3481-43cc-a821-fc28837fca29" />


## Features

- **High-Quality Audio Generation**: Leverages the powerful HiggsAudio v2 3B parameter model
- **Voice Cloning**: Clone voices using reference audio or built-in voice presets
- **Multiple Voice Presets**: Includes pre-configured voices (belinda, en_woman, en_man, etc.)
- **Flexible Audio Prioritization**: Control whether to use voice presets or custom reference audio
- **Customizable System Prompts**: Fine-tune audio generation with scene descriptions and style control
- **GPU Acceleration**: Supports CUDA for faster generation
- **ComfyUI Integration**: Seamless integration with ComfyUI workflows

## Installation

### Prerequisites

- Python 3.8+
- ComfyUI
- CUDA-compatible GPU (recommended)

### ComfyUI Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI-HiggsAudio_Wrapper.git
```
### Install Dependencies

```bash
pip install -r requirements.txt
```

2. Restart ComfyUI

3. The nodes will appear under the "Higgs Audio" category

## Usage

### Basic Workflow

The wrapper provides several nodes that can be chained together:

1. **Load Higgs Audio Model** - Loads the generation model
2. **Load Higgs Audio Tokenizer** - Loads the audio tokenizer
3. **Load Higgs Audio System Prompt** - Configures generation style
4. **Load Higgs Audio Prompt** - Sets the text to convert to speech
5. **Higgs Audio Generator** - Performs the actual audio generation

### Voice Cloning Options

#### Using Voice Presets

The wrapper includes several built-in voice presets:
- `belinda` - Female voice
- `en_woman` - English female voice
- `en_man` - English male voice
- `mabel` - Alternative female voice
- `vex` - Character voice
- `chadwick` - Male voice
- `broom_salesman` - Character voice
- `zh_man_sichuan` - Chinese male voice (Sichuan dialect)
- `voice_clone` - Use custom reference audio

#### Using Custom Reference Audio

1. Set voice preset to `voice_clone`
2. Connect reference audio to the `reference_audio` input
3. Optionally provide reference text that describes the audio

#### Audio Priority Settings

Control which audio source takes precedence:

- **`auto`** (default) - Uses voice preset if selected, otherwise reference audio
- **`preset_dropdown`** - Always prioritizes dropdown selection over reference audio
- **`reference_input`** - Always prioritizes reference audio over dropdown
- **`force_preset`** - Forces use of preset, ignoring reference audio completely

## Configuration

### What Actually Affects Audio Quality

**Important**: System prompts and scene descriptions have minimal effect on HiggsAudio output. Focus on these factors that actually work:

#### Voice Quality Control
- **Reference Audio**: High-quality voice samples (24kHz+) with clear articulation
- **Voice Presets**: Different presets have distinct characteristics - test to find the best fit
- **Reference Text**: Clear, well-punctuated text that matches the reference audio

#### System Prompt (Minimal Impact)
Keep system prompts simple since complex scene descriptions are largely ignored:
```
Generate audio following instruction.
```

### Generation Parameters

- **max_new_tokens** (128-4096): Controls audio length and pacing
- **temperature** (0.0-2.0): Controls voice consistency (0.8 = more stable, 1.2 = more varied)
- **top_p** (0.1-1.0): Affects pronunciation variation (0.9-0.95 recommended)
- **top_k** (-1-100): Fine-tunes voice characteristics (50 = default)
- **device**: auto/cuda/cpu (auto = recommended)

## File Structure

```
ComfyUI-HiggsAudio_Wrapper/
├── __init__.py                 # Node registration
├── nodes.py                    # Main node implementations
├── requirements.txt            # Python dependencies
├── voice_examples/             # Voice preset files
│   ├── config.json            # Voice preset configuration
│   ├── en_woman.wav           # Female English voice
│   ├── en_man.wav             # Male English voice
│   └── ...                    # Other voice presets
└── boson_multimodal/          # HiggsAudio engine
    └── ...
```

## Realistic Expectations

### What HiggsAudio Does Well
- **Voice Cloning**: Excellent at replicating voice characteristics from reference audio
- **Speech Quality**: Generates natural-sounding speech with good pronunciation
- **Multiple Voices**: Built-in voice presets for different character types
- **Consistency**: Maintains voice characteristics across longer text

### Current Limitations
- **Scene Control**: System prompts for acoustic environments (reverb, background sounds) have minimal effect
- **Emotional Control**: Limited ability to control emotional expression through text prompts
- **Background Audio**: Cannot generate environmental sounds or music
- **Real-time**: Requires processing time, not suitable for real-time applications

### Best Use Cases
- Voice-over generation with consistent character voices
- Audiobook narration with cloned voices
- Character voices for games or animations
- Text-to-speech with specific voice characteristics

For acoustic effects like reverb or background sounds, consider post-processing with audio editing software.

## Troubleshooting

### Common Issues

#### Poor Audio Quality
- Use higher quality reference audio (24kHz+ recommended)
- Try different voice presets to find the best match
- Adjust temperature (0.8 for stability, 1.2 for variation)
- Ensure reference text matches the reference audio content

#### "audio_base64 is None" Error
- Ensure reference audio is properly formatted
- Check that voice preset files exist in `voice_examples/`
- Verify audio file is not corrupted

#### Inconsistent Voice Output
- Lower the temperature parameter (try 0.8)
- Use higher quality reference audio
- Ensure reference audio has consistent background noise levels

#### CUDA Out of Memory
- Reduce `max_new_tokens`
- Use `device: cpu` instead of auto/cuda
- Close other GPU-intensive applications

#### Model Loading Issues
- Ensure stable internet connection for model download
- Check available disk space (models are several GB)
- Verify transformers version compatibility

### Performance Tips

1. **First Run**: Model downloading may take time
2. **GPU Memory**: 8GB+ VRAM recommended for optimal performance
3. **Caching**: Models are cached after first load for faster subsequent runs
4. **Voice Quality**: Use high-quality reference audio for best results
5. **Parameter Tuning**: Lower temperature (0.8) for consistent voice, higher (1.2) for variation
6. **Text Formatting**: Use proper punctuation for natural speech rhythm

## API Reference

### HiggsAudio Node Inputs

#### Required
- `MODEL_PATH`: Path to HiggsAudio model
- `AUDIO_TOKENIZER_PATH`: Path to audio tokenizer
- `system_prompt`: System prompt for generation control
- `prompt`: Text to convert to speech
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `device`: Computation device

#### Optional
- `voice_preset`: Voice preset selection
- `reference_audio`: Custom reference audio
- `reference_text`: Text corresponding to reference audio
- `audio_priority`: Audio source prioritization

### Output

- `output`: Generated audio in ComfyUI format
- `used_voice_info`: Information about which voice source was used

## Requirements

See `requirements.txt` for complete list:

- torch==2.5.1
- torchaudio==2.5.1
- transformers>=4.45.1,<4.47.0
- librosa
- And others...

## License

[Add your license information here]

### Third-Party Licenses

The `boson_multimodal/audio_processing/` directory contains code derived from third-party repositories, primarily from xcodec. Please see the `LICENSE` in that directory for complete attribution and licensing information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Provide detailed error messages and system information

## Acknowledgments

- HiggsAudio team for the underlying model
- ComfyUI community for the framework
- Contributors and testers

---

**Note**: This wrapper requires significant computational resources. A CUDA-compatible GPU with 8GB+ VRAM is recommended for optimal performance.
