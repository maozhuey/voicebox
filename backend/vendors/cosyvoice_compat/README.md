# CosyVoice Qwen2 compatibility source

`modeling_qwen2_440.py` is the Apache-2.0 licensed Qwen2 implementation from
Hugging Face Transformers 4.40.1. The Fun-CosyVoice3-0.5B-2512 checkpoint
records that exact Transformers version in `CosyVoice-BlankEN/config.json`.

Source:
`transformers/models/qwen2/modeling_qwen2.py` from the official
`transformers==4.40.1` wheel.

Voicebox loads this file only for CosyVoice 3. Other engines continue using the
project-wide Transformers installation.
