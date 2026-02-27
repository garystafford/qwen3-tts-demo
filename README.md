# High‑Quality Text-to-Speech with Qwen3 TTS Open‑Weight Models

Build a TTS pipeline with voice cloning, voice design, and broadcast-ready post-processing, entirely from open-weight models on consumer-grade hardware.

## Install Instructions (Windows)

### Hugging Face

```powershell
# Install Hugging Face Hub library with XET support
pip install huggingface_hub[hf_xet]

# Set your HuggingFace token if needed for gated models
$env:HF_TOKEN = "hf_your_token_here"  # PowerShell
# or in cmd:
# set HF_TOKEN=hf_your_token_here
```

### Qwen3 TTS

```powershell
# Install Python 3.12
winget install Python.Python.3.12

# Create and activate venv
py -3.12 -m venv .venv-qwen
.\.venv-qwen\Scripts\Activate.ps1  # PowerShell
# .\.venv-qwen\Scripts\activate    # cmd

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp312-cp312-win_amd64.whl

# Run
python qwen3_app.py
```

### Coqui XTTS v2

```powershell
# Install Python 3.11 (XTTS has compatibility issues with 3.12+)
winget install Python.Python.3.11

# Create and activate venv
py -3.11 -m venv .venv-xtts
.\.venv-xtts\Scripts\Activate.ps1  # PowerShell
# .\.venv-xtts\Scripts\activate    # cmd

# Install dependencies
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-xtts.txt

# Run
python xtts_app.py
```

### F5-TTS

```powershell
# Install Python 3.12 (separate venv from Qwen due to dependency conflicts)
winget install Python.Python.3.12

# Create and activate venv
py -3.12 -m venv .venv-f5
.\.venv-f5\Scripts\Activate.ps1  # PowerShell
# .\.venv-f5\Scripts\activate    # cmd

# Install dependencies
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-f5.txt
pip install https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp312-cp312-win_amd64.whl

# Run
python f5_app.py
```

## Audio Samples

```text
Dr. Elena Vasquez had spent three decades studying ancient Mesopotamian architecture, but nothing in her archaeological career had prepared her for this. “The geometry is unequivocally deliberate,” she whispered. “Someone engineered a structure that our civilization shouldn’t have been capable of.”

“I walked through the garden yesterday, and you know what I noticed? The roses were already blooming. It surprised me, honestly. But then again, spring has been warmer than usual this year. Don’t you think?”
```
