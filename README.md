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

**Casual/nostalgic (the original):**

_I walked through the garden yesterday, and you know what I noticed? The roses were already blooming. It surprised me, honestly. But then again, spring has been warmer than usual this year. Don’t you think?_

**Warm/reflective:**

_The thing about old bookstores is the smell. You walk in and it hits you immediately — dust, paper, maybe a little coffee from the back. I found a first edition there once, just sitting on a shelf between two paperbacks. Nobody had noticed it. Can you imagine?_

**Steady/explanatory:**

_So the way it works is actually simpler than people think. You take the readings in the morning, compare them to the baseline, and if anything shifts more than two standard deviations, you flag it. Most days, nothing happens. But when it does, you want to be paying attention._

**Quiet/intimate:**

_I almost didn’t go. I was tired, it was raining, and I’d already talked myself out of it twice. But something made me grab my coat and walk out the door anyway. And I’m glad I did. Sometimes the things you almost skip turn out to be the ones you remember._

**Energetic/storytelling:**

_Okay, so picture this. We’re standing at the top of the trail, the wind is absolutely howling, and my brother turns to me and says, completely straight-faced, “I think we should run it.” And before I can say anything, he’s gone. Just — gone. Down the mountain like a lunatic._

**Dramatic/narrative:**

_Dr. Elena Vasquez had spent three decades studying ancient Mesopotamian architecture, but nothing in her archaeological career had prepared her for this. “The geometry is unequivocally deliberate,” she whispered. “Someone engineered a structure that our civilization shouldn’t have been capable of.”_

## Audiobook Narration Voice Cloning Snippets (~10s)

The samples above work well for conversational and character voices, but voice cloning for audiobook narration benefits from a dedicated reference clip. A good ~10s snippet should capture the neutral center of a narrator’s voice — steady pacing (~150 wpm), warm tone, clear articulation, and natural micro-pauses at clause boundaries. Avoid dialogue, exclamations, casual openers (“So...”), or second-person address, as these bias the cloned voice away from the settled, literary register that generalizes across fiction genres.

**Literary/atmospheric:**

_The house at the end of Harlow Street had been empty for eleven years. Its windows were dark, its garden overgrown, and the mailbox still bore a name no one in the neighborhood recognized. But on the first morning of October, someone turned the lights on._

**Character-grounded, steady:**

_She had a habit of arriving early and sitting near the back, where she could watch the room fill without being noticed. It was a small thing, but it told you everything about how she moved through the world — carefully, deliberately, always one step ahead._

**Measured, reflective:**

_There are places that hold memory the way old wood holds warmth. You walk in and feel it immediately — not nostalgia, exactly, but something close. A kind of stillness that tells you this room has seen a hundred quiet evenings just like this one._

**Forward momentum, plot-like:**

_The letter arrived on a Tuesday, folded twice and sealed with wax. Inside was a single paragraph, unsigned, written in a hand she didn’t recognize. She read it three times before setting it down. Then she picked up the phone and canceled everything._

---

_The contents of this repository represent my viewpoints and not those of my past or current employers, including Amazon Web Services (AWS). All third-party libraries, modules, plugins, and SDKs are the property of their respective owners._

