"""
F5-TTS - Fast Flow-Matching Text-to-Speech with Voice Cloning
Gradio Web UI for Windows with CUDA support
"""

import os
import time
import numpy as np
import soundfile as sf
import torch
import gradio as gr

# F5-TTS imports
from f5_tts.api import F5TTS

# Configuration
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f5_output")
CHUNKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f5_chunks")

# Create directories
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Global model cache
_f5tts = None


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_f5_model():
    """Load F5-TTS model (lazy loading). Uses F5TTS API for auto-download."""
    global _f5tts

    if _f5tts is None:
        log("Loading F5-TTS model (will auto-download if needed)...")
        start = time.time()
        device = get_device()
        log(f"Using device: {device}")

        _f5tts = F5TTS(model="F5TTS_v1_Base", device=device)

        log(f"Model loaded in {time.time() - start:.1f}s")

    return _f5tts


def get_voice_files():
    """Get list of available voice reference files."""
    if not os.path.exists(VOICES_DIR):
        return []

    voice_files = []
    for f in os.listdir(VOICES_DIR):
        if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            voice_files.append(f)
    return sorted(voice_files)


def refresh_voices():
    """Refresh the voice file dropdown."""
    voices = get_voice_files()
    if voices:
        return gr.update(choices=voices, value=voices[0])
    return gr.update(choices=[], value=None)


def _split_into_chunks(text: str, max_chars: int = 500) -> list[str]:
    """
    Split text into chunks for F5-TTS.

    F5-TTS has a hard 30-second limit per generation (reference + output
    audio combined). The model's internal infer_process() further subdivides
    each chunk dynamically based on reference audio duration. Our 500-char
    split is a coarse first pass to keep chunks at a manageable size.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Handle sentences longer than max_chars
            while len(sentence) > max_chars:
                # Split at comma or space
                split_point = sentence[:max_chars].rfind(', ')
                if split_point == -1:
                    split_point = sentence[:max_chars].rfind(' ')
                if split_point == -1:
                    split_point = max_chars
                chunks.append(sentence[:split_point].strip())
                sentence = sentence[split_point:].strip(' ,')
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text]


def generate_speech(
    text: str,
    voice_file: str,
    ref_text: str,
    speed: float,
    nfe_steps: int,
    voice_seed: int,
    save_chunks: bool,
    progress=gr.Progress()
):
    """Generate speech from text using voice cloning."""
    if not text.strip():
        return None, "Please enter some text."

    if not voice_file:
        return None, "Please select or upload a voice reference file."

    voice_path = os.path.join(VOICES_DIR, voice_file)
    if not os.path.exists(voice_path):
        return None, f"Voice file not found: {voice_file}"

    try:
        # Load model
        progress(0, desc="Loading model...")
        f5tts = load_f5_model()

        seed_str = str(int(voice_seed)) if voice_seed >= 0 else "off"
        log(f"Speed: {speed}, NFE steps: {nfe_steps}, Seed: {seed_str}")

        # Split text into chunks
        chunks = _split_into_chunks(text, max_chars=500)
        log(f"Text split into {len(chunks)} chunks")

        # Clear old chunks if saving
        if save_chunks:
            for f in os.listdir(CHUNKS_DIR):
                if f.startswith("chunk_") and f.endswith(".wav"):
                    os.remove(os.path.join(CHUNKS_DIR, f))

        all_audio = []
        sample_rate = f5tts.target_sample_rate

        for i, chunk in enumerate(chunks):
            progress((i + 0.5) / len(chunks), desc=f"Generating chunk {i+1}/{len(chunks)}...")
            log(f"Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")

            start = time.time()

            # Set seed before each chunk for voice consistency
            if voice_seed >= 0:
                torch.manual_seed(voice_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(voice_seed)

            # Generate audio using F5TTS API
            audio_chunk, sr, _ = f5tts.infer(
                ref_file=voice_path,
                ref_text=ref_text if ref_text.strip() else "",
                gen_text=chunk,
                speed=speed,
                nfe_step=nfe_steps,
            )

            elapsed = time.time() - start
            log(f"Chunk {i+1} generated in {elapsed:.1f}s")

            sample_rate = sr
            all_audio.append(audio_chunk)

            # Save chunk if requested
            if save_chunks:
                chunk_path = os.path.join(CHUNKS_DIR, f"chunk_{i+1:03d}.wav")
                sf.write(chunk_path, audio_chunk, sample_rate)
                log(f"Saved chunk to {chunk_path}")

        progress(1.0, desc="Combining audio...")

        # Combine all chunks
        combined_audio = np.concatenate(all_audio)

        # Save final output
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"f5_output_{timestamp}.wav")
        sf.write(output_path, combined_audio, sample_rate)
        log(f"Saved final audio to {output_path}")

        status = f"Generated {len(chunks)} chunks. Saved to {output_path}"
        if save_chunks:
            status += f"\nChunks saved to {CHUNKS_DIR}"

        return (sample_rate, combined_audio), status

    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"


def upload_voice(audio_file):
    """Handle voice file upload."""
    if audio_file is None:
        return gr.update(), "No file uploaded."

    try:
        if hasattr(audio_file, 'name'):
            source_path = audio_file.name
        else:
            source_path = audio_file

        filename = os.path.basename(source_path)
        dest_path = os.path.join(VOICES_DIR, filename)

        import shutil
        shutil.copy2(source_path, dest_path)

        log(f"Uploaded voice file: {filename}")

        voices = get_voice_files()
        return gr.update(choices=voices, value=filename), f"Uploaded: {filename}"

    except Exception as e:
        log(f"Upload error: {e}")
        return gr.update(), f"Upload error: {str(e)}"


# Build Gradio UI
def create_ui():
    with gr.Blocks(title="F5-TTS - Voice Cloning") as app:
        gr.Markdown("# F5-TTS - Fast Text-to-Speech with Voice Cloning")
        gr.Markdown("High-quality voice cloning using flow-matching (5-10x faster than autoregressive models)")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text to speak",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=10
                )

                with gr.Row():
                    speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    nfe_steps = gr.Slider(
                        label="NFE Steps (quality vs speed)",
                        minimum=8,
                        maximum=64,
                        value=32,
                        step=4,
                        info="Higher = better quality but slower"
                    )
                    voice_seed = gr.Number(
                        label="Voice consistency seed",
                        info="Fixes the random seed before each chunk so the voice stays consistent across chunks. Set to -1 to disable.",
                        value=-1,
                        precision=0
                    )

                save_chunks = gr.Checkbox(
                    label="Save individual chunks (listen while generating)",
                    value=True
                )

                generate_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Voice Reference")

                voice_dropdown = gr.Dropdown(
                    label="Select voice file",
                    choices=get_voice_files(),
                    value=get_voice_files()[0] if get_voice_files() else None
                )

                ref_text_input = gr.Textbox(
                    label="Reference text (optional)",
                    placeholder="Transcription of the voice sample. Leave empty for auto-transcription.",
                    lines=3
                )

                refresh_btn = gr.Button("Refresh voices")

                gr.Markdown("---")
                gr.Markdown("### Upload new voice")

                voice_upload = gr.Audio(
                    label="Upload voice sample (5-15 sec WAV/MP3)",
                    type="filepath"
                )

                upload_btn = gr.Button("Upload to voices folder")
                upload_status = gr.Textbox(label="Upload status", interactive=False)

        gr.Markdown("---")

        with gr.Row():
            audio_output = gr.Audio(label="Generated Speech", type="numpy")
            status_output = gr.Textbox(label="Status", lines=3, interactive=False)

        gr.Markdown(f"""
        ---
        ### Tips:
        - **Voice samples**: 5-15 seconds of clear speech works best
        - **Reference text**: Providing the transcription improves quality and saves VRAM
        - **NFE Steps**: 32 is a good balance; lower (16) for speed, higher (64) for quality
        - **Chunks**: Audio saved to `f5_chunks/` folder as it generates
        - **Output**: Final combined audio saved to `f5_output/` folder
        """)

        # Event handlers
        refresh_btn.click(
            fn=refresh_voices,
            outputs=[voice_dropdown]
        )

        upload_btn.click(
            fn=upload_voice,
            inputs=[voice_upload],
            outputs=[voice_dropdown, upload_status]
        )

        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, voice_dropdown, ref_text_input, speed, nfe_steps, voice_seed, save_chunks],
            outputs=[audio_output, status_output]
        )

    return app


if __name__ == "__main__":
    log("Starting F5-TTS Web UI...")
    log(f"Voices directory: {VOICES_DIR}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Chunks directory: {CHUNKS_DIR}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    app = create_ui()
    app.launch(share=False)
