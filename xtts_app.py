"""
Coqui XTTS v2 - Fast TTS with Voice Cloning
Gradio Web UI for Windows with CUDA support
"""

import os

# Accept Coqui TOS automatically
os.environ["COQUI_TOS_AGREED"] = "1"

import time
import numpy as np
import soundfile as sf
import torch
import torchaudio
import gradio as gr

# Force torchaudio to use soundfile backend (avoids torchcodec issues on Windows)
torchaudio.set_audio_backend("soundfile")

# Fix for PyTorch 2.6+ security change - patch torch.load before importing TTS
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from TTS.api import TTS

# Configuration
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xtts_output")
CHUNKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xtts_chunks")

# Create directories
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Global model cache
_tts_model = None
_cancel_requested = False


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    """Load XTTS v2 model (lazy loading)."""
    global _tts_model
    if _tts_model is None:
        log("Loading XTTS v2 model...")
        start = time.time()
        device = get_device()
        log(f"Using device: {device}")

        # Load XTTS v2 model
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        log(f"Model loaded in {time.time() - start:.1f}s")
    return _tts_model


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


def _split_long_sentence(sentence: str, max_sentence: int = 240) -> list[str]:
    """
    Split a sentence that exceeds max_sentence chars at natural break points
    (semicolons, colons, em dashes, commas).

    XTTS v2 has a hard limit of 402 BPE tokens per sentence (assertion crash
    if exceeded). For English, ~250 characters maps to that token limit.
    We use 240 as a safety margin. Each fragment is terminated with a period
    so XTTS treats them as separate sentences.
    """
    if len(sentence) <= max_sentence:
        return [sentence]

    # Try splitting at these delimiters in order of preference
    delimiters = ['; ', ': ', '—', ' — ', ', ']

    for delim in delimiters:
        if delim in sentence:
            parts = sentence.split(delim)
            result = []
            current = parts[0]
            for part in parts[1:]:
                candidate = current + delim + part
                if len(candidate) <= max_sentence:
                    current = candidate
                else:
                    if current.strip():
                        # Ensure fragment ends with a period so XTTS sees it as a sentence
                        frag = current.strip().rstrip('.,;:!?')
                        result.append(frag + '.')
                    current = part
            if current.strip():
                # Keep original ending punctuation on the last fragment
                result.append(current.strip())

            # Check if all parts are now under the limit
            if all(len(r) <= max_sentence for r in result):
                return result

    # Last resort: split at the nearest space before the limit
    result = []
    while len(sentence) > max_sentence:
        split_at = sentence.rfind(' ', 0, max_sentence)
        if split_at == -1:
            split_at = max_sentence
        frag = sentence[:split_at].strip().rstrip('.,;:!?')
        result.append(frag + '.')
        sentence = sentence[split_at:].strip()
    if sentence:
        result.append(sentence)
    return result


def _split_into_chunks(text: str, max_chars: int = 1500) -> list[str]:
    """
    Split text into chunks by paragraph breaks (hard returns only).
    Combines short paragraphs to stay under max_chars.
    Long sentences are pre-split to stay under the XTTS 250-char/402-token
    hard sentence limit.
    """
    import re

    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    log(f"Found {len(paragraphs)} paragraphs, combining into chunks of max {max_chars} chars")

    # Pre-split long sentences within each paragraph
    processed_paragraphs = []
    for para in paragraphs:
        # Split paragraph into sentences (on . ! ? followed by space or end)
        sentences = re.split(r'(?<=[.!?])\s+', para)
        new_sentences = []
        for sent in sentences:
            new_sentences.extend(_split_long_sentence(sent))
        processed_paragraphs.append(' '.join(new_sentences))

    chunks = []
    current_chunk = ""

    for para in processed_paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {para}".strip() if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    for i, c in enumerate(chunks):
        log(f"  Chunk {i+1}: {len(c)} chars - {c[:60]}...")

    return chunks


def cancel_generation():
    """Cancel ongoing generation."""
    global _cancel_requested
    _cancel_requested = True
    log("Cancel requested")
    return "Cancelling..."


def generate_speech(
    text: str,
    voice_file: str,
    language: str,
    speed: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    voice_seed: int,
    save_chunks: bool,
    progress=gr.Progress()
):
    """Generate speech from text using voice cloning."""
    global _cancel_requested
    _cancel_requested = False

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
        tts = load_model()

        seed_str = str(int(voice_seed)) if voice_seed >= 0 else "off"
        log(f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}, Rep penalty: {repetition_penalty}, Seed: {seed_str}")

        # Split text into chunks
        chunks = _split_into_chunks(text)
        log(f"Text split into {len(chunks)} chunks")

        # Clear old chunks if saving
        if save_chunks:
            for f in os.listdir(CHUNKS_DIR):
                if f.startswith("chunk_") and f.endswith(".wav"):
                    os.remove(os.path.join(CHUNKS_DIR, f))

        all_audio = []
        sample_rate = 24000  # XTTS default sample rate

        cancelled = False
        for i, chunk in enumerate(chunks):
            if _cancel_requested:
                log("Generation cancelled by user")
                cancelled = True
                break

            progress((i + 0.5) / len(chunks), desc=f"Generating chunk {i+1}/{len(chunks)}...")
            log(f"Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")

            start = time.time()

            # Set seed before each chunk for voice consistency
            if voice_seed >= 0:
                torch.manual_seed(voice_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(voice_seed)

            # Generate audio for this chunk
            wav = tts.tts(
                text=chunk,
                speaker_wav=voice_path,
                language=language,
                speed=speed,
                temperature=temperature,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
            )

            elapsed = time.time() - start
            log(f"Chunk {i+1} generated in {elapsed:.1f}s")

            # Convert to numpy array
            audio_chunk = np.array(wav, dtype=np.float32)
            all_audio.append(audio_chunk)

            # Add silence between chunks (0.7s paragraph pause)
            if i < len(chunks) - 1:
                silence = np.zeros(int(sample_rate * 0.7), dtype=np.float32)
                all_audio.append(silence)

            # Save chunk if requested
            if save_chunks:
                chunk_path = os.path.join(CHUNKS_DIR, f"chunk_{i+1:03d}.wav")
                sf.write(chunk_path, audio_chunk, sample_rate)
                log(f"Saved chunk to {chunk_path}")

        if not all_audio:
            return None, "Cancelled before any audio was generated."

        progress(1.0, desc="Combining audio...")

        # Combine all chunks
        combined_audio = np.concatenate(all_audio)

        # Save final output
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"xtts_output_{timestamp}.wav")
        sf.write(output_path, combined_audio, sample_rate)

        # Convert to int16 for Gradio playback (avoids float32 conversion warning)
        combined_audio = (combined_audio * 32767).astype(np.int16)
        log(f"Saved final audio to {output_path}")

        completed = len(all_audio) // 2 + 1 if len(all_audio) > 1 else len(all_audio)
        if cancelled:
            status = f"Cancelled after {completed} of {len(chunks)} chunks. Partial output saved to {output_path}"
        else:
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
        # Get the file name
        if hasattr(audio_file, 'name'):
            source_path = audio_file.name
        else:
            source_path = audio_file

        filename = os.path.basename(source_path)
        dest_path = os.path.join(VOICES_DIR, filename)

        # Copy file to voices directory
        import shutil
        shutil.copy2(source_path, dest_path)

        log(f"Uploaded voice file: {filename}")

        # Refresh dropdown and select the new file
        voices = get_voice_files()
        return gr.update(choices=voices, value=filename), f"Uploaded: {filename}"

    except Exception as e:
        log(f"Upload error: {e}")
        return gr.update(), f"Upload error: {str(e)}"


# Build Gradio UI
def create_ui():
    with gr.Blocks(title="XTTS v2 - Voice Cloning TTS") as app:
        gr.Markdown("# XTTS v2 - Fast Text-to-Speech with Voice Cloning")
        gr.Markdown("Clone any voice from a short audio sample (3-10 seconds recommended)")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text to speak",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=10
                )

                with gr.Row():
                    language = gr.Dropdown(
                        label="Language",
                        choices=[
                            ("English", "en"),
                            ("Spanish", "es"),
                            ("French", "fr"),
                            ("German", "de"),
                            ("Italian", "it"),
                            ("Portuguese", "pt"),
                            ("Polish", "pl"),
                            ("Turkish", "tr"),
                            ("Russian", "ru"),
                            ("Dutch", "nl"),
                            ("Czech", "cs"),
                            ("Arabic", "ar"),
                            ("Chinese", "zh-cn"),
                            ("Japanese", "ja"),
                            ("Korean", "ko"),
                            ("Hungarian", "hu"),
                        ],
                        value="en"
                    )
                    speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )

                with gr.Accordion("Quality Settings", open=False):
                    temperature = gr.Slider(
                        label="Temperature (lower=consistent, higher=expressive)",
                        minimum=0.1, maximum=1.5, value=0.75, step=0.05
                    )
                    top_p = gr.Slider(
                        label="Top-p (nucleus sampling)",
                        minimum=0.1, maximum=1.0, value=0.85, step=0.05
                    )
                    top_k = gr.Slider(
                        label="Top-k (token choices)",
                        minimum=1, maximum=100, value=50, step=1
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition penalty (prevents stuttering)",
                        minimum=1.0, maximum=20.0, value=10.0, step=0.5
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

                with gr.Row():
                    generate_btn = gr.Button("Generate Speech", variant="primary", scale=3)
                    cancel_btn = gr.Button("Cancel", variant="stop", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### Voice Reference")

                voice_dropdown = gr.Dropdown(
                    label="Select voice file",
                    choices=get_voice_files(),
                    value=get_voice_files()[0] if get_voice_files() else None
                )

                refresh_btn = gr.Button("Refresh voices")

                gr.Markdown("---")
                gr.Markdown("### Upload new voice")

                voice_upload = gr.Audio(
                    label="Upload voice sample (3-10 sec WAV/MP3)",
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
        - **Voice samples**: Place WAV/MP3 files in the `voices/` folder, or upload above
        - **Best quality**: Use 3-10 second clear voice samples without background noise
        - **Chunks**: Audio is saved to `xtts_chunks/` folder as it generates
        - **Output**: Final combined audio saved to `xtts_output/` folder
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

        # Inputs to disable during generation
        input_components = [text_input, voice_dropdown, language, speed, generate_btn]

        def disable_inputs():
            return [gr.update(interactive=False)] * len(input_components)

        def enable_inputs():
            return [gr.update(interactive=True)] * len(input_components)

        generate_event = generate_btn.click(
            fn=disable_inputs,
            outputs=input_components
        ).then(
            fn=generate_speech,
            inputs=[text_input, voice_dropdown, language, speed, temperature, top_p, top_k, repetition_penalty, voice_seed, save_chunks],
            outputs=[audio_output, status_output]
        ).then(
            fn=enable_inputs,
            outputs=input_components
        )

        cancel_btn.click(
            fn=cancel_generation,
            outputs=[status_output]
        )

    return app


if __name__ == "__main__":
    log("Starting XTTS v2 Web UI...")
    log(f"Voices directory: {VOICES_DIR}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Chunks directory: {CHUNKS_DIR}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    app = create_ui()
    app.launch(share=False)
