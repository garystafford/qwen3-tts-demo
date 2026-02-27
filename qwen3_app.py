"""
Qwen3 TTS - Text-to-Speech with Custom Voices, Voice Design, and Voice Cloning
Gradio Web UI for Windows with CUDA support

Supports three modes:
  - CustomVoice: Predefined speakers with style instructions
  - VoiceDesign: Create voices from natural-language descriptions
  - Base (Voice Clone): Clone any voice from a reference audio sample
"""

import os
import re
import time

import numpy as np
import soundfile as sf
import torch
import gradio as gr

# ── Configuration ──────────────────────────────────────────────────────────────
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_output")
CHUNKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_chunks")

# ── Generation Constants ──────────────────────────────────────────────────────
MAX_CHUNK_CHARS = 3_200         # Max characters per chunk (paragraph-based fallback)
MAX_SECTION_CHARS = 5_000       # Max characters per section chunk (when section headers detected)
MAX_NEW_TOKENS = 4_096          # Max output tokens per chunk (12Hz tokenizer hard limit: 8000)
MAX_SENTENCE_CHARS = 240        # Sentences longer than this are split at natural breaks
INTER_CHUNK_SILENCE = 0.7       # Seconds of silence inserted between chunks

# Create directories
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Global model cache
_tts_model = None
_current_model_type = None
_cancel_requested = False

MODEL_CHOICES = {
    "1.7B CustomVoice (predefined speakers)": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "1.7B VoiceDesign (create voices from descriptions)": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "1.7B Base (voice cloning)": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B CustomVoice (faster, predefined speakers)": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "0.6B Base (faster, voice cloning)": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}


def _resolve_model_path(repo_id: str) -> str:
    """Resolve a HuggingFace repo ID to its local cache snapshot path."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == repo_id:
                for rev in repo.revisions:
                    return str(rev.snapshot_path)
    except Exception:
        pass
    # Fallback to repo_id if not found in cache
    return repo_id


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_key: str = "1.7B CustomVoice (predefined speakers)"):
    """Load Qwen3 TTS model (lazy loading, switches if model type changes)."""
    global _tts_model, _current_model_type
    model_id = MODEL_CHOICES[model_key]

    if _tts_model is not None and _current_model_type == model_key:
        return _tts_model

    from qwen_tts import Qwen3TTSModel

    # Unload previous model if switching
    if _tts_model is not None:
        log(f"Unloading previous model ({_current_model_type})...")
        del _tts_model
        _tts_model = None
        torch.cuda.empty_cache()

    log(f"Loading {model_key} model ({model_id})...")
    start = time.time()
    device = get_device()
    log(f"Using device: {device}")

    # Reset VRAM tracking for clean measurement
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    local_path = _resolve_model_path(model_id)
    log(f"Resolved model path: {local_path}")

    # Use flash_attention_2 if available for faster inference
    attn_impl = None
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        log("Using flash_attention_2")
    except ImportError:
        log("Flash attention not available, using default attention")

    _tts_model = Qwen3TTSModel.from_pretrained(
        local_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    _current_model_type = model_key

    log(f"Model loaded in {time.time() - start:.1f}s")

    # Log VRAM usage
    if device == "cuda":
        torch.cuda.synchronize()
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        log(f"VRAM: {vram_allocated:.2f} GB model weights, {vram_reserved:.2f} GB reserved")

    # Log available speakers/languages
    speakers = _tts_model.get_supported_speakers()
    languages = _tts_model.get_supported_languages()
    if speakers:
        log(f"Supported speakers: {speakers}")
    if languages:
        log(f"Supported languages: {languages}")

    return _tts_model


def unload_model():
    """Unload the current model from GPU and free VRAM."""
    global _tts_model, _current_model_type

    if _tts_model is None:
        msg = "No model loaded."
        log(msg)
        return msg

    model_name = _current_model_type
    log(f"Unloading model ({model_name})...")
    del _tts_model
    _tts_model = None
    _current_model_type = None
    torch.cuda.empty_cache()

    import gc
    gc.collect()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        msg = f"Unloaded {model_name}. GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved."
    else:
        msg = f"Unloaded {model_name}."

    log(msg)
    return msg


def get_speakers():
    """Get available speakers (returns defaults if model not loaded yet)."""
    if _tts_model is not None:
        speakers = _tts_model.get_supported_speakers()
        if speakers:
            return sorted(speakers)
    # Known default speakers for CustomVoice 1.7B model
    return ["aiden", "dylan", "eric", "ono_anna", "ryan",
            "serena", "sohee", "uncle_fu", "vivian"]


def get_languages():
    """Get available languages (returns defaults if model not loaded yet)."""
    if _tts_model is not None:
        languages = _tts_model.get_supported_languages()
        if languages:
            return sorted(languages)
    return ["English", "Chinese", "Japanese", "Korean", "French",
            "German", "Spanish", "Italian", "Portuguese", "Russian"]


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


def upload_voice(audio_file):
    """Handle voice file upload."""
    if audio_file is None:
        return gr.update(), "No file uploaded."
    try:
        import shutil
        if hasattr(audio_file, 'name'):
            source_path = audio_file.name
        else:
            source_path = audio_file
        filename = os.path.basename(source_path)
        dest_path = os.path.join(VOICES_DIR, filename)
        shutil.copy2(source_path, dest_path)
        log(f"Uploaded voice file: {filename}")
        voices = get_voice_files()
        return gr.update(choices=voices, value=filename), f"Uploaded: {filename}"
    except Exception as e:
        log(f"Upload error: {e}")
        return gr.update(), f"Upload error: {str(e)}"



def _split_long_sentence(sentence: str, max_sentence: int = MAX_SENTENCE_CHARS) -> list[str]:
    """
    Split a sentence that exceeds max_sentence chars at natural break points
    (semicolons, colons, commas). Each fragment is terminated with
    a period so the TTS engine treats them as separate sentences.
    """
    if len(sentence) <= max_sentence:
        return [sentence]

    delimiters = ['; ', ': ', ' -- ', ', ']

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
                        frag = current.strip().rstrip('.,;:!?')
                        result.append(frag + '.')
                    current = part
            if current.strip():
                result.append(current.strip())

            if all(len(r) <= max_sentence for r in result):
                return result

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


# Section header pattern: "Part 1 –", "Part 2 -", etc.
_SECTION_HEADER_RE = re.compile(r'^\s*Part\s+\d+\s*[–\-]', re.IGNORECASE)


def _pack_paragraphs(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks by paragraph breaks.
    Combines short paragraphs to stay under max_chars.
    Long sentences are pre-split to stay under MAX_SENTENCE_CHARS.
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    processed_paragraphs = []
    for para in paragraphs:
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

    return chunks


def _split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS, use_sections: bool = False) -> list[str]:
    """
    Split text into chunks for TTS generation.

    If use_sections is True and the text contains section headers
    (e.g. "Part 1 – Title"), splits at those boundaries first, then
    sub-splits any oversized sections by paragraph.
    Otherwise uses paragraph-based greedy packing.
    """
    lines = text.split('\n')

    # Find lines that match "Part N – ..."
    section_starts = [i for i, line in enumerate(lines) if _SECTION_HEADER_RE.match(line)]

    if use_sections and section_starts:
        log(f"Found {len(section_starts)} section headers, splitting at section breaks")

        sections = []
        for idx, start in enumerate(section_starts):
            end = section_starts[idx + 1] if idx + 1 < len(section_starts) else len(lines)

            # Include any preamble (title, etc.) with the first section
            section_lines = lines[(0 if idx == 0 else start):end]

            # Strip standalone "..." separator lines
            cleaned = [line for line in section_lines if not re.match(r'^\s*\.{3}\s*$', line)]
            section_text = '\n'.join(cleaned).strip()
            if section_text:
                sections.append(section_text)

        # Sub-split any sections that exceed MAX_SECTION_CHARS
        chunks = []
        for section in sections:
            if len(section) <= MAX_SECTION_CHARS:
                chunks.append(section)
            else:
                log(f"  Section exceeds {MAX_SECTION_CHARS} chars ({len(section)}), sub-splitting by paragraph")
                chunks.extend(_pack_paragraphs(section, max_chars))
    else:
        if not use_sections:
            log(f"Section breaks disabled, using paragraph-based chunking (max {max_chars} chars)")
        else:
            log(f"No section headers found, using paragraph-based chunking (max {max_chars} chars)")
        chunks = _pack_paragraphs(text, max_chars)

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
    model_key: str,
    speaker: str,
    language: str,
    instruct: str,
    voice_design_instruct: str,
    voice_file: str,
    ref_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    voice_seed: int,
    use_section_breaks: bool,
    save_chunks: bool,
    progress=gr.Progress()
):
    """Generate speech from text using Qwen3 TTS."""
    global _cancel_requested
    _cancel_requested = False

    if not text.strip():
        return None, "Please enter some text."

    is_custom = "CustomVoice" in model_key
    is_voice_design = "VoiceDesign" in model_key
    is_clone = not is_custom and not is_voice_design

    if is_custom and not speaker:
        return None, "Please select a speaker."

    if is_voice_design and not voice_design_instruct.strip():
        return None, "Please enter a voice description."

    if is_clone and not voice_file:
        return None, "Please select a voice reference file."

    try:
        total_start = time.time()

        # Log settings
        log(f"--- Generate Speech ---")
        log(f"Model: {model_key}")
        log(f"Language: {language}")
        if is_custom:
            log(f"Speaker: {speaker}")
            log(f"Instruct: {instruct}")
        elif is_voice_design:
            log(f"Voice design: {voice_design_instruct}")
        else:
            log(f"Voice file: {voice_file}")
            log(f"Ref text: {ref_text[:80] + '...' if len(ref_text) > 80 else ref_text}")
        seed_str = str(int(voice_seed)) if voice_seed >= 0 else "off"
        log(f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}, Rep penalty: {repetition_penalty}, Seed: {seed_str}")
        log(f"Text length: {len(text)} chars")

        # Load model
        progress(0, desc="Loading model...")
        tts = load_model(model_key)

        # For voice clone, build the prompt once and reuse across chunks
        voice_clone_prompt = None
        if is_clone:
            voice_path = os.path.join(VOICES_DIR, voice_file)
            if not os.path.exists(voice_path):
                return None, f"Voice file not found: {voice_file}"

            progress(0.05, desc="Building voice clone prompt...")
            log(f"Building voice clone prompt from {voice_file}...")
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=voice_path,
                ref_text=ref_text if ref_text.strip() else None,
                x_vector_only_mode=not bool(ref_text.strip()),
            )
            voice_clone_prompt = prompt_items

        # Split text into chunks
        chunks = _split_into_chunks(text, use_sections=use_section_breaks)
        log(f"Text split into {len(chunks)} chunks")

        # Clear old chunks if saving
        if save_chunks:
            for f in os.listdir(CHUNKS_DIR):
                if f.startswith("chunk_") and f.endswith(".wav"):
                    os.remove(os.path.join(CHUNKS_DIR, f))

        all_audio = []
        sample_rate = None

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

            if is_custom:
                wavs, sr = tts.generate_custom_voice(
                    text=chunk,
                    speaker=speaker,
                    language=language,
                    instruct=instruct if instruct.strip() else None,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            elif is_voice_design:
                wavs, sr = tts.generate_voice_design(
                    text=chunk,
                    language=language,
                    instruct=voice_design_instruct.strip(),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            else:
                wavs, sr = tts.generate_voice_clone(
                    text=chunk,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                    max_new_tokens=MAX_NEW_TOKENS,
                )

            if sample_rate is None:
                sample_rate = sr

            elapsed = time.time() - start
            log(f"Chunk {i+1} generated in {elapsed:.1f}s")

            audio_chunk = wavs[0].astype(np.float32)
            all_audio.append(audio_chunk)

            # Add silence between chunks
            if i < len(chunks) - 1:
                silence = np.zeros(int(sample_rate * INTER_CHUNK_SILENCE), dtype=np.float32)
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
        output_path = os.path.join(OUTPUT_DIR, f"qwen3_output_{timestamp}.wav")
        sf.write(output_path, combined_audio, sample_rate)

        # Convert to int16 for Gradio playback (avoids float32 conversion warning)
        combined_audio = (combined_audio * 32767).astype(np.int16)
        log(f"Saved final audio to {output_path}")

        total_elapsed = time.time() - total_start
        audio_duration = len(np.concatenate([a for a in all_audio if len(a) > sample_rate * 0.5])) / sample_rate
        total_mins = total_elapsed / 60
        audio_mins = audio_duration / 60
        rtf = total_elapsed / audio_duration if audio_duration > 0 else 0
        log(f"Total generation time: {total_mins:.1f}min for {audio_mins:.1f}min of audio (RTF: {rtf:.2f}x)")

        completed = len(all_audio) // 2 + 1 if len(all_audio) > 1 else len(all_audio)
        if cancelled:
            status = f"Cancelled after {completed} of {len(chunks)} chunks. Partial output saved to {output_path}"
        else:
            status = f"Generated {len(chunks)} chunks in {total_mins:.1f}min ({audio_mins:.1f}min audio, RTF {rtf:.2f}x)"
        status += f"\nSaved to {output_path}"
        if save_chunks:
            status += f"\nChunks saved to {CHUNKS_DIR}"

        return (sample_rate, combined_audio), status

    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"


# Build Gradio UI
def create_ui():
    with gr.Blocks(title="Qwen3 TTS") as app:
        gr.Markdown("# Qwen3 TTS - Text-to-Speech")
        gr.Markdown("Generate speech using predefined speakers or clone a voice from a reference audio sample")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text to speak",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=10
                )

                with gr.Row():
                    model_selector = gr.Dropdown(
                        label="Model",
                        choices=list(MODEL_CHOICES.keys()),
                        value="1.7B CustomVoice (predefined speakers)"
                    )
                    language = gr.Dropdown(
                        label="Language",
                        choices=get_languages(),
                        value="English"
                    )
                    unload_btn = gr.Button("Unload Model", variant="secondary", scale=0)

                with gr.Accordion("Quality Settings", open=False):
                    temperature = gr.Slider(
                        label="Temperature",
                        info="Controls randomness of token selection. Lower values (0.3-0.5) produce more predictable, monotone speech. Higher values (0.8-1.2) produce more expressive, varied speech but risk artifacts.",
                        minimum=0.1, maximum=1.5, value=0.7, step=0.05
                    )
                    top_p = gr.Slider(
                        label="Top-p (nucleus sampling)",
                        info="Only considers tokens whose cumulative probability is within this threshold. At 1.0 all tokens are considered. Lower values (0.7-0.9) restrict choices to the most likely tokens, producing cleaner but less varied output.",
                        minimum=0.1, maximum=1.0, value=1.0, step=0.05
                    )
                    top_k = gr.Slider(
                        label="Top-k",
                        info="Limits token selection to the top K most likely candidates at each step. Lower values (10-20) produce more focused output. Higher values (50-100) allow more variety. Works together with top-p.",
                        minimum=1, maximum=100, value=50, step=1
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition penalty",
                        info="Penalizes tokens that have already appeared, reducing stuttering and loops. At 1.0 there is no penalty. Values around 1.05-1.2 gently discourage repetition. Too high (>2.0) can cause unnatural speech.",
                        minimum=1.0, maximum=5.0, value=1.05, step=0.05
                    )
                    voice_seed = gr.Number(
                        label="Voice consistency seed",
                        info="Fixes the random seed before each chunk so the voice stays consistent across chunks. Set to -1 to disable.",
                        value=-1,
                        precision=0
                    )

                use_section_breaks = gr.Checkbox(
                    label="Split at section breaks (Part N –)",
                    info="When enabled, splits text at 'Part N –' headers instead of fixed character limits. Best for long texts with chapter/section markers.",
                    value=False
                )
                save_chunks = gr.Checkbox(
                    label="Save individual chunks (listen while generating)",
                    value=True
                )

                with gr.Row():
                    generate_btn = gr.Button("Generate Speech", variant="primary", scale=3)
                    cancel_btn = gr.Button("Cancel", variant="stop", scale=1)

            with gr.Column(scale=1):
                # --- CustomVoice controls ---
                with gr.Group(visible=True) as custom_voice_group:
                    gr.Markdown("### Speaker")
                    speaker = gr.Dropdown(
                        label="Select speaker",
                        choices=get_speakers(),
                        value="aiden"
                    )
                    instruct = gr.Textbox(
                        label="Voice style instruction",
                        placeholder="e.g. Warm, Captivating, Storyteller...",
                        value="Warm, Captivating, Storyteller, Slight British accent, Male.",
                        lines=2
                    )

                # --- VoiceDesign controls ---
                with gr.Group(visible=False) as voice_design_group:
                    gr.Markdown("### Voice Design")
                    gr.Markdown("Describe the voice you want to create using natural language.")
                    voice_design_instruct = gr.Textbox(
                        label="Voice description",
                        placeholder="e.g. A deep, resonant male voice with a British accent, speaking slowly and authoritatively.",
                        lines=3
                    )

                # --- Voice Clone controls ---
                with gr.Group(visible=False) as voice_clone_group:
                    gr.Markdown("### Voice Reference")
                    voice_dropdown = gr.Dropdown(
                        label="Select voice file",
                        choices=get_voice_files(),
                        value=get_voice_files()[0] if get_voice_files() else None
                    )
                    refresh_btn = gr.Button("Refresh voices")
                    ref_text = gr.Textbox(
                        label="Reference text (transcript of voice sample)",
                        placeholder="Type what is said in the voice sample for better cloning...",
                        lines=2
                    )
                    gr.Markdown("---")
                    gr.Markdown("**Upload new voice**")
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

        gr.Markdown("""
        ---
        ### Tips:
        - **Three modes** — switch with the Model dropdown:
          - *CustomVoice*: uses predefined speakers (Aiden, Serena, etc.) with a style instruction
          - *VoiceDesign*: creates a voice from a natural-language description (no reference audio needed)
          - *Base (voice cloning)*: clones any voice from a WAV/MP3 reference file
        - **Voice files**: place WAV/MP3 samples in `voices/` folder, or upload in Voice Clone mode
        - **Reference text**: when cloning, providing the transcript of your voice sample improves quality
        - **Chunks**: saved to `qwen3_chunks/` as they generate — listen while waiting
        - **Output**: final combined audio saved to `qwen3_output/`
        """)

        # Event handlers — toggle voice panels on model change
        def on_model_switch(model_key):
            is_custom = "CustomVoice" in model_key
            is_design = "VoiceDesign" in model_key
            is_clone = not is_custom and not is_design
            return (
                gr.update(visible=is_custom),   # custom_voice_group
                gr.update(visible=is_design),   # voice_design_group
                gr.update(visible=is_clone),    # voice_clone_group
            )

        model_selector.change(
            fn=on_model_switch,
            inputs=[model_selector],
            outputs=[custom_voice_group, voice_design_group, voice_clone_group]
        )

        unload_btn.click(
            fn=unload_model,
            outputs=[status_output]
        )

        refresh_btn.click(
            fn=refresh_voices,
            outputs=[voice_dropdown]
        )

        upload_btn.click(
            fn=upload_voice,
            inputs=[voice_upload],
            outputs=[voice_dropdown, upload_status]
        )

        input_components = [text_input, model_selector, language, generate_btn]

        def disable_inputs():
            return [gr.update(interactive=False)] * len(input_components)

        def enable_inputs():
            return [gr.update(interactive=True)] * len(input_components)

        generate_btn.click(
            fn=disable_inputs,
            outputs=input_components
        ).then(
            fn=generate_speech,
            inputs=[text_input, model_selector, speaker, language, instruct,
                    voice_design_instruct, voice_dropdown, ref_text,
                    temperature, top_p, top_k, repetition_penalty,
                    voice_seed, use_section_breaks, save_chunks],
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
    log("Starting Qwen3 TTS Web UI...")
    log(f"Voices directory: {VOICES_DIR}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Chunks directory: {CHUNKS_DIR}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    app = create_ui()
    app.launch(share=False)
