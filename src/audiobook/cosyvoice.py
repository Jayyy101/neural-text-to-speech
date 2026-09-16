"""Reusable CosyVoice3 zero-shot adapter and shared synthesis primitives."""

import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import time
import wave


PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
SETTLED_SETTINGS = {
    "load_trt": False,
    "load_vllm": False,
    "fp16": False,
    "stream": False,
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def normalize_prompt_transcript(value):
    transcript = value.strip()
    if transcript.startswith(PROMPT_PREFIX):
        transcript = transcript[len(PROMPT_PREFIX):].strip()
    if not transcript or "<|endofprompt|>" in transcript:
        raise ValueError(
            "Provide a non-empty reference transcript without embedded prompt delimiters."
        )
    return transcript


def load_cosyvoice_runtime(cosyvoice_root):
    """Import model dependencies only when real generation is initialized."""
    import torch
    import torchaudio

    matcha_path = str(cosyvoice_root / "third_party/Matcha-TTS")
    root_path = str(cosyvoice_root)
    for path in (matcha_path, root_path):
        if path not in sys.path:
            sys.path.insert(0, path)
    from cosyvoice.cli.cosyvoice import AutoModel

    return torch, torchaudio, AutoModel


def create_cosyvoice_model(auto_model, model_dir, settings):
    return auto_model(
        model_dir=str(model_dir),
        load_trt=settings["load_trt"],
        load_vllm=settings["load_vllm"],
        fp16=settings["fp16"],
    )


def infer_zero_shot(model, torch, text, prompt_text, prompt_wav, stream=False):
    """Generate one coherent passage and join only CosyVoice's internal chunks."""
    chunks = []
    for output in model.inference_zero_shot(
        text, prompt_text, str(prompt_wav), stream=stream
    ):
        chunks.append(output["tts_speech"])
    if not chunks:
        raise RuntimeError("CosyVoice yielded no audio chunks.")
    return torch.cat(chunks, dim=1), len(chunks)


def set_cosyvoice_random_seed(seed):
    """Apply CosyVoice's process-wide RNG seed after model initialization."""
    from cosyvoice.utils.common import set_all_random_seed

    set_all_random_seed(seed)


def write_pcm16_wav(torchaudio, output_path, speech, sample_rate):
    torchaudio.save(
        str(output_path),
        speech.cpu(),
        sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )


def wav_info(path, expected_rate):
    """Validate a complete, nonempty mono PCM16 WAV at the expected rate."""
    with wave.open(str(path), "rb") as audio:
        frames = audio.getnframes()
        rate = audio.getframerate()
        if frames <= 0 or rate <= 0:
            raise ValueError("Synthesis returned an empty WAV.")
        if (rate != expected_rate or audio.getnchannels() != 1
                or audio.getsampwidth() != 2 or audio.getcomptype() != "NONE"):
            raise ValueError("Expected mono PCM16 WAV at the model sample rate.")
        remaining = frames
        while remaining:
            count = min(remaining, 65536)
            if len(audio.readframes(count)) != count * 2:
                raise ValueError("Synthesis returned a truncated WAV.")
            remaining -= count
        return {
            "sample_rate_hz": rate,
            "channels": audio.getnchannels(),
            "sample_width_bytes": audio.getsampwidth(),
            "frames": frames,
            "duration_seconds": frames / rate,
        }


class CosyVoiceAdapter:
    """Own one initialized CosyVoice3 model for a production CLI invocation."""

    def __init__(self, cosyvoice_root, model_dir, prompt_wav, prompt_text_file):
        self.cosyvoice_root = Path(cosyvoice_root).expanduser().resolve()
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.prompt_wav = Path(prompt_wav).expanduser().resolve()
        self.prompt_text_file = Path(prompt_text_file).expanduser().resolve()
        self.settings = dict(SETTLED_SETTINGS)
        self._model = None
        self._torch = None
        self._torchaudio = None
        self._prompt_text = None
        self._metadata = None

    def configuration(self):
        return {
            "backend": "cosyvoice3_zero_shot",
            "cosyvoice_repo": str(self.cosyvoice_root),
            "model_dir": str(self.model_dir),
            "prompt_wav": str(self.prompt_wav),
            "prompt_transcript_file": str(self.prompt_text_file),
            "settings": dict(self.settings),
        }

    def initialize(self):
        if self._model is not None:
            return dict(self._metadata)

        model_config = self.model_dir / "cosyvoice3.yaml"
        for required in (
            self.cosyvoice_root, self.model_dir, model_config,
            self.prompt_wav, self.prompt_text_file,
        ):
            if not required.exists():
                raise FileNotFoundError(required)

        transcript = normalize_prompt_transcript(
            self.prompt_text_file.read_text(encoding="utf-8-sig")
        )
        prompt_text = PROMPT_PREFIX + transcript
        import_started = time.perf_counter()
        torch, torchaudio, auto_model = load_cosyvoice_runtime(self.cosyvoice_root)
        import_seconds = time.perf_counter() - import_started
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Run this from the cosyvoice-b WSL environment."
            )

        load_started = time.perf_counter()
        model = create_cosyvoice_model(auto_model, self.model_dir, self.settings)
        model_load_seconds = time.perf_counter() - load_started
        self._torch = torch
        self._torchaudio = torchaudio
        self._model = model
        self._prompt_text = prompt_text
        self._metadata = {
            **self.configuration(),
            "cosyvoice_git_head": git_head(self.cosyvoice_root),
            "model_config_sha256": file_sha256(model_config),
            "prompt_wav_sha256": file_sha256(self.prompt_wav),
            "prompt_transcript": transcript,
            "prompt_transcript_sha256": hashlib.sha256(
                transcript.encode("utf-8")
            ).hexdigest(),
            "prompt_transcript_file_sha256": file_sha256(self.prompt_text_file),
            "sample_rate_hz": model.sample_rate,
            "runtime": {
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "cuda_available": True,
                "cuda_build": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(0),
            },
            "import_seconds": import_seconds,
            "model_load_seconds": model_load_seconds,
        }
        return dict(self._metadata)

    def generate_scene(self, text, output_path, seed=None):
        if self._model is None:
            raise RuntimeError("CosyVoice adapter must be initialized before generation.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Scene narration text must not be empty.")
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(output_path)

        torch = self._torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("Seed must be an integer from 0 through 4294967295.")
            set_cosyvoice_random_seed(seed)
        speech, chunk_count = infer_zero_shot(
            self._model, torch, text, self._prompt_text, self.prompt_wav,
            stream=self.settings["stream"],
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        write_pcm16_wav(
            self._torchaudio, output_path, speech, self._model.sample_rate
        )
        audio = wav_info(output_path, self._model.sample_rate)
        return {
            "cosyvoice_chunks": chunk_count,
            "inference_seconds": inference_seconds,
            "rtf": inference_seconds / audio["duration_seconds"],
            "peak_torch_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
        }
