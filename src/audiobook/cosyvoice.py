"""Reusable CosyVoice3 zero-shot adapter and shared synthesis primitives."""

import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
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
TEXT_PREPROCESSING_POLICY = "remove_u000d_v1"
PINNED_FRONTEND_SPLITTING_SETTINGS = {
    "language": "zh",
    "token_max_n": 80,
    "token_min_n": 60,
    "merge_len": 20,
    "comma_split": False,
    "limits_are_soft": True,
}
VALIDATED_RL_SHA256 = "74d34b01a80c7154670ae75ac372d1b1712c78bceae9f467eb9f1f6f61ec764f"
VALIDATED_FRONTEND_SHA256 = "4db610ce6bfa2a03e808e31232aa150d51de09a1ef7391d9b88b92da0a35146b"


def ensure_rl_model_view(cosyvoice_root, view_dir):
    """Expose the validated RL LLM without changing the installed model."""
    model = Path(cosyvoice_root).expanduser().resolve() / "pretrained_models/Fun-CosyVoice3-0.5B"
    checkpoint = model / "llm.rl.pt"
    if file_sha256(checkpoint) != VALIDATED_RL_SHA256:
        raise RuntimeError("Local CosyVoice3 RL checkpoint differs from validation.")
    view = Path(view_dir).expanduser().resolve()
    view.parent.mkdir(parents=True, exist_ok=True)
    if not view.exists():
        view.mkdir()
        for asset in model.iterdir():
            if asset.name not in {"llm.pt", "llm.rl.pt"}:
                (view / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
        (view / "llm.pt").symlink_to(checkpoint)
    expected = {asset.name for asset in model.iterdir()} - {"llm.pt", "llm.rl.pt"}
    expected.add("llm.pt")
    if {asset.name for asset in view.iterdir()} != expected:
        raise RuntimeError("CosyVoice3 RL model view contains unexpected assets.")
    for name in expected:
        target = checkpoint if name == "llm.pt" else model / name
        link = view / name
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise RuntimeError(f"CosyVoice3 RL model view differs: {name}")
    if file_sha256(view / "llm.pt") != VALIDATED_RL_SHA256:
        raise RuntimeError("CosyVoice3 RL model view does not expose the pinned checkpoint.")
    return view


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


def preprocess_synthesis_text(source_text):
    """Remove only carriage returns at the CosyVoice synthesis boundary."""
    synthesis_text = source_text.replace("\r", "")
    return synthesis_text, {
        "policy": TEXT_PREPROCESSING_POLICY,
        "input_text": source_text,
        "input_text_sha256": hashlib.sha256(
            source_text.encode("utf-8")
        ).hexdigest(),
        "output_text": synthesis_text,
        "output_text_sha256": hashlib.sha256(
            synthesis_text.encode("utf-8")
        ).hexdigest(),
        "removed_cr_count": source_text.count("\r"),
        "changed": synthesis_text != source_text,
    }


def _canonical_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frontend_identity(frontend, cosyvoice_root, model_dir, assets):
    tokenizer = frontend.tokenizer
    inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    get_vocab = getattr(inner_tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise RuntimeError("CosyVoice frontend tokenizer has no stable vocabulary.")
    vocabulary = get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise RuntimeError("CosyVoice frontend tokenizer vocabulary is invalid.")
    try:
        wetext_version = version("wetext")
    except PackageNotFoundError:
        wetext_version = None
    return {
        "frontend": "cosyvoice_wetext",
        "cosyvoice_git_head": git_head(cosyvoice_root),
        "cosyvoice_frontend_py_sha256": file_sha256(
            cosyvoice_root / "cosyvoice/cli/frontend.py"
        ),
        "cosyvoice_frontend_utils_py_sha256": file_sha256(
            cosyvoice_root / "cosyvoice/utils/frontend_utils.py"
        ),
        "cosyvoice_tokenizer_py_sha256": file_sha256(
            cosyvoice_root / "cosyvoice/tokenizer/tokenizer.py"
        ),
        "model_config_sha256": file_sha256(model_dir / "cosyvoice3.yaml"),
        "wetext_distribution_version": wetext_version,
        "wetext_asset_sha256": {
            name: file_sha256(path) for name, path in assets.items()
        },
        "tokenizer": {
            "module": type(tokenizer).__module__,
            "class": type(tokenizer).__name__,
            "inner_module": type(inner_tokenizer).__module__,
            "inner_class": type(inner_tokenizer).__name__,
            "vocabulary_size": len(vocabulary),
            "vocabulary_sha256": _canonical_sha256(vocabulary),
        },
        "text_preprocessing_policy": TEXT_PREPROCESSING_POLICY,
        "splitting_settings": dict(PINNED_FRONTEND_SPLITTING_SETTINGS),
    }


def configure_pinned_wetext_frontend(frontend, asset_root=None):
    """Bind unit planning to exact local WeText FST assets."""
    from wetext import Normalizer

    root = (
        Path(asset_root).expanduser().resolve()
        if asset_root is not None
        else Path.home() / ".cache/modelscope/hub/pengzhendong/wetext"
    )
    paths = {
        "zh_tagger": root / "zh/tn/tagger.fst",
        "zh_verbalizer": root / "zh/tn/verbalizer.fst",
        "en_tagger": root / "en/tn/tagger.fst",
        "en_verbalizer": root / "en/tn/verbalizer.fst",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    frontend.zh_tn_model = Normalizer(
        tagger_path=str(paths["zh_tagger"]),
        verbalizer_path=str(paths["zh_verbalizer"]),
        lang="zh",
    )
    frontend.en_tn_model = Normalizer(
        tagger_path=str(paths["en_tagger"]),
        verbalizer_path=str(paths["en_verbalizer"]),
        lang="en",
    )
    frontend.text_frontend = "wetext"
    return paths


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


def create_cosyvoice_model(auto_model, model_dir, settings,
                          offline_cached_wetext=False):
    if not offline_cached_wetext:
        return auto_model(
            model_dir=str(model_dir), load_trt=settings["load_trt"],
            load_vllm=settings["load_vllm"], fp16=settings["fp16"],
        )
    # Keep startup on the same already-cached WeText assets used by the
    # accepted RL runs. The patched lookup is restored after model loading.
    import wetext.wetext as wetext_impl
    cache = Path.home() / ".cache/modelscope/hub/pengzhendong/wetext"
    for language in ("zh", "en"):
        for name in ("tagger.fst", "verbalizer.fst"):
            if not (cache / language / "tn" / name).is_file():
                raise RuntimeError("Pinned local WeText assets are incomplete.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    original_download = wetext_impl.snapshot_download

    def local_wetext(model_id):
        if model_id != "pengzhendong/wetext":
            raise RuntimeError(f"Unexpected remote model request: {model_id}")
        return str(cache)

    wetext_impl.snapshot_download = local_wetext
    try:
        return auto_model(
            model_dir=str(model_dir),
            load_trt=settings["load_trt"],
            load_vllm=settings["load_vllm"],
            fp16=settings["fp16"],
        )
    finally:
        wetext_impl.snapshot_download = original_download


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


class CosyVoiceFrontendAdapter:
    """Load the pinned production frontend without invoking TTS inference."""

    def __init__(self, cosyvoice_root, model_dir, wetext_asset_root=None,
                 offline_cached_wetext=False):
        self.cosyvoice_root = Path(cosyvoice_root).expanduser().resolve()
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.wetext_asset_root = (
            Path(wetext_asset_root).expanduser().resolve()
            if wetext_asset_root is not None else None
        )
        self.settings = dict(SETTLED_SETTINGS)
        self.offline_cached_wetext = offline_cached_wetext
        self._model = None
        self._frontend = None
        self._metadata = None

    def initialize(self):
        if self._model is not None:
            return json.loads(json.dumps(self._metadata))

        if (self.offline_cached_wetext
                and file_sha256(self.model_dir / "llm.pt") != VALIDATED_RL_SHA256):
            raise RuntimeError("CosyVoice3 RL checkpoint differs from the validated run.")

        model_config = self.model_dir / "cosyvoice3.yaml"
        frontend_source = self.cosyvoice_root / "cosyvoice/cli/frontend.py"
        frontend_utils_source = (
            self.cosyvoice_root / "cosyvoice/utils/frontend_utils.py"
        )
        tokenizer_source = self.cosyvoice_root / "cosyvoice/tokenizer/tokenizer.py"
        for required in (
            self.cosyvoice_root, self.model_dir, model_config,
            frontend_source, frontend_utils_source, tokenizer_source,
        ):
            if not required.exists():
                raise FileNotFoundError(required)

        import_started = time.perf_counter()
        torch, torchaudio, auto_model = load_cosyvoice_runtime(self.cosyvoice_root)
        import_seconds = time.perf_counter() - import_started
        load_started = time.perf_counter()
        model = create_cosyvoice_model(
            auto_model, self.model_dir, self.settings,
            offline_cached_wetext=self.offline_cached_wetext,
        ) if self.offline_cached_wetext else create_cosyvoice_model(
            auto_model, self.model_dir, self.settings,
        )
        model_load_seconds = time.perf_counter() - load_started
        frontend = model.frontend
        assets = configure_pinned_wetext_frontend(
            frontend, self.wetext_asset_root
        )
        identity = _frontend_identity(
            frontend, self.cosyvoice_root, self.model_dir, assets
        )
        if (self.offline_cached_wetext
                and _canonical_sha256(identity) != VALIDATED_FRONTEND_SHA256):
            raise RuntimeError("Current native frontend differs from the validated RL configuration.")
        self._model = model
        self._frontend = frontend
        self._metadata = {
            "identity": identity,
            "locations": {
                "cosyvoice_repo": str(self.cosyvoice_root),
                "model_dir": str(self.model_dir),
                "wetext_assets": {
                    name: str(path) for name, path in assets.items()
                },
            },
            "runtime": {
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_build": torch.version.cuda,
            },
            "import_seconds": import_seconds,
            "model_load_seconds": model_load_seconds,
            "inference_calls": 0,
        }
        return json.loads(json.dumps(self._metadata))

    def normalize(self, text):
        if self._frontend is None:
            raise RuntimeError(
                "CosyVoice frontend adapter must be initialized before planning."
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Synthesis-unit planning text must not be empty.")
        result = self._frontend.text_normalize(
            text, split=True, text_frontend=True
        )
        return list(result)

    def normalize_heading(self, heading):
        """Use the same native frontend for an explicitly detected title."""
        if self._frontend is None:
            raise RuntimeError("CosyVoice frontend must be initialized before planning.")
        result = self._frontend.text_normalize(
            heading, split=False, text_frontend=True
        )
        if not isinstance(result, str) or not result:
            raise RuntimeError("Native chapter-title normalization is invalid.")
        return result

    def release(self):
        """Free the planning model before the synthesis model is loaded."""
        self._frontend = None
        self._model = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CosyVoiceAdapter:
    """Own one initialized CosyVoice3 model for a production CLI invocation."""

    def __init__(self, cosyvoice_root, model_dir, prompt_wav, prompt_text_file,
                 offline_cached_wetext=False):
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
        self._unit_frontend_identity = None
        self.offline_cached_wetext = offline_cached_wetext

    def configuration(self):
        return {
            "backend": "cosyvoice3_zero_shot",
            "cosyvoice_repo": str(self.cosyvoice_root),
            "model_dir": str(self.model_dir),
            "prompt_wav": str(self.prompt_wav),
            "prompt_transcript_file": str(self.prompt_text_file),
            "settings": dict(self.settings),
            "text_preprocessing": {"policy": TEXT_PREPROCESSING_POLICY},
        }

    def preprocessing_provenance(self, source_text):
        return preprocess_synthesis_text(source_text)[1]

    def initialize(self):
        if self._model is not None:
            return dict(self._metadata)

        if (self.offline_cached_wetext
                and file_sha256(self.model_dir / "llm.pt") != VALIDATED_RL_SHA256):
            raise RuntimeError("CosyVoice3 RL checkpoint differs from the validated run.")

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
        model = create_cosyvoice_model(
            auto_model, self.model_dir, self.settings,
            offline_cached_wetext=self.offline_cached_wetext,
        ) if self.offline_cached_wetext else create_cosyvoice_model(
            auto_model, self.model_dir, self.settings,
        )
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

        synthesis_text, text_preprocessing = preprocess_synthesis_text(text)

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
            self._model, torch, synthesis_text, self._prompt_text, self.prompt_wav,
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
            "text_preprocessing": text_preprocessing,
            "cosyvoice_chunks": chunk_count,
            "inference_seconds": inference_seconds,
            "rtf": inference_seconds / audio["duration_seconds"],
            "peak_torch_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
        }

    def initialize_units(self):
        """Reuse the warm model and certify its frontend against the unit plan."""
        metadata = self.initialize()
        if self._unit_frontend_identity is None:
            assets = configure_pinned_wetext_frontend(self._model.frontend)
            self._unit_frontend_identity = _frontend_identity(
                self._model.frontend, self.cosyvoice_root, self.model_dir, assets
            )
        return {
            **metadata,
            "frontend_identity_sha256": _canonical_sha256(
                self._unit_frontend_identity
            ),
        }

    def generate_unit(self, normalized_text, output_path, seed):
        """Synthesize exactly one frozen unit with both frontend passes disabled."""
        if self._unit_frontend_identity is None:
            raise RuntimeError("Unit frontend must be initialized before generation.")
        if not isinstance(normalized_text, str) or not normalized_text.strip():
            raise ValueError("Frozen synthesis-unit text must not be empty.")
        if "\r" in normalized_text:
            raise ValueError("Frozen synthesis-unit text contains a carriage return.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer from 0 through 4294967295.")
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(output_path)
        torch = self._torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        set_cosyvoice_random_seed(seed)
        started = time.perf_counter()
        chunks = list(self._model.inference_zero_shot(
            normalized_text, self._prompt_text, str(self.prompt_wav),
            stream=self.settings["stream"], text_frontend=False,
        ))
        if len(chunks) != 1:
            raise RuntimeError(
                f"Frozen unit produced {len(chunks)} outputs; expected exactly one."
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        write_pcm16_wav(
            self._torchaudio, output_path, chunks[0]["tts_speech"],
            self._model.sample_rate,
        )
        audio = wav_info(output_path, self._model.sample_rate)
        return {
            "cosyvoice_chunks": 1,
            "inference_seconds": inference_seconds,
            "rtf": inference_seconds / audio["duration_seconds"],
            "peak_torch_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
            "frontend_bypass": True,
        }
