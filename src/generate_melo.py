from melo.api import TTS
import torch
import time
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DEFAULT_TEXTS = {
    "EN": "Hello. My name is Jay. This is a test of the multilingual text to speech system.",
    "ZH": "你好。我叫杰。今天天气很好。",
}

DEFAULT_SPEAKERS = {
    "EN": "EN-Default",
    "ZH": "ZH",
}

AVAILABLE_SPEAKERS = {
    "EN": ["EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"],
    "ZH": ["ZH"],
}

DEFAULT_SPEED = 1.0


def print_device_info():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def normalize_language(user_language):
    user_language = user_language.strip().upper()

    if user_language in {"EN", "ENGLISH"}:
        return "EN"
    if user_language in {"ZH", "ZH-CN", "CHINESE", "MANDARIN"}:
        return "ZH"

    return "EN"


def build_output_path(language):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/melo_{language.lower()}_{timestamp}.wav"


def load_model(language):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return TTS(language=language, device=device)


def validate_speaker(model, speaker_name):
    speaker_ids = model.hps.data.spk2id
    if speaker_name not in speaker_ids:
        raise ValueError(
            f"Speaker '{speaker_name}' not found. Available speakers: {list(speaker_ids.keys())}"
        )
    return speaker_ids[speaker_name]


def parse_speed(user_speed):
    if not str(user_speed).strip():
        return DEFAULT_SPEED

    speed = float(user_speed)
    if speed < 0.8 or speed > 1.2:
        raise ValueError("Speed must be between 0.8 and 1.2.")
    return speed


def synthesize_melo(text, language, speaker_name=None, speed=1.0):
    language = normalize_language(language)
    speed = parse_speed(speed)

    os.makedirs("outputs", exist_ok=True)

    if not text.strip():
        text = DEFAULT_TEXTS[language]

    if speaker_name is None or not str(speaker_name).strip():
        speaker_name = DEFAULT_SPEAKERS[language]

    model = load_model(language)
    speaker_id = validate_speaker(model, speaker_name)
    output_path = build_output_path(language)

    start = time.time()
    model.tts_to_file(
        text,
        speaker_id,
        output_path,
        speed=speed,
    )
    elapsed = round(time.time() - start, 3)

    result = {
        "language": language,
        "speaker": speaker_name,
        "speed": speed,
        "output_path": output_path,
        "inference_time": elapsed,
    }

    return result


def generate_speech_interactive():
    print_device_info()

    print("\nNote: Use ZH for Chinese or mixed Chinese-English text.")
    user_language = input("Enter language (EN or ZH): ")
    language = normalize_language(user_language)

    print("\nSuggested speakers for this language:")
    for speaker_name in AVAILABLE_SPEAKERS[language]:
        print("-", speaker_name)

    user_speaker = input("\nEnter speaker name (or press Enter for default): ").strip()
    user_speed = input("Enter speed between 0.8 and 1.2 (or press Enter for default 1.0): ")
    user_text = input("Enter text to synthesize: ").strip()

    result = synthesize_melo(
        text=user_text,
        language=language,
        speaker_name=user_speaker,
        speed=user_speed,
    )

    print("\nSynthesis complete.")
    print("Language used:", result["language"])
    print("Speaker used:", result["speaker"])
    print("Speed used:", result["speed"])
    print("Saved to:", result["output_path"])
    print("Inference time:", result["inference_time"], "seconds")


if __name__ == "__main__":
    generate_speech_interactive()