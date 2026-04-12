from melo.api import TTS
import torch
import time

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


def print_device_info():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def normalize_language(user_language):
    user_language = user_language.strip().upper()

    if user_language in {"EN", "ENGLISH"}:
        return "EN"
    if user_language in {"ZH", "CHINESE", "MANDARIN", "ZH-CN"}:
        return "ZH"

    return "EN"


def load_model(language):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return TTS(language=language, device=device)


def generate_speech(model, text, language="EN", file_path="outputs/output_melo.wav", speaker_name=None, speed=1.0):
    start = time.time()

    speaker_ids = model.hps.data.spk2id

    if speaker_name is None:
        speaker_name = DEFAULT_SPEAKERS.get(language, "EN-Default")

    if speaker_name not in speaker_ids:
        raise ValueError(
            f"Speaker '{speaker_name}' not found. Available speakers: {list(speaker_ids.keys())}"
        )

    model.tts_to_file(
        text,
        speaker_ids[speaker_name],
        file_path,
        speed=speed,
    )

    print("Language used:", language)
    print("Speaker used:", speaker_name)
    print("Inference time:", round(time.time() - start, 3), "seconds")


if __name__ == "__main__":
    print_device_info()

    user_language = input("Enter language (EN or ZH): ")
    language = normalize_language(user_language)

    print("\nSuggested speakers for this language:")
    for speaker_name in AVAILABLE_SPEAKERS[language]:
        print("-", speaker_name)

    user_speaker = input("\nEnter speaker name (or press Enter for default): ").strip()
    speaker_to_use = user_speaker if user_speaker else DEFAULT_SPEAKERS[language]

    user_text = input("Enter text to synthesize: ").strip()
    text_to_speak = user_text or DEFAULT_TEXTS[language]

    model = load_model(language)
    print("Loaded speaker IDs:", list(model.hps.data.spk2id.keys()))

    generate_speech(
        model,
        text_to_speak,
        language=language,
        speaker_name=speaker_to_use,
    )
    