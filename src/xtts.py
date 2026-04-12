from TTS.api import TTS
import torch
import time

DEFAULT_TEXT = "Hello Jay. This is a test of the multilingual XTTS text to speech system."

DEFAULT_SPEAKERS = {
    "en": "Daisy Studious",
    "zh-cn": "Daisy Studious",
}

AVAILABLE_SPEAKERS = [
    "Daisy Studious",
    "Gracie Wise",
    "Andrew Chipper",
    "Luis Moray",
    "Claribel Dervla",
]


def print_device_info():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def normalize_language(user_language):
    user_language = user_language.strip().lower()

    if user_language == "en":
        return "en"
    if user_language in {"zh", "zh-cn", "chinese", "mandarin"}:
        return "zh-cn"

    return "en"


tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def generate_speech(text, language="en", file_path="outputs/output_xtts.wav", speaker=None):
    start = time.time()

    if speaker is None:
        speaker = DEFAULT_SPEAKERS.get(language, "Daisy Studious")

    tts.tts_to_file(
        text=text,
        speaker=speaker,
        language=language,
        file_path=file_path,
    )

    print("Language used:", language)
    print("Speaker used:", speaker)
    print("Inference time:", round(time.time() - start, 3), "seconds")


if __name__ == "__main__":
    print_device_info()

    user_language = input("Enter language (en or zh-cn): ")
    language = normalize_language(user_language)

    print("\nTry one of these speakers:")
    for speaker_name in AVAILABLE_SPEAKERS:
        print("-", speaker_name)

    user_speaker = input("\nEnter speaker name (or press Enter for default): ").strip()
    speaker_to_use = user_speaker if user_speaker else DEFAULT_SPEAKERS.get(language, "Daisy Studious")

    user_text = input("Enter text to synthesize: ").strip()
    text_to_speak = user_text or DEFAULT_TEXT

    generate_speech(text_to_speak, language=language, speaker=speaker_to_use)