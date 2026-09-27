from TTS.api import TTS
from pypinyin import pinyin, Style
import torch
import time


DEFAULT_TEXT = "Hello Jay. This is your RTX 4070 Ti generating neural speech."


def print_device_info():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def preprocess_text(text, language):
    if language == "zh":
        result = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
        return " ".join([item[0] for item in result])
    return text


# Load model once
tts = TTS(model_name="tts_models/en/ljspeech/vits")


def generate_speech(text, file_path="outputs/output.wav", language="en"):
    start = time.time()

    tts.tts_to_file(
        text=text,
        file_path=file_path,
    )

    print("Inference time:", round(time.time() - start, 3), "seconds")


if __name__ == "__main__":
    print_device_info()

    language = input("Enter language (en or zh): ").strip().lower() or "en"
    if language not in {"en", "zh"}:
        language = "en"

    user_text = input("Enter text to synthesize: ").strip()
    text_to_speak = user_text or DEFAULT_TEXT

    processed_text = preprocess_text(text_to_speak, language)
    print("Processed text:", processed_text)

    generate_speech(processed_text, language=language)