import os
import time
import azure.cognitiveservices.speech as speechsdk

DEFAULT_TEXTS = {
    "en": "Hello. My name is Jay. This is a test of the multilingual text to speech system.",
    "zh": "你好。我叫杰。今天天气很好。",
}

DEFAULT_VOICES = {
    "en": "en-US-AvaMultilingualNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}


def normalize_language(user_language):
    user_language = user_language.strip().lower()
    if user_language in {"zh", "zh-cn", "chinese", "mandarin"}:
        return "zh"
    return "en"


def synthesize_to_file(text, voice_name, output_file):
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")

    if not key or not region:
        raise ValueError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION environment variable.")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice_name
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    start = time.time()
    result = synthesizer.speak_text_async(text).get()
    elapsed = round(time.time() - start, 3)

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Voice used:", voice_name)
        print("Inference time:", elapsed, "seconds")
        print("Saved to:", output_file)
    else:
        cancellation = speechsdk.SpeechSynthesisCancellationDetails(result)
        print("Synthesis failed.")
        print("Reason:", cancellation.reason)
        print("Error details:", cancellation.error_details)


if __name__ == "__main__":
    user_language = input("Enter language (en or zh): ")
    language = normalize_language(user_language)

    user_text = input("Enter text to synthesize: ").strip()
    text_to_speak = user_text or DEFAULT_TEXTS[language]

    voice = DEFAULT_VOICES[language]
    output_file = "outputs/output_azure.wav"

    synthesize_to_file(text_to_speak, voice, output_file)