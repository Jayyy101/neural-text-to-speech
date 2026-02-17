from TTS.api import TTS
import torch
import time

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))


# Load model
tts = TTS(model_name="tts_models/en/ljspeech/vits")

#Measure inference time
start = time.time()

tts.tts_to_file(
    text="Hello Jay. This is your RTX 4070 Ti generating neural speech.",
    file_path="output.wav"
)

print("inference time:", round(time.time() - start, 3), "seconds")