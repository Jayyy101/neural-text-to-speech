# Neural Multilingual Text-to-Speech System

A locally controlled neural Text-to-Speech (TTS) system that converts English, Mandarin Chinese, and mixed Chinese-English text into speech using MeloTTS.

This project started with an English VITS prototype, then expanded through XTTS and Azure Neural TTS testing. The final system uses MeloTTS as the main backend because it supports multilingual synthesis, runs locally, and integrates well with a custom Python GUI.

---

## Project Overview

The goal of this project is to build an end-to-end multilingual TTS application that can:

- accept English and Chinese text input
- generate natural-sounding speech using a neural TTS model
- run locally without depending on a commercial TTS API
- save generated speech as timestamped WAV files
- provide a simple GUI for user interaction

Azure Neural TTS was used only as a quality benchmark. XTTS was tested as a multilingual experiment. The earlier English VITS version was used as the baseline prototype.

---

## Final System

The final system uses:

- **MeloTTS** as the main speech synthesis backend
- **Tkinter** for the graphical user interface
- **PyTorch with CUDA** for GPU-accelerated inference
- **Timestamped WAV outputs** saved to the `outputs/` folder

The GUI allows users to:

- choose a language: English or Chinese
- choose a speaker
- adjust speech speed
- type or clear input text
- generate speech
- open the generated audio file
- view status updates while speech is being generated

---

## Project Files

```text
src/
├── generate.py          # older English VITS baseline prototype
├── xtts.py              # XTTS multilingual experiment
├── generate_azure.py    # Azure Neural TTS benchmark
├── generate_melo.py     # final MeloTTS backend
└── gui.py               # Tkinter GUI for the final system
```

---

## Features

- Local neural TTS generation
- English text-to-speech support
- Mandarin Chinese text-to-speech support
- Mixed Chinese-English text support
- Speaker selection
- Speed control
- GUI-based input and playback
- Timestamped WAV output files
- Model caching for faster repeated generation
- Progress indicator during synthesis

---

## System

Tested on:

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- CPU: Ryzen 7 7800X3D
- RAM: 32GB
- Python: 3.10
- OS: Windows 10

---

## Setup

Create and activate the conda environment:

```bash
conda create -n melo python=3.10
conda activate melo
```

Install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Install MeloTTS:

```bash
pip install melo-tts
```

Depending on the system, additional packages may be needed for Chinese text processing and audio generation.

Make sure commands are run from the main project folder.

For example, if cloned from GitHub:

```bash
cd neural-text-to-speech
```

---

## Run the GUI

From the main project folder, run:

```bash
python src/gui.py
```

The GUI allows users to enter text, choose a language, select a speaker, adjust speed, generate speech, and open the most recent audio output.

Generated audio files are saved in:

```text
outputs/
```

The `outputs/` folder is ignored by Git so generated WAV files are not uploaded to the repository.

---

## Evaluation Summary

The system was tested with English, Chinese, and mixed Chinese-English inputs. Testing focused on pronunciation quality, pacing, multilingual support, and inference time.

| Test | Language | Speaker | Speed | Inference Time | Notes |
|---|---|---|---:|---:|---|
| English short sentence | EN | EN-Default | 1.0 | 2.538s | Good pronunciation, but slightly fast and struggled with “MeloTTS.” |
| English paragraph | EN | EN-Default | 1.0 | 0.266s | Clear pronunciation and better pauses. |
| Chinese sentence | ZH | ZH | 1.0 | 2.640s | Good pronunciation and pauses, but slightly fast. |
| Mixed Chinese-English | ZH | ZH | 1.0 | 0.476s | Chinese sounded strong; English words had a noticeable Chinese accent. |
| English paragraph slower | EN | EN-Default | 0.9 | 0.334s | Pacing sounded better than 1.0. |
| Chinese sentence slower | ZH | ZH | 0.9 | 0.190s | Still slightly fast; 0.8 sounded better. |

Repeated generation became faster because the backend caches loaded models during the same GUI session.

---

## Current Limitations

- Mixed Chinese-English input works best with the Chinese model, but English words may sound accented.
- Some technical terms, such as “MeloTTS,” may need input formatting to improve pronunciation.
- Speech speed may need adjustment depending on the language and input length.
- The system currently runs best on the configured Windows GPU environment.
- The GUI is local only and has not yet been deployed as a web application.

---

## Future Improvements

- Add a web frontend using Flask or FastAPI so the system can be accessed from other devices.
- Deploy the backend on a local or cloud GPU server for remote use.
- Improve pronunciation handling for technical words and mixed-language input.
- Add more detailed evaluation metrics for pronunciation quality, speed, and user feedback.
- Compare MeloTTS more directly against Azure Neural TTS, XTTS, and the earlier VITS baseline.
- Package the app as a desktop application for easier use.

---

## Author

Jay Ma  
GitHub: [Jayyy101](https://github.com/Jayyy101)