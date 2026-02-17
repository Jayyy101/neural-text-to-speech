# Neural Text-to-Speech System (GPU Accelerated)

## Overview

This project implements a GPU-accelerated neural Text-to-Speech (TTS) system using the VITS architecture from Coqui TTS.  
The system runs on an NVIDIA RTX 4070 Ti SUPER and achieves sub-second inference times.

The goal of this project is to build a reproducible, high-performance TTS pipeline and expand it toward multilingual and real-time applications.

---

## Features

- CUDA-enabled PyTorch inference
- VITS neural TTS architecture
- Automatic phonemization via eSpeak NG
- Sub-1 second speech generation
- Real-time factor ≈ 0.13 (faster than real-time)
- Structured, reproducible environment setup

---

## System Specifications

- **GPU:** NVIDIA RTX 4070 Ti SUPER  
- **CPU:** Ryzen 7 7800X3D  
- **RAM:** 32GB  
- **Python:** 3.10  
- **PyTorch:** CUDA 12.6 build  
- **OS:** Windows 10  

---

## Performance Example

Example inference:

- Input: 2 sentences  
- Processing time: ~0.98 seconds  
- Real-time factor: ~0.138  

This means the model generates speech approximately 7× faster than real-time playback.

---

## Project Structure


---

## Setup

### 1. Create Conda Environment

```bash
conda create -n tts python=3.10
conda activate tts
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install TTS
python src/generate.py


---

# 🚀 Then Run

```powershell
git add README.md
git commit -m "Polish README formatting and clean structure"
git push
