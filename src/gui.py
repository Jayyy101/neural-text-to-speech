import tkinter as tk
from tkinter import ttk
import threading
import os

AVAILABLE_SPEAKERS = {
    "EN": ["EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"],
    "ZH": ["ZH"],
}

DEFAULT_SPEAKERS = {
    "EN": "EN-Default",
    "ZH": "ZH",
}


SAMPLE_TEXTS = {
    "EN": "Hello, this is a test of my MeloTTS graphical interface. The system can take user input, generate speech locally, and save the audio as a WAV file.",
    "ZH": "你好，我叫杰。今天我正在测试我的 MeloTTS 图形界面。这个系统可以在本地电脑上生成语音，并且支持中文和英文输入。",
}

latest_output_path = None

root = tk.Tk()
root.title("MeloTTS GUI")
root.geometry("500x500")


def update_speakers(event=None):
    language = language_choice.get()
    speaker_choice["values"] = AVAILABLE_SPEAKERS[language]
    speaker_choice.set(DEFAULT_SPEAKERS[language])

    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, SAMPLE_TEXTS[language])

def open_output():
    if latest_output_path is None:
        status_label.config(text="No output file yet.")
        return

    if not os.path.exists(latest_output_path):
        status_label.config(text="Output file was not found.")
        return

    os.startfile(latest_output_path)

def start_generation():
    thread = threading.Thread(target=generate_speech)
    thread.daemon = True
    thread.start()

def clear_text():
    text_box.delete("1.0", tk.END)
    status_label.config(text="Text cleared.")

def generate_speech():
    global latest_output_path

    user_text = text_box.get("1.0", tk.END).strip()
    language = language_choice.get()
    speaker = speaker_choice.get()
    speed = speed_slider.get()

    generate_button.config(state=tk.DISABLED)
    open_button.config(state=tk.DISABLED)

    try:
        status_label.config(text="Generating speech...")
        progress_bar.pack(pady=5)
        progress_bar.start()
        root.update()

        from generate_melo import synthesize_melo

        result = synthesize_melo(
            text=user_text,
            language=language,
            speaker_name=speaker,
            speed=speed
        )

        latest_output_path = os.path.abspath(result["output_path"])

        print("Synthesis complete.")
        print("Saved to:", latest_output_path)
        print("Inference time:", result["inference_time"], "seconds")

        file_name = os.path.basename(latest_output_path)
        
        status_label.config(
            text=f"Done. Saved as {file_name}"
        )
        open_button.config(state=tk.NORMAL)

    except Exception as error:
        print("Error:", error)
        status_label.config(text=f"Error: {error}")

    finally:
        progress_bar.stop()
        progress_bar.pack_forget()
        generate_button.config(state=tk.NORMAL)


language_label = tk.Label(root, text="Select language:")
language_label.pack(pady=5)

language_choice = ttk.Combobox(root, values=["EN", "ZH"], state="readonly")
language_choice.set("EN")
language_choice.pack(pady=5)
language_choice.bind("<<ComboboxSelected>>", update_speakers)

speaker_label = tk.Label(root, text="Select speaker:")
speaker_label.pack(pady=5)

speaker_choice = ttk.Combobox(
    root,
    values=AVAILABLE_SPEAKERS["EN"],
    state="readonly"
)
speaker_choice.set(DEFAULT_SPEAKERS["EN"])
speaker_choice.pack(pady=5)

speed_label = tk.Label(root, text="Select speed:")
speed_label.pack(pady=5)

speed_slider = tk.Scale(
    root,
    from_=0.8,
    to=1.2,
    resolution=0.1,
    orient=tk.HORIZONTAL
)
speed_slider.set(1.0)
speed_slider.pack(pady=5)

text_label = tk.Label(root, text="Enter text:")
text_label.pack(pady=10)

text_box = tk.Text(root, height=6, width=50, wrap=tk.WORD)
text_box.pack()
text_box.insert(tk.END, SAMPLE_TEXTS["EN"])

clear_button = tk.Button(root, text="Clear Text", command=clear_text)
clear_button.pack(pady=5)

generate_button = tk.Button(root, text="Generate Speech", command=start_generation)
generate_button.pack(pady=15)

open_button = tk.Button(
    root,
    text="Open Output",
    command=open_output,
    state=tk.DISABLED
)
open_button.pack(pady=5)

status_label = tk.Label(root, text="Ready.")
status_label.pack(pady=10)

progress_bar = ttk.Progressbar(
    root,
    mode="indeterminate",
    length=250
)


root.mainloop()