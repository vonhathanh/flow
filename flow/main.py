import pyautogui
import numpy as np
import time
import whisper

from utils import process_recorded_speech, remove_audio, save_audio
from flow.ignored_words import IGNORED_WORDS
from flow.configs import *

model = whisper.load_model("tiny.en")


def is_silent(data_chunk):
    """Returns 'True' if below the 'silent' threshold"""
    return max(data_chunk) < THRESHOLD


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening... Speak to start recording.")

    audio_buffer = []
    silent_chunks = 0
    audio_started = False

    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        silent = is_silent(audio_data)
        print("is silent ", silent)

        if audio_started:
            audio_buffer.append(data)

        if not silent and not audio_started:
            print("Recording started...")
            audio_started = True
            audio_buffer.append(data)

        elif silent and audio_started:
            silent_chunks += 1
            if silent_chunks > SILENCE_LIMIT * (RATE / CHUNK):
                break
        else:
            silent_chunks = 0

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return audio_buffer


if __name__ == "__main__":
    while True:
        audio_data = record_audio()

        if not audio_data:
            continue

        file_name = f"recorded_audio_{int(time.time())}.wav"

        save_audio(audio_data, file_name)

        result = model.transcribe(file_name)

        remove_audio(file_name)

        final_text = process_recorded_speech(result["text"])

        if final_text in IGNORED_WORDS:
            continue

        # Type the text
        pyautogui.typewrite(process_recorded_speech(result["text"]))

        # Press Enter
        pyautogui.press('enter')
