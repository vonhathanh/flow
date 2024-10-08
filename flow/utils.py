import os
import time
import wave
from flow.configs import *

def process_recorded_speech(transcribe_result: dict) -> str:
    return transcribe_result["text"]

def save_audio(audio_buffer):
    if not is_valid_audio(audio_buffer):
        return

    file_name = os.path.join(RECORDINGS_DIR, f"{int(time.time())}.wav")

    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_buffer))
    wf.close()

def remove_audio(filename: str):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def is_valid_audio(audio_buffer) -> bool:
    return True

def is_silent(data_chunk):
    """Returns 'True' if below the 'silent' threshold"""
    return max(data_chunk) < THRESHOLD