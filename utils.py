import os
import wave
from configs import *

def process_recorded_speech(speech: str) -> str:
    return speech

def save_audio(audio_buffer, filename: str):
    wf = wave.open(filename, 'wb')
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