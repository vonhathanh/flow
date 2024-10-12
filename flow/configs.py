import os
import pyaudio

from enum import StrEnum

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Voice Activity Detection parameters
THRESHOLD = int(os.getenv("THRESHOLD", "200"))  # Adjust this value based on your microphone and environment
SILENCE_LIMIT = 0.5  # Number of seconds of silence to stop the recording

# directory to store the audio files
RECORDINGS_DIR = "./recordings/"

# default whisper model
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")


class AppStatus(StrEnum):
    LOADING = "Loading model..."
    READY = "Ready, speak to start recording"
    TRANSCRIBING = "Transcribing your speech..."
