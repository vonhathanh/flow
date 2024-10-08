import os

import pyaudio

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Voice Activity Detection parameters
THRESHOLD = 1000  # Adjust this value based on your microphone and environment
SILENCE_LIMIT = 1  # Number of seconds of silence to stop the recording

# directory to store the audio files
RECORDINGS_DIR = "./recordings/"

# default whisper model
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")