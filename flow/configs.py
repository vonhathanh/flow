import json
import os
import pyaudio

from enum import StrEnum


class AppStatus(StrEnum):
    LOADING = "Loading new model..."
    READY = "Ready, speak to start recording"
    STOPPED = "App stopped"


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

CONFIG_FILE = "./config.json"

AVAILABLE_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large", "turbo"]

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w+") as f:
        json.dump({"model": "tiny.en"}, f)

with open(CONFIG_FILE, "r") as f:
    # default whisper model
    CURRENT_MODEL = json.load(f)["model"]
