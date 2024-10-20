import json
import os
import pyaudio
import configparser

from enum import Enum


class AppStatus(str, Enum):
    LOADING = "Loading new model..."
    READY = "Ready, speak to start recording"
    STOPPED = "App stopped"
    CLOSING = "Closing app..."

CONFIG_FILE = "./config.ini"

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Voice Activity Detection parameters
THRESHOLD = int(config["DEFAULT"]["threshold"])  # Adjust this value based on your microphone and environment
SILENCE_LIMIT = 0.5  # Number of seconds of silence to stop the recording

IGNORED_WORDS = {"", " "}

# directory to store the audio files
RECORDINGS_DIR = "./recordings/"

AVAILABLE_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large", "turbo"]

# default whisper model
CURRENT_MODEL = config["DEFAULT"]["model"]
