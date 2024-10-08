import multiprocessing
import os
import time
import warnings

import pyautogui
import numpy as np
import whisper

from glob import glob
from os.path import join

from utils import process_recorded_speech, remove_audio, save_audio, is_silent
from flow.ignored_words import IGNORED_WORDS
from flow.configs import *

# Filter out the specific FutureWarning from torch.load
warnings.simplefilter(action='ignore', category=FutureWarning)

model = whisper.load_model(WHISPER_MODEL)

# a flag to indicate the program has been stopped, for future use
is_stopped = False

os.makedirs(RECORDINGS_DIR, exist_ok=True)


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    audio_buffer = []
    silent_chunks = 0
    audio_started = False

    print("Listening... Speak to start recording.")

    while True:
        data = stream.read(CHUNK)
        raw_audio_data = np.frombuffer( stream.read(CHUNK), dtype=np.int16)
        # print(f"max: {raw_audio_data.max()}, min: {raw_audio_data.min()}, mean: {raw_audio_data.mean()}")
        silent = is_silent(raw_audio_data)
        # print("is silent ", silent)

        if audio_started:
            audio_buffer.append(data)

        # user starts talking
        if not silent and not audio_started:
            print("Recording started...")
            audio_started = True
            audio_buffer.append(data)

        elif silent and audio_started:
            silent_chunks += 1
            if silent_chunks > SILENCE_LIMIT * (RATE / CHUNK):
                # user has stopped talking, saves recording to file
                silent_chunks = 0
                audio_started = False
                save_audio(audio_buffer)
                audio_buffer = []
        # user still talking
        else:
            silent_chunks = 0

        if is_stopped:
            break

        time.sleep(0.01)

    stream.stop_stream()
    stream.close()
    p.terminate()


def process_new_audio_files():
    while True:
        for file_name in sorted(glob(join(RECORDINGS_DIR, "*.wav"))):
            print("process file: ", file_name)
            transcribe(file_name)

        time.sleep(0.01)


def transcribe(file_name):
    result = model.transcribe(file_name)

    print("transcribe result: ", result)

    remove_audio(file_name)

    final_text = process_recorded_speech(result)

    if final_text in IGNORED_WORDS:
        print(f"Skipping: {final_text}")
    else:
        # Type the text
        pyautogui.typewrite(final_text)
        # Press Enter
        pyautogui.press('enter')


def main():
    record_process = multiprocessing.Process(target=record_audio)
    transcribe_process = multiprocessing.Process(target=process_new_audio_files)

    record_process.start()
    transcribe_process.start()

    try:
        while not is_stopped:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print("Terminating processes...")

    record_process.terminate()
    transcribe_process.terminate()

    record_process.join()
    transcribe_process.join()

if __name__ == "__main__":
    main()
