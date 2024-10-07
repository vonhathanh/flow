import asyncio
import multiprocessing
import os

import pyautogui
import numpy as np
import whisper

from glob import glob
from os.path import join

from utils import process_recorded_speech, remove_audio, save_audio, is_silent
from flow.ignored_words import IGNORED_WORDS
from flow.configs import *

model = whisper.load_model("tiny.en")

# a flag to indicate the program has been stopped, for future use
is_stopped = False

os.makedirs(RECORDINGS_DIR, exist_ok=True)


async def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    audio_buffer = []
    silent_chunks = 0
    audio_started = False

    print("Listening... Speak to start recording.")

    while True:
        data = stream.read(CHUNK)
        raw_audio_data = np.frombuffer(data, dtype=np.int16)

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

        await asyncio.sleep(0.01)

    stream.stop_stream()
    stream.close()
    p.terminate()


async def process_new_audio_files():
    while True:
        for file_name in sorted(glob(join(RECORDINGS_DIR, "*.wav"))):
            print("process file: ", file_name)
            with multiprocessing.Pool(1) as pool:
                pool.map(transcribe, (file_name,))

        await asyncio.sleep(0.01) # small idle for the cpu


def transcribe(file_name):
    result = model.transcribe(file_name)

    print("transcribe result: ", result)

    remove_audio(file_name)

    final_text = process_recorded_speech(result["text"])
    if final_text in IGNORED_WORDS:
        print(f"Skipping: {final_text}")
        return

    # Type the text
    pyautogui.typewrite(process_recorded_speech(result["text"]))
    # Press Enter
    pyautogui.press('enter')


async def main():
    await asyncio.gather(
        record_audio(),
        process_new_audio_files(),
    )


if __name__ == "__main__":
    asyncio.run(main())
