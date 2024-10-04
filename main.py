import pyautogui
import pyaudio
import numpy as np
import wave
import time
import whisper

model = whisper.load_model("small.en")

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Voice Activity Detection parameters
THRESHOLD = 500  # Adjust this value based on your microphone and environment
SILENCE_LIMIT = 2  # Number of seconds of silence to stop the recording


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


def save_audio(audio_buffer, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_buffer))
    wf.close()
    print(f"Audio saved as {filename}")


if __name__ == "__main__":
    while True:
        audio_data = record_audio()
        if audio_data:
            file_name = f"recorded_audio_{int(time.time())}.wav"
            save_audio(audio_data, file_name)
            result = model.transcribe(file_name)

            # Type the text
            pyautogui.typewrite(result["text"])

            # Press Enter
            pyautogui.press('enter')
        else:
            print("No audio recorded.")
