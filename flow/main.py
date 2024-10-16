import multiprocessing
import time
import warnings
import sys
import pyautogui
import numpy as np
import whisper

from glob import glob
from os.path import join
from multiprocessing import Queue

from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QMainWindow, QVBoxLayout, QPushButton, QComboBox

from utils import process_recorded_speech, remove_audio, save_audio, is_silent
from flow.ignored_words import IGNORED_WORDS
from flow.configs import *

# Filter out the specific FutureWarning from torch.load
warnings.simplefilter(action='ignore', category=FutureWarning)

# a flag to indicate the program has been stopped, for future use
is_stopped = False

os.makedirs(RECORDINGS_DIR, exist_ok=True)

model = whisper.load_model(CURRENT_MODEL)


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    audio_buffer = []
    silent_chunks = 0
    audio_started = False

    print(AppStatus.READY)

    while True:
        data = stream.read(CHUNK)
        raw_audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        # print(f"max: {raw_audio_data.max()}, min: {raw_audio_data.min()}, mean: {raw_audio_data.mean()}")
        silent = is_silent(raw_audio_data)

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

    stream.stop_stream()
    stream.close()
    p.terminate()


def process_new_audio_files(q: Queue):
    global model
    while True:
        if not q.empty():
            model_name = q.get()
            print(f"{model_name=}")
            model = whisper.load_model(model_name)
            print("new model loaded successfully")

        for file_name in sorted(glob(join(RECORDINGS_DIR, "*.wav"))):
            print("process file: ", file_name)
            transcribe(file_name)

        time.sleep(0.01)


def transcribe(file_name):
    result = model.transcribe(file_name)

    remove_audio(file_name)

    final_text = process_recorded_speech(result)

    if final_text in IGNORED_WORDS:
        print(f"Skipping: {final_text}")
    else:
        pyautogui.hotkey("shift", "tab")
        # Type the text
        pyautogui.typewrite(final_text)
        # Press Enter
        pyautogui.press('enter')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.q = Queue()
        self.setWindowTitle("Voice Flow")
        self.setGeometry(100, 100, 200, 100)

        layout = QVBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_processes)
        layout.addWidget(self.start_button)

        self.model_name = QLabel("Select model: ")
        layout.addWidget(self.model_name)

        self.model_selection_combobox = QComboBox()
        self.model_selection_combobox.addItems(AVAILABLE_MODELS)
        self.model_selection_combobox.activated.connect(self.update_model_name)
        layout.addWidget(self.model_selection_combobox)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processes)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.model = whisper.load_model(CURRENT_MODEL)

        self.record_process = multiprocessing.Process(target=record_audio)
        self.transcribe_process = multiprocessing.Process(target=process_new_audio_files, args=(self.q,))

    def update_model_name(self):
        self.q.put(self.model_selection_combobox.currentText())

    def start_processes(self):
        self.record_process.start()
        self.transcribe_process.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_processes(self):
        self.record_process.terminate()
        self.transcribe_process.terminate()

        self.record_process.join()
        self.transcribe_process.join()

        print("All processes have been terminated!")

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
