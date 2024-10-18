import multiprocessing
import time
import warnings
import sys
import pyautogui
import numpy as np
import whisper

from glob import glob
from os.path import join
from multiprocessing import Queue, Process

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QMainWindow, QVBoxLayout, QPushButton, QComboBox

from utils import process_recorded_speech, remove_audio, save_audio, is_silent
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


def reload_model(model_name: str, message_queue: Queue):
    global model

    model = whisper.load_model(model_name)

    message_queue.put(True)


def process_new_audio_files():
    while True:
        for file_name in sorted(glob(join(RECORDINGS_DIR, "*.wav"))):
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
        self.setWindowTitle("Voice Flow")
        self.setGeometry(100, 100, 200, 100)

        self.message_queue = Queue()

        # Set up timer to check the queue periodically
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_message_queue)
        self.timer.start(100)  # Check every 100ms

        layout = QVBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_processes)
        layout.addWidget(self.start_button)

        self.app_status = QLabel(AppStatus.READY)
        layout.addWidget(self.app_status)

        self.model_name = QLabel("Select model: ")
        layout.addWidget(self.model_name)

        self.model_selection_combobox = QComboBox()
        self.model_selection_combobox.addItems(AVAILABLE_MODELS)
        self.model_selection_combobox.activated.connect(self.reload_model)
        self.model_selection_combobox.setCurrentText(CURRENT_MODEL)
        layout.addWidget(self.model_selection_combobox)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processes)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.record_process: Process = None
        self.transcribe_process: Process = None
        self.model_loading_process: Process = None

        # start the app immediately
        self.start_processes()

    def check_message_queue(self):
        if not self.message_queue.empty():
            # clear the queue so we can reuse it next time
            self.message_queue.get()
            self.app_status.setText(AppStatus.READY)
            # stop the loading process to save resource
            self.model_loading_process.join()

    def reload_model(self):
        model_name = self.model_selection_combobox.currentText()

        self.app_status.setText(AppStatus.LOADING)
        # delegates the task of loading new model to another process so
        # the main GUI process wouldn't be blocked
        self.model_loading_process = Process(target=reload_model, args=(model_name, self.message_queue))
        self.model_loading_process.start()

        with open(CONFIG_FILE, "w") as f:
            json.dump({"model": model_name}, f)

    def start_processes(self):
        self.record_process = multiprocessing.Process(target=record_audio)
        self.transcribe_process = multiprocessing.Process(target=process_new_audio_files)

        self.record_process.start()
        self.transcribe_process.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.app_status.setText(AppStatus.READY)

    def stop_processes(self):
        self.record_process.terminate()
        self.transcribe_process.terminate()

        self.record_process.join()
        self.transcribe_process.join()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.app_status.setText(AppStatus.STOPPED)

    def closeEvent(self, ev):
        self.app_status.setText(AppStatus.CLOSING)
        if self.record_process.is_alive():
            self.record_process.terminate()
            self.transcribe_process.terminate()

            self.record_process.join()
            self.transcribe_process.join()

        print("All processes have been terminated!")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
