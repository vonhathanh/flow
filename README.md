# Hello world

- A free, cross-platform app that automatically write what you speak to current opening window
- Context: I was learning C and tired of having to write notes all the time, so I decided to create this app. \
  I also saw some similar apps, but they are not free, so I decided to make a free version for myself and also test the
  new whisper model by OpenAI :)
- I hope this app will make me a better English speaker as speaking skill is my weakness :)
- I'm using both Windows and Ubuntu so Windows will be the first targeted platform, it will be easy to port to other
  OS (in my dream)
- For future release, I want to make a C/C++ version of this app for learning how to program in C/C++ :)

# Installation

- Requirements: Python 3.10+, Windows 10, atleast 8GB of RAM
- You need to install python libraries first: `pip install -r requirements.txt`

## Windows
- Install chocolatey: https://chocolatey.org/install
- Install ffmpeg: run Power Shell with administrator: `choco install ffmpeg`
- Install Rust: `pip install setuptools-rust`
- You are good to go :)

# Notes

- The default model is "small", we may need to change it later if the voice detection's accuracy is bad
- I have no knowledge about audio/speech processing, so helps/advice from other folks are welcomes

# Todos

- ~~Delete audio files after transcribe~~
- Remove hardcoded values
- Better text aligment
- Add support for Ubuntu, MacOS, Iphone, Android, Windows Phone, Rasberry Pi, Andruino
- Try other apps and compare the performance/accuracy
- Test the voice detection accuracies 
- Auto use GPU if possible
- Auto detect RAM and choose the most optimal model size
- Test model with math symbols
- Listen for sound and transcribe concurrently
