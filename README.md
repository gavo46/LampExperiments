LampExperiments

A 5-DOF desk lamp that behaves like a character. It notices when someone is present and tracks their face, holds a spoken conversation, remembers objects it is shown, and nods along to music. All inference runs locally.

Design notes: documentation/DESIGN.md — architecture, design decisions, measurements, and limitations.

Requirements

macOS (Apple Silicon); see the design notes for Linux/Ubuntu deployment notes
Python 3.13
Ollama running locally
ffmpeg (Whisper depends on it)

Setup

bash
# system dependencies
brew install ffmpeg ollama
brew services start ollama

# models
ollama pull llama3.2
ollama pull <vision-model>

# python dependencies
pip install -r requirements.txt

The face detection model (vision/blaze_face_short_range.tflite) is included in the repo.

Running

bash
mjpython main.py

mjpython rather than python — MuJoCo's interactive viewer requires it on macOS.
A MuJoCo window opens with the lamp. Camera and microphone permissions are requested on first run; grant both or engagement and speech will not work.

Interacting

What you do	What happens
Enter the camera's view	The lamp notices you and tracks your face
Speak	It transcribes, responds, and speaks back
Hold up an object and say "look at this"	It describes and remembers the object
Ask about it later	It answers from memory
Say "let's jam"	Music plays and the lamp nods along
Say "stop the music"	Playback stops
Leave the frame	It disengages

Configuration

config.py holds the tunable values — camera index, audio thresholds, and MEASURE, which toggles the instrumentation used to produce the numbers in the design notes.

Project structure

character/     behaviour, state, light, sound, music
control/       pose definitions and joint interpolation
speech/        listening, thinking, speaking
vision/        camera, face tracking, scene description
robot/         URDF and mesh assets
utils/         instrumentation
main.py        entry point

Notes
[Anything a reader needs to know before running — the music asset, known rough edges, whatever you want to flag up front.]
