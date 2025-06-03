brew install ffmpeg

Openai-Whisper is not working with Python 3.13, so use 3.12.x instead

pyenv install 3.12.10
pyenv local 3.12.10
python -m venv .venv
source .venv/bin/activate

pip install -U pip

pip install openai-whisper faster-whisper
pip install pyannote.audio torchaudio huggingface_hub


Create HuggingFace account and create a read token:
https://huggingface.co/settings/tokens

huggingface-cli login
paste your token
Add to git credentials: Y

open
https://huggingface.co/pyannote/speaker-diarization
Click on "Access repository" button 

idem:
https://huggingface.co/pyannote/segmentation


python src/transcribe_diarize.py ~/Downloads/path_to_audio_file.m4a