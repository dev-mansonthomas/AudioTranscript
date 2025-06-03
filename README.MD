# 🎹 Local Speaker Diarization + Transcription (Whisper + PyAnnote)

Transcribe and diarize any `.m4a` audio file **locally** using OpenAI Whisper and PyAnnote.  
Supports multi-speaker detection and generates a **clean, interactive HTML transcript**.

---

## 🔧 Prerequisites

### ✅ Install `ffmpeg`

For macOS:

````bash
brew install ffmpeg
````

For Ubuntu:

````bash
sudo apt install ffmpeg
````

---

## 🐍 Python Setup (Use Python 3.12.x)

> ❗ **OpenAI Whisper is not compatible with Python 3.13. Use 3.12.x instead.**

### Install build tools (Ubuntu):

````bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
llvm libncursesw5-dev xz-utils tk-dev \
libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git
````

### Install `pyenv`

````bash
curl https://pyenv.run | bash
````

Add to your `.bashrc` or `.zshrc`:

````bash
# PyEnv (virtual env for Python)
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"
````

Then restart your shell:

````bash
exec "$SHELL"
````

---

## 🧪 Create Python Environment

````bash
pyenv install 3.12.10
pyenv local 3.12.10

python -m venv .venv
source .venv/bin/activate
````

Upgrade pip:

````bash
pip install -U pip
````

Install dependencies:

```bash
pip install openai-whisper faster-whisper
pip install pyannote.audio torchaudio huggingface_hub
pip install -r requirements.txt
```

Check if the graphic card is recognized by torch : 
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"

NVIDIA GeForce RTX 4090
```
---

## 🔐 Hugging Face Setup

1. Create a Hugging Face account: https://huggingface.co
2. Create a **read token**:  
   https://huggingface.co/settings/tokens

3. Authenticate in terminal:

````bash
git config --global credential.helper store
huggingface-cli login
# Paste your token
# Accept to store credentials: Y
````

4. Accept model access:
    - https://huggingface.co/pyannote/speaker-diarization → **Click "Access repository"**
    - https://huggingface.co/pyannote/segmentation → **Click "Access repository"**

---

## ▶️ Run the Script

````bash
python src/transcribe_diarize.py ~/Downloads/path_to_audio_file.m4a
````

This will:
- Convert `.m4a` to `.wav` if needed
- Diarize the speakers
- Transcribe the content with Whisper
- Ask for speaker names
- Generate a **styled HTML transcript** with dynamic speaker labels

---

## ✨ Output

A `.html` file will be generated next to your audio file.  
Open it in any browser to read the transcript.

---