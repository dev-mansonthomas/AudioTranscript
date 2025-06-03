import sys
import subprocess
from pathlib import Path
from pyannote.audio import Pipeline
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import whisper
import json
import torchaudio

def convert_to_wav_if_needed(input_path: str) -> str:
    input_path = Path(input_path)
    if input_path.suffix.lower() != ".wav":
        output_path = input_path.with_suffix(".wav")
        print(f"🔄 Converting {input_path.name} to WAV for diarization...")
        subprocess.run(["ffmpeg", "-y", "-i", str(input_path), str(output_path)], check=True)
        return str(output_path)
    return str(input_path)

def diarize_and_transcribe(audio_path: str):
    print("🔊 Loading diarization model...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=True
    )
    diarization_pipeline.to(torch.device("cuda"))
    wav_path = convert_to_wav_if_needed(audio_path)

    print("🧠 Running diarization...")
    num_speakers = input("🔢 Enter the expected number of speakers (leave blank for automatic detection): ").strip()
    waveform, sample_rate = torchaudio.load(wav_path)
    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

    if num_speakers.isdigit():
        diarization = diarization_pipeline(audio_dict, num_speakers=int(num_speakers))
    else:
        diarization = diarization_pipeline(audio_dict)

    print("✍️ Transcribing with Whisper...")
    model = whisper.load_model("large-v3", device="cuda")
    transcript = model.transcribe(audio_path, verbose=False)

    print("🪄 Combining diarization and transcript...")
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        text = " ".join(
            seg["text"].strip()
            for seg in transcript["segments"]
            if turn.start <= seg["start"] <= turn.end
        )
        if text:
            speaker_segments.append((speaker, text))

    return speaker_segments

def ask_for_speaker_names(unique_speakers):
    speaker_names = {}
    print("\n👤 Please enter a name for each speaker:")
    for i, speaker in enumerate(unique_speakers):
        name = input(f"Name for {speaker} (e.g. CEO, CTO, John...): ").strip()
        speaker_names[speaker] = name or f"Speaker {i+1}"
    return speaker_names

def write_html(output_path: Path, speaker_segments, speaker_names):
    print(f"📝 Writing HTML to {output_path}...")

    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>Transcript</title>",
        "  <style>",
        "    body { font-family: sans-serif; padding: 20px; background: #f4f4f4; }",
        "    .speaker { font-weight: bold; color: #333; margin-top: 1em; }",
        "    .block { background: white; border-left: 5px solid #007BFF; border-radius: 8px; padding: 10px 15px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "  </style>",
        "</head>",
        "<body>",
        "<h1>Transcript</h1>",
    ]

    for speaker, text in speaker_segments:
        color = hash(speaker) % 360  # unique color per speaker
        html_lines.append(
            f"<div class='block' style='border-left-color:hsl({color},70%,50%)'><div class='speaker' data-speaker='{speaker}'></div>{text}</div>"
        )

    html_lines.append(f"""
<script>
  const speakerNames = {json.dumps(speaker_names)};
  window.addEventListener('DOMContentLoaded', () => {{
    document.querySelectorAll('[data-speaker]').forEach(el => {{
      const id = el.getAttribute('data-speaker');
      el.innerText = speakerNames[id] || id;
    }});
  }});
</script>
    """)

    html_lines.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

def main():
    if len(sys.argv) != 2:
        print("Usage: python transcribe_to_html.py path/to/audiofile.m4a")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    speaker_segments = diarize_and_transcribe(str(audio_path))
    unique_speakers = sorted(set(s for s, _ in speaker_segments))
    speaker_names = ask_for_speaker_names(unique_speakers)

    output_path = audio_path.with_suffix(".html")
    write_html(output_path, speaker_segments, speaker_names)

    print(f"\n✅ Done! Transcript saved to: {output_path}")

if __name__ == "__main__":
    main()