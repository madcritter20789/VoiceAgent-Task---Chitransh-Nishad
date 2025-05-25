from TTS.api import TTS
import os

# Load TTS model (LJSpeech or others)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

def speak(text: str, filename="output.wav"):
    tts.tts_to_file(text=text, file_path=filename)
    os.system(f"ffplay -nodisp -autoexit {filename}")  # Requires ffmpeg
