import sounddevice as sd
import soundfile as sf

def record_voice(filename="input.wav", duration=5):
    samplerate = 44100
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print("✅ Recording saved.")
    return filename
