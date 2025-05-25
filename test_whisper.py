import whisper

model = whisper.load_model("base")
result = model.transcribe("sample.wav")
print(result["text"])