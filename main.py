from stt import transcribe
from tts import speak
from llm_agent import ask_llm
from rag_pipeline import load_docs, embed_docs, retrieve_context
from utils import record_voice

# Load assistant prompt
with open("prompt.txt", encoding="utf-8") as f:
    agent_prompt = f.read()

# Load and embed docs
docs = load_docs("docs/Agenta.pdf")
db = embed_docs(docs)

print("\n🎙️ Voice Assistant Started! Press Ctrl+C to stop.\n")

while True:
    try:
        record_voice()
        user_input = transcribe("input.wav")
        print("👤 User:", user_input)

        context = retrieve_context(user_input, db)
        response = ask_llm(agent_prompt, user_input, context)

        print("🤖 Nira:", response)
        speak(response)
    except KeyboardInterrupt:
        print("🔚 Session ended.")
        break
