import os
import requests
from faster_whisper import WhisperModel

# -------------------------------
# 1. LOAD WHISPER MODEL (CPU)
# -------------------------------
model = WhisperModel("base", device="cpu", compute_type="int8")

# -------------------------------
# 2. AUDIO FILE PATH
# -------------------------------
audio_file = "C:/Users/Binayak/Desktop/lecture.ai/sample2.mp3"

# Extract file name (without extension)
base_name = os.path.splitext(os.path.basename(audio_file))[0]

# Dynamic output file names
output_file = f"{base_name}_output.txt"
summary_file = f"{base_name}_summary.txt"

# -------------------------------
# 3. TRANSCRIBE + TRANSLATE TO ENGLISH
# -------------------------------
segments, info = model.transcribe(audio_file, task="translate")

print("Detected language:", info.language)
print("\n--- TRANSCRIPT (ENGLISH) ---\n")

full_text = ""

for segment in segments:
    text = segment.text.strip()
    print(text)
    full_text += text + " "

# Save transcript
with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\n✅ Saved transcript to {output_file}")


# -------------------------------
# 4. SUMMARIZATION FUNCTION
# -------------------------------
def summarize_text(text):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": f"""
You are a smart study assistant.

The transcript may contain translated content from Hindi or mixed language.

Convert everything into clean, structured English notes.

From the lecture transcript below, extract:

1. Short Summary
2. Key Points (bullet points)
3. Important Concepts
4. Possible Exam Questions

Transcript:
{text}
""",
            "stream": False
        }
    )

    return response.json()["response"]


# -------------------------------
# 5. SPLIT TEXT INTO CHUNKS
# -------------------------------
def split_text(text, chunk_size=2000):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------------------
# 6. PROCESS IN CHUNKS
# -------------------------------
print("\n--- GENERATING SUMMARY (CHUNKED) ---\n")

chunks = split_text(full_text)

all_summaries = []

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}...")
    summary = summarize_text(chunk)
    all_summaries.append(summary)


# -------------------------------
# 7. FINAL COMBINED SUMMARY
# -------------------------------
final_summary = "\n\n".join(all_summaries)

print("\n--- FINAL SUMMARY ---\n")
print(final_summary)

# Save summary
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(final_summary)

print(f"\n✅ Saved summary to {summary_file}")