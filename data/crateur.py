from datasets import load_dataset
import re

# dataset moderne (sans script)
dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

def clean(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9.,!?;:'\"()\-\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

LIMIT = 1_000_000_000
size = 0

with open("dataset.txt", "w", encoding="utf-8") as f:
    for sample in dataset:
        text = sample["text"]

        # ⚠️ filtre français simple
        if " le " not in text and " la " not in text:
            continue

        text = clean(text)

        if len(text) < 50:
            continue

        f.write(text + "\n")
        size += len(text.encode("utf-8"))

        if size >= LIMIT:
            break

print("✅ dataset.txt créé (~1 Go)")