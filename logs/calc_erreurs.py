import re
import json
from tqdm import tqdm

input_file = "vocab.json"
output_file = "vocab_clean.json"

vocab = {}
buffer = ""

chunk_size = 1024 * 1024  # 1 MB

with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    total_size = f.seek(0, 2)
    f.seek(0)

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Lecture") as pbar:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            buffer += chunk

            # 🔍 extraction des paires valides
            pairs = re.findall(r'"(.*?)"\s*:\s*(\d+)', buffer)

            for k, v in pairs:
                vocab[k] = int(v)

            # 🧹 éviter que le buffer devienne trop gros
            buffer = buffer[-1000:]

            pbar.update(len(chunk))

print(f"✅ Tokens récupérés : {len(vocab)}")

# 💾 sauvegarde
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print("✅ Fichier sauvegardé :", output_file)