import json
import re
import os
from tqdm import tqdm

# 📂 fichiers
VOCAB_PATH = "vocab_clean.json"
TEXT_PATH = "dataset.txt"
OUTPUT_PATH = "vocab_final.json"

# 🧠 1. Charger vocab existant
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

print("Tokens existants :", len(vocab))

# 🔢 2. Prochain ID
next_id = max(vocab.values()) + 1 if vocab else 0

# 🔍 tokenizer simple
def extract_tokens(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# 📏 taille du fichier pour la progression
file_size = os.path.getsize(TEXT_PATH)

new_tokens = 0

# 📖 3. Lecture avec barre de progression (par taille réelle)
with open(TEXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Analyse du dataset") as pbar:
        
        for line in f:
            tokens = extract_tokens(line)
            
            for t in tokens:
                if t not in vocab:
                    vocab[t] = next_id
                    next_id += 1
                    new_tokens += 1
            
            # mise à jour barre = taille de la ligne lue
            pbar.update(len(line.encode("utf-8")))
            
            # 🔥 affiche infos en live
            pbar.set_postfix({
                "tokens": len(vocab),
                "ajoutés": new_tokens
            })

print("\n✅ Nouveaux tokens ajoutés :", new_tokens)
print("📊 Taille finale vocab :", len(vocab))

# 💾 4. Sauvegarde
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print("✅ vocab_final.json généré")