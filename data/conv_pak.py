import pandas as pd
import re
import unicodedata

# Nettoyage
def clean(text):
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9.,!?;:'\"()\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Charger le parquet
df = pd.read_parquet("dataset.parquet")

print(df.columns)  # IMPORTANT pour voir les noms exacts

with open("dataset.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():

        # ⚠️ adapte les noms ici selon ton dataset
        user = clean(row.get("prompt", ""))
        assistant = clean(row.get("response", ""))

        if len(user) > 10 and len(assistant) > 10:
            f.write(f"User: {user}\n")
            f.write(f"Assistant: {assistant}\n\n")