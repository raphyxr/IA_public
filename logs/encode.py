from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("Le chat dort sur le canapé")

print(len(embedding))  # dimension du vecteur
print(embedding[:5])   # premières valeurs