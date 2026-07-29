# On importe la bibliothèque principale pour le calcul tensoriel
import torch

# On importe les modules réseaux de neurones
import torch.nn as nn

# On importe les optimiseurs (algorithmes d'apprentissage)
import torch.optim as optim


# ----------------------
# 1. Création d’un petit dataset
# ----------------------

# Petite phrase d'entraînement
text = "il etait une fois un roi"

# On découpe la phrase en mots
words = text.split()

# On crée un vocabulaire unique (sans doublons)
vocab = list(set(words))

# On crée un dictionnaire mot -> index numérique
word_to_ix = {word: i for i, word in enumerate(vocab)}

# On crée l'inverse : index -> mot
ix_to_word = {i: word for word, i in word_to_ix.items()}

# Taille du vocabulaire
vocab_size = len(vocab)

# Liste qui va contenir les paires (mot actuel → mot suivant)
data = []

# Pour chaque mot sauf le dernier
for i in range(len(words) - 1):

    # On récupère l’index du mot courant
    input_word = word_to_ix[words[i]]

    # On récupère l’index du mot suivant
    target_word = word_to_ix[words[i + 1]]

    # On ajoute la paire dans les données
    data.append((input_word, target_word))


# ----------------------
# 2. Définition du modèle
# ----------------------

# Dimension des vecteurs d’embedding (taille du vecteur de chaque mot)
embedding_dim = 10


# On crée une classe de modèle
class SimpleModel(nn.Module):

    # Constructeur du modèle
    def __init__(self):

        # On appelle le constructeur parent
        super().__init__()

        # Couche embedding : transforme un index en vecteur
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Couche linéaire : transforme le vecteur en scores pour chaque mot du vocabulaire
        self.linear = nn.Linear(embedding_dim, vocab_size)

    # Définition du passage avant (forward pass)
    def forward(self, x):

        # On transforme l’index en vecteur dense
        x = self.embedding(x)

        # On transforme ce vecteur en scores pour chaque mot possible
        x = self.linear(x)

        # On retourne les scores (logits)
        return x


# On instancie le modèle
model = SimpleModel()


# ----------------------
# 3. Entraînement
# ----------------------

# Fonction de perte : compare la prédiction au vrai mot
loss_fn = nn.CrossEntropyLoss()

# Optimiseur Adam (algorithme d’apprentissage)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# On répète l’entraînement 300 fois
for epoch in range(300):

    # Variable pour suivre la perte totale
    total_loss = 0

    # Pour chaque paire (mot → mot suivant)
    for input_word, target_word in data:

        # On transforme l’index en tenseur
        input_tensor = torch.tensor([input_word])

        # On transforme la cible en tenseur
        target_tensor = torch.tensor([target_word])

        # -------- Forward pass --------

        # Le modèle fait une prédiction
        output = model(input_tensor)

        # On calcule l’erreur entre la prédiction et la vraie réponse
        loss = loss_fn(output, target_tensor)

        # -------- Backpropagation --------

        # On remet les gradients à zéro
        optimizer.zero_grad()

        # On calcule les gradients (dérivées)
        loss.backward()

        # On met à jour les poids (dont l’embedding)
        optimizer.step()

        # On ajoute la perte à la perte totale
        total_loss += loss.item()

    # On affiche la perte tous les 50 cycles
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


# Message de fin
print("Entraînement terminé !")


# ----------------------
# 4. Affichage des embeddings appris
# ----------------------

# Pour chaque mot du vocabulaire
for word in vocab:

    # On affiche le mot et son vecteur appris
    print(word, model.embedding.weight[word_to_ix[word]].detach())