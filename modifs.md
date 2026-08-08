# Modifications de `model_streaming_v3.py`

Les numéros de lignes correspondent au fichier dans son état actuel. Après une
modification, les lignes suivantes peuvent se décaler : recherche aussi le
texte indiqué avant de modifier.

## 1. Faire parcourir toutes les données

**Ligne 19** — remplace :

```python
MAX_TOKENS_PER_EPOCH = 500000
```

par :

```python
MAX_TOKENS_PER_EPOCH = None
```

**Pourquoi :** le programme s'arrête après les 500 000 premiers tokens, puis
recommence au début du dataset à chaque époque. Il ne voit donc jamais tout le
corpus. Une époque deviendra plus longue, mais l'entraînement sera correct.

---

## 2. Ajouter la position des tokens

Sans positions, le Transformer ne connaît pas l'ordre des mots. Par exemple,
`chat mange poisson` et `poisson mange chat` ont presque la même information
pour lui. Cette modification lui indique la position 0, 1, 2, etc. de chaque
token dans le contexte.

### 2.1 Créer l'embedding de position

**Après la ligne 547**, juste après :

```python
emb_layer = nn.Embedding(vocab_size, emb_dim)
```

ajoute :

```python
pos_embedding = nn.Embedding(context_size, emb_dim)
```

### 2.2 L'utiliser pendant l'entraînement d'un batch

**Ligne 307** — remplace :

```python
def prediction(X_batch, y_batch, emb_layer, transformer, fc, optimizer, criterion):
```

par :

```python
def prediction(X_batch, y_batch, emb_layer, pos_embedding, transformer, fc, optimizer, criterion):
```

**Ligne 312** — remplace :

```python
vecs = embedding(X_batch, emb_layer)
```

par :

```python
positions = torch.arange(X_batch.size(1), device=device).unsqueeze(0)
vecs = embedding(X_batch, emb_layer) + pos_embedding(positions)
```

### 2.3 Transmettre `pos_embedding` à `train_streaming`

**Après la ligne 429**, ajoute :

```python
    pos_embedding,  # Embeddings qui indiquent la position des tokens
```

Le début de la fonction doit devenir :

```python
def train_streaming(
    path,
    emb_layer,
    pos_embedding,
    transformer,
```

**Après la ligne 445**, ajoute :

```python
    pos_embedding.train()
```

**Ligne 449** — remplace :

```python
verifier_modules_sur_device(device, emb_layer, transformer, fc)
```

par :

```python
verifier_modules_sur_device(device, emb_layer, pos_embedding, transformer, fc)
```

**Lignes 473 et 497** — aux deux endroits, remplace :

```python
prediction(X_batch, y_batch, emb_layer, transformer, fc, optimizer, criterion)
```

par :

```python
prediction(X_batch, y_batch, emb_layer, pos_embedding, transformer, fc, optimizer, criterion)
```

**Après la ligne 638**, dans l'appel à `train_streaming`, ajoute :

```python
            pos_embedding,
```

Le début de cet appel doit devenir :

```python
            DATASET_IDS_PATH,
            emb_layer,
            pos_embedding,
            transformer,
```

### 2.4 Ajouter ses paramètres à Adam et le déplacer sur le GPU

**Ligne 560** — remplace :

```python
list(emb_layer.parameters()) + list(transformer.parameters()) + list(fc.parameters()),
```

par :

```python
list(emb_layer.parameters()) + list(pos_embedding.parameters()) + list(transformer.parameters()) + list(fc.parameters()),
```

**Après la ligne 601**, ajoute :

```python
            pos_embedding.to(device)
```

**Après la ligne 622**, ajoute :

```python
    pos_embedding.to(device)
```

**Ligne 625** — remplace :

```python
verifier_modules_sur_device(device, emb_layer, transformer, fc)
```

par :

```python
verifier_modules_sur_device(device, emb_layer, pos_embedding, transformer, fc)
```

### 2.5 L'utiliser pour le test et la génération

**Ligne 672** — remplace :

```python
vecs = embedding(x_ids, emb_layer)
```

par :

```python
positions = torch.arange(x_ids.size(1), device=device).unsqueeze(0)
vecs = embedding(x_ids, emb_layer) + pos_embedding(positions)
```

**Ligne 703** — fais exactement le même remplacement. La génération doit
utiliser le même calcul que l'entraînement.

### 2.6 Démarrer avec un checkpoint neuf

**Ligne 30** — pour le premier entraînement après cette modification,
remplace :

```python
RESUME_IF_CHECKPOINT_EXISTS = True
```

par :

```python
RESUME_IF_CHECKPOINT_EXISTS = False
```

**Pourquoi :** l'ancien checkpoint ne contient pas `pos_embedding` et ne
correspond donc plus à l'architecture. Après avoir sauvegardé le nouveau
checkpoint, tu pourras remettre `True`.

---

## 3. Ne pas réduire le vocabulaire tant que le modulo existe

Ne modifie pas uniquement `VOCAB_LIMIT`. La ligne 42 contient :

```python
return token_id % vocab_limit
```

Cette ligne peut transformer plusieurs tokens différents en un même ID quand
leurs IDs dépassent la limite. Le modèle reçoit alors des réponses attendues
contradictoires, ce qui limite la baisse de la loss.

La correction complète doit se faire dans le script qui crée `dataset_ids.txt`
: il faut construire un vocabulaire par fréquence et remplacer les tokens rares
par un ID spécial, par exemple `<unk>`. Ne remplace pas le modulo par une valeur
au hasard.

---

## Comment vérifier le résultat

Regarde la valeur `epoch N loss` : c'est la moyenne de l'époque. Ne conclus pas
à partir de la loss d'un seul chunk, car elle varie beaucoup selon les phrases
contenues dans ce morceau des données.
