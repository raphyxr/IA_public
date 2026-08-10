# Complément de corrections — `model_streaming_v3.py`

Ce fichier complète `modifs.md`. Les numéros de lignes visent la version
actuelle du script racine `model_streaming_v3.py` et peuvent se décaler après
chaque modification.

## 4. Transmettre `pos_embedding` à tous les appels de `prediction`

La fonction `prediction` attend maintenant l'argument `pos_embedding`, mais
deux appels dans `train_streaming` ne le transmettent pas. Sans cette correction,
l'entraînement s'arrête dès le premier batch avec une erreur d'argument.

**Lignes 477 et 501** — remplace, aux deux endroits :

```python
prediction(X_batch, y_batch, emb_layer, transformer, fc, optimizer, criterion)
```

par :

```python
prediction(X_batch, y_batch, emb_layer, pos_embedding, transformer, fc, optimizer, criterion)
```

## 5. Mettre `pos_embedding` sur le CPU ou GPU choisi

Tous les modules et tensors utilisés dans une même opération PyTorch doivent
être sur le même device. Sur GPU, l'oubli provoque une erreur ; sur CPU, il peut
passer inaperçu.

Après `emb_layer.to(device)`, ajoute `pos_embedding.to(device)` aux deux blocs
suivants :

- vers les lignes 606 à 608, dans le chargement de checkpoint ;
- vers les lignes 627 à 629, juste avant la vérification des modules.

Les deux blocs doivent être :

```python
emb_layer.to(device)
pos_embedding.to(device)
transformer.to(device)
fc.to(device)
```

## 6. Inclure les positions dans les checkpoints

Un checkpoint doit contenir tous les poids entraînables. Actuellement, il
sauvegarde `emb`, `transformer` et `fc`, mais pas `pos_embedding`. Une reprise
continue donc avec des positions aléatoires : ce n'est pas une vraie reprise.

### 6.1 Sauvegarde

**Vers la ligne 394**, remplace :

```python
def save_checkpoint(path, emb_layer, transformer, fc, optimizer=None, meta=None):
```

par :

```python
def save_checkpoint(path, emb_layer, pos_embedding, transformer, fc, optimizer=None, meta=None):
```

Puis remplace le début de `payload` par :

```python
payload = {
    "emb": emb_layer.state_dict(),
    "pos_embedding": pos_embedding.state_dict(),
    "transformer": transformer.state_dict(),
    "fc": fc.state_dict(),
}
```

### 6.2 Appels de sauvegarde

Aux appels vers les lignes 489, 513 et 662, ajoute `pos_embedding` juste après
`emb_layer`. Exemple :

```python
save_checkpoint(checkpoint_path, emb_layer, pos_embedding, transformer, fc,
                optimizer=optimizer, meta=meta)
```

### 6.3 Chargement

Après la ligne qui charge `ckpt["emb"]`, ajoute :

```python
pos_embedding.load_state_dict(ckpt["pos_embedding"])
```

La fonction utilitaire `load_checkpoint_if_compatible` doit recevoir elle aussi
`pos_embedding` et charger cette même clé si elle est utilisée dans un notebook.

Les anciens checkpoints n'ont pas cette clé. Pour le premier entraînement après
cette modification, utilise :

```python
RESUME_IF_CHECKPOINT_EXISTS = False
```

Une fois un nouveau checkpoint créé, tu peux remettre `True`.

## 7. Employer les positions pendant la génération

**Vers la ligne 710**, remplace :

```python
vecs = embedding(x_ids, emb_layer)
```

par :

```python
positions = torch.arange(x_ids.size(1), device=device).unsqueeze(0)
vecs = embedding(x_ids, emb_layer) + pos_embedding(positions)
```

L'entraînement, le test et la génération doivent effectuer le même calcul ;
sinon, le modèle apprend avec les positions mais prédit sans elles.

## 8. Corriger `init_training_objects` si la fonction est utilisée

Cette fonction crée encore une architecture sans embeddings de position. Après
la création de `emb_layer`, ajoute :

```python
pos_embedding = nn.Embedding(CONTEXT_SIZE, emb_dim)
```

Inclue ensuite ses paramètres dans `params` :

```python
params = (
    list(emb_layer.parameters())
    + list(pos_embedding.parameters())
    + list(transformer.parameters())
    + list(fc.parameters())
)
```

Et retourne le module :

```python
return emb_layer, pos_embedding, transformer, fc, optimizer, criterion
```

Cela modifie le nombre de valeurs retournées : les appels dans le notebook
doivent être adaptés.

La branche `else` de cette fonction crée actuellement Adam alors qu'elle est
prévue pour SGD. Remplace, dans cette branche :

```python
optimizer = torch.optim.Adam(params, lr=lr)
```

par :

```python
optimizer = torch.optim.SGD(params, lr=lr)
```

## 9. Vocabulaire : correction toujours nécessaire

La fonction `compress_id` contient toujours :

```python
return token_id % vocab_limit
```

Cette opération fusionne des tokens différents lorsqu'ils dépassent
`VOCAB_LIMIT`. Elle doit être supprimée dans la chaîne de création du dataset,
pas remplacée par une autre limite arbitraire. Reconstruis `dataset-ids.txt`
avec un vocabulaire fondé sur la fréquence et un ID spécial `<unk>` pour les
tokens rares. Tant que le modulo existe, l'apprentissage reçoit des cibles
contradictoires et la qualité restera limitée.

## 10. Choisir une seule copie du script

Les fichiers suivants ne sont pas équivalents :

```text
model_streaming_v3.py
model_3/model_streaming_v3.py
```

Les corrections ci-dessus concernent le fichier racine. La copie dans `model_3`
est plus ancienne : elle utilise encore `MAX_TOKENS_PER_EPOCH = 500000` et ne
contient pas d'embedding de position. Ne pas alterner entre les deux fichiers.

## 11. Exécution locale ou Kaggle

Le script racine définit actuellement :

```python
BASE_DIR = "/kaggle/input/datasets/bgbg1000/model-v-3/"
```

Pour une exécution dans le dossier local du projet, utilise :

```python
BASE_DIR = ""
```

Le script cherchera alors `dataset-ids.txt` ou `dataset_ids.txt` à côté du
script. Remets le chemin Kaggle approprié lorsque tu exécutes sur Kaggle.
