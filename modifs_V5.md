## 6. Le modulo de compression du vocabulaire n'est toujours pas corrigé

### Problème

La fonction `compress_id` contient encore :

```python
return token_id % vocab_limit
```

Elle est appelée pendant la lecture du dataset. Deux tokens distincts peuvent
alors obtenir le même ID ; le modèle reçoit des cibles contradictoires.

### Correction à faire dans le créateur du dataset

Cette correction ne consiste pas seulement à modifier `VOCAB_LIMIT`. Il faut :

1. compter la fréquence de chaque token du corpus ;
2. conserver les `VOCAB_LIMIT - 1` tokens les plus fréquents ;
3. réserver un ID, par exemple `0`, pour `<unk>` ;
4. remplacer les tokens rares par `<unk>` dans `dataset_ids.txt` ;
5. enregistrer le même vocabulaire dans `vocab.json`.

Une fois `dataset_ids.txt` reconstruit sans ID hors vocabulaire, modifie le
lecteur du script.

**Aux lignes 40 à 42**, supprime entièrement :

```python
def compress_id(token_id, vocab_limit=VOCAB_LIMIT):
    """Compresse un ID brut dans [0, vocab_limit-1] pour limiter la RAM."""
    return token_id % vocab_limit
```

**À la ligne 64**, remplace :

```python
yield compress_id(int(p))
```

par :

```python
yield int(p)
```

**À la ligne 71**, remplace :

```python
yield compress_id(int(reste))
```

par :

```python
yield int(reste)
```

### Pourquoi

Tant que le modulo est actif, l'apprentissage et les phrases générées restent
limités par les collisions entre tokens.

---

## 7. Choisir le chemin de checkpoint adapté à Kaggle

### Problème

Sur Kaggle, le modèle est sauvegardé dans `/kaggle/working`, mais le chemin de
chargement pointe vers le dossier du script. Le checkpoint qui vient d'être créé
n'est donc pas automatiquement celui qui sera repris.

### Correction

**À la ligne 28**, seulement si tu veux reprendre le checkpoint du même
environnement, remplace :

```python
CHECKPOINT_LOAD_PATH = os.path.join(SCRIPT_DIR, "modelV3.0.pt")
```

par :

```python
CHECKPOINT_LOAD_PATH = CHECKPOINT_SAVE_PATH
```

### Pourquoi

Sur une nouvelle session Kaggle, `/kaggle/working` est temporaire. Il faut donc
d'abord exporter le checkpoint, l'importer comme dataset, puis définir
`CHECKPOINT_LOAD_PATH` vers le fichier importé.

---

## 8. Régler le chemin de données selon l'environnement

Le script utilise actuellement un chemin Kaggle :

```python
BASE_DIR = "/kaggle/input/datasets/bgbg1000/model-v-3/"
```

**À la ligne 22**, pour une exécution locale dans le dossier Windows du projet,
remplace cette ligne par :

```python
BASE_DIR = ""
```

Le script cherchera alors `dataset-ids.txt` ou `dataset_ids.txt` à côté de
`model_streaming_v3.py`. Sur Kaggle, conserve le chemin Kaggle réel.

---

## Résumé

Les priorités sont de recharger `pos_embedding` pendant une reprise, de
reconstruire le vocabulaire sans modulo, de tester le modèle en mode évaluation
et de générer avec les 128 tokens de contexte attendus.
