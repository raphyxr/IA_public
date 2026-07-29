import json
from itertools import islice

input_file = "dataset_tokenpp.txt"
ids_file = "dataset_ids.txt"
vocab_file = "vocab.json"

def lire_blocs_lignes(fichier, taille_bloc=1000):
    with open(fichier, "r", encoding="utf-8") as f:
        while True:
            lignes = list(islice(f, taille_bloc))
            if not lignes:
                break
            yield lignes

token2id = {}

# reset fichier de sortie
with open(ids_file, "w", encoding="utf-8") as f_out:
    first = True
    for bloc in lire_blocs_lignes(input_file, 1000):
        tokens = " ".join(bloc).split()
        print("1000 LIGNES")

        ids_bloc = []
        for tok in tokens:
            if tok not in token2id:
                token2id[tok] = len(token2id)
            ids_bloc.append(token2id[tok])

        texte_ids = " ".join(map(str, ids_bloc))
        if not first and texte_ids:
            f_out.write(" ")
        f_out.write(texte_ids)
        first = False

with open(vocab_file, "w", encoding="utf-8") as f_vocab:
    json.dump(token2id, f_vocab, ensure_ascii=False, indent=2)

print("Conversion terminee.")
