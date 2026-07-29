import json


def encode(mot, jsons):
    with open(jsons, "r", encoding="utf-8") as f:
        dico = json.load(f)

    ids = []
    next_id = max((int(v) for v in dico.values()), default=0) + 1

    for token in mot.split():
        if token in dico:
            ids.append(int(dico[token]))
        else:
            dico[token] = next_id
            ids.append(next_id)
            next_id += 1

    with open(jsons, "w", encoding="utf-8") as f:
        json.dump(dico, f, ensure_ascii=False, indent=2)

    return ids


if __name__ == "__main__":
    with open("dataset.txt", "r", encoding="utf-8") as f:
        fichier = f.read()

    idsss = encode(fichier, "vocab.json")
    ids_text = " ".join(str(i) for i in idsss)

    with open("dataset-i.txt", "w", encoding="utf-8") as f:
        f.write(ids_text)
        
