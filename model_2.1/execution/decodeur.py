import json


def decode(ids, jsons):
    ph = ""

    with open(jsons, "r", encoding="utf-8") as f:
        dico = json.load(f)

    id_to_mot = {int(v): k for k, v in dico.items()}

    for i in str(ids).split():
        mot = id_to_mot.get(int(i), "")
        ph = ph + " " + mot

    ph = ph.strip()
    
    return ph

def encode(mot, jsons):
    cles = []

    with open(jsons, "r", encoding="utf-8") as f:
        dico = json.load(f)

    for i in mot.split():
        cle = dico.get(i)
        if cle is not None:
            cles.append(int(cle))


    return cles


    

# Test decode (chercher des valeurs par clés)
#decode("hello world cat yes", "test_data.json")
# Résultat attendu: "bonjour monde"

# Test encodeur (chercher des clés par valeurs)
#encode("bonjour monde chat oui", "test_data.json")
# Résultat attendu: "hello world"
