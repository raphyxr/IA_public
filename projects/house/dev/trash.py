import json
import pandas as pd


def encode(colonnes, chemin_json):
    with open(chemin_json, "r", encoding="utf-8") as f:
        dico = json.load(f)

    # Évite les doublons d'ID si le dictionnaire a été modifié.
    prochain_id = max(map(int, dico.values()), default=0) + 1
    ids = [] # liste des encodages 
    modifie = False # true: on modifie, false, il n'y a rien a modifier

    for token in colonnes: # pour chaque colonne de la ligne
        token = str(token).strip() # on enleve les espaces avant et apres la valeur
        if token not in dico: # si le token n'est pas dans le dico
            dico[token] = prochain_id # on ajoute le mot a la liste des mots a ajouter 
            prochain_id += 1 #et on ajoute 1 pour l'id suivant
            modifie = True # on indique qu'on a des modifs a faire 

        ids.append(int(dico[token])) # on ajoute l'id a la liste des mots decodés

    if modifie: # si il y a besoin de modifier
        with open(chemin_json, "w", encoding="utf-8") as f: # on ouvre le fichier json pour ajouter le modifs
            json.dump(dico, f, ensure_ascii=False, indent=2) # on y ajoute les ids en plus

    return ids # on renvoi la liste encodée des mots


def selectionner(y):
    """ selectionne une ligne de données"""
    global df
    return df.iloc[y].astype(str).tolist()

if __name__ == "__main__":
    df = pd.read_csv("data.csv", quotechar='"', skipinitialspace=True)
    for i in range(len(df)):
        colonnes = selectionner(i)
        e = encode(colonnes, "dico.json")
        print(e)

