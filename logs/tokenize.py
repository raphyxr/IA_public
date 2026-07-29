import re
import unicodedata
from itertools import islice





print("lancement")



def lire_par_blocs(fichier, taille_bloc=1000):
    with open(fichier, "r", encoding="utf-8") as f:
        while True:
            bloc = list(islice(f, taille_bloc))
            if not bloc:
                break
            yield bloc


def tokenize(text):
    print ("tokenisation...")
    text_norm = unicodedata.normalize("NFKC", text)
    text_min = text_norm.lower()
    pattern = r"[^\W\d_]+(?:'[^\W\d_]+)?"
    tokens = re.findall(pattern, text_min)
    return tokens




# Utilisation
for bloc in lire_par_blocs("dataset.txt", 1000):
    
    print("Bloc chargé :", len(bloc))
    txt_token = tokenize(" ".join(bloc))
    with open("dataset_token.txt", "a", encoding="utf-8") as f:
        f.write(" ".join(txt_token) + "\n") 










