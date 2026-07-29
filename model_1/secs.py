print("démarrage...")
import pandas as pd
print("import 1 done")
from sklearn.feature_extraction.text import TfidfVectorizer
print("import 2 done")
from sklearn.svm import LinearSVC
print("import 3 done")
from rapidfuzz import process, fuzz
print("import 4 done")
import re
print("import 5 done")
import random
import csv
# ---------------------------
#  Charger le modèle depuis CSV
# ---------------------------
textes = []
labels = []
classe = []
def charger_modele():
    global data, vectorizer, model

    data = pd.read_csv("dataset.csv")
    global textes
    global labels
    global classe
    textes = data["texte"].astype(str).tolist()
    labels = data["classe"].astype(str).tolist()
    classe = data["pos_tag"].astype(str).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X = vectorizer.fit_transform(textes)

    model = LinearSVC()
    model.fit(X, labels)


# ---------------------------
# Ajouter un exemple dans le CSV
# ---------------------------
def ajouter_exemple(texte, classe):
    global data

    nouveau = pd.DataFrame({"texte": [texte], "classe": [classe]})
    data = pd.concat([data, nouveau], ignore_index=True)

    data.to_csv("dataset.csv", index=False)
    print(f"✔ Exemple ajouté : {texte} → {classe}")


# ---------------------------
# Trouver un mot proche dans le CSV (tolérance fautes)
# ---------------------------
def trouver_mot_proche(message, seuil=70):
    liste_mots = data["texte"].tolist()

    resultat = process.extractOne(
        message,
        liste_mots,
        scorer=fuzz.ratio
    )

    if resultat:
        mot_proche, score, index = resultat
        if score >= seuil:
            categorie = data.loc[index, "classe"]
            return mot_proche, categorie
    
    return None, None


# ---------------------------
# Découper une phrase en mots
# ---------------------------
def decouper_phrase(phrase):
    phrase = phrase.lower()
    phrase = re.sub(r"[^a-zA-Z0-9éèàùç ]", " ", phrase)
    return phrase.split()


# ---------------------------
# Catégoriser UN mot avec ton système complet
# ---------------------------
def analyser_mot(mot):
    # 1 : prédiction ML
    vect = vectorizer.transform([mot])
    categorie_predite = model.predict(vect)[0]

    # 2 : fuzzy matching
    mot_proche, categorie_proche = trouver_mot_proche(mot)

    # Si fuzzy a trouvé un mot similaire
    if mot_proche:
        print(f" (mot '{mot}' similaire à '{mot_proche}' → {categorie_proche})")

        # Si les deux catégories concordent → apprentissage auto
        
    return categorie_predite

def corrige(mot):
    global categorie_predite
    global categorie_proche
    analyser_mot(mot)
    if categorie_proche == categorie_predite:
            if mot not in data["texte"].values:
                ajouter_exemple(mot, categorie_predite)
                charger_modele()

# ---------------------------
# PROGRAMME PRINCIPAL
# ---------------------------
print("Chargement du modèle...")
charger_modele()
print("IA entraînée avec tolérance aux fautes et analyse de phrase !")

while True:
    message = input("\nÉcris une phrase (ou 'n' pour reload, 'q' pour quitter) : ").lower()

    if message == "n":
        print("🔄 Ré-entraînement...")
        charger_modele()
        continue

    if message == "q":
        print("Au revoir !")
        break

    # Découper la phrase
    mots = decouper_phrase(message)

    print("\n🔎 Analyse des mots :")
    resultats = []

    for mot in mots:
        c = analyser_mot(mot)
        
        if c == "nom":
            print("nom trouvé")
            nom = analyser_mot(mot)
            print(nom)

            

        if c == "article":
            print(f" → {mot} :    Ignoré car catégorie article")
            article = analyser_mot(mot)
            print(article)
            
            
        resultats.append((mot, c))
        print(f" → {mot} : {c}")
    








    # Vérification utilisateur
    correction = input("\nEst-ce correct ? (o/n) : ").lower()
    if correction == "n":
        mot_cible = input("Quel mot est incorrect ? : ").lower()
        bonne_cat = input("Quelle est sa vraie catégorie ? : ").lower()

        ajouter_exemple(mot_cible, bonne_cat)
        print("🔄 Ré-entraînement du modèle...")
        charger_modele()
        print("✔ Correction prise en compte !")
    else:
        print("OOOKKKKAI")

































while gen == True:
        print("generation")
        with open("noms.csv", newline="", encoding="utf-8") as f:
             mots = [row[0] for row in csv.reader(f)]
        nom = random.choice(mots)
        print(nom)
        with open("pron.csv", newline="", encoding="utf-8") as f:
             mots = [row[0] for row in csv.reader(f)]
        arti = random.choice(mots)
        print(arti)
        with open("verbes.csv", newline="", encoding="utf-8") as f:
             mots = [row[0] for row in csv.reader(f)]
        verbe = random.choice(mots)
        print(verbe)
        phrase = arti + " " + verbe + " " + nom
        print(phrase)
        decomp = decouper_phrase(phrase)
        print(decomp)
        print("analyse des mots générés :")
        for mot in decomp:
            categorie = analyser_mot(mot)
            print(categorie)
            if categorie == "article":
                print("nom trouvé :", mot)
                gen = False
        
