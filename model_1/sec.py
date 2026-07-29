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
gen = True
data_n = None
vectorizer_n = None
model_n = None
# ---------------------------
#  Charger le modèle depuis CSV
# ---------------------------
def charger_modele():
    global data, vectorizer, model

    data = pd.read_csv("dataset.csv")

    textes = data["texte"].astype(str).tolist()
    labels = data["classe"].astype(str).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X = vectorizer.fit_transform(textes)

    model = LinearSVC()
    model.fit(X, labels)
# ---------------------------
# Trouver un mot proche dans le CSV (tolérance fautes)
# ---------------------------
def trouver_mot_proche(mot, seuil=70):
    liste_mots = data["texte"].tolist()

    resultat = process.extractOne(
        mot,
        liste_mots,
        scorer=fuzz.ratio
    )

    if resultat:
        mot_proche, score, index = resultat
        if score >= seuil:
            return index

    return None

# ---------------------------
# Découper une phrase en mots
# ---------------------------
def decouper_phrase(phrase):
    phrase = phrase.lower()
    phrase = re.sub(r"[^a-zA-Z0-9éèàùç ]", " ", phrase)
    return phrase.split()

# ---------------------------
# Catégoriser UN mot
# (on NE met pas pos_tag ici exprès, pour rester fidèle à TA logique)
# ---------------------------
def analyser_mot(mot):
    vect = vectorizer.transform([mot])
    categorie_predite = model.predict(vect)[0]
    return categorie_predite


# ---------------------------
# PROGRAMME PRINCIPAL
# ---------------------------
print("Chargement du modèle...")
charger_modele()
print("IA entraînée avec analyse de phrase !")

while True:
    message = input("\nÉcris une phrase (ou 'n' pour reload, 'q' pour quitter) : ").lower()

    if message == "n":
        print("🔄 Ré-entraînement...")
        charger_modele()
        continue

    if message == "q":
        print("Au revoir !")
        break

    mots = decouper_phrase(message)

    print("\n🔎 Analyse des mots :")
    STOP_WORDS = {"je", "tu", "il", "elle", "le", "la", "les", "de", "des", "un", "une"}


    categories_depart = []
    for mot in mots:
        # catégorie venant du modèle
        categorie = analyser_mot(mot)
        categories_depart.append(categorie)

        # recherche pos_tag dans le CSV avec fuzzy
        indice = trouver_mot_proche(mot)
        if indice is not None:
            pos_tag = data.loc[indice, "pos_tag"]
        else:
            pos_tag = "inconnu"

        # -------------------------------
        # Ici : EXACTEMENT ce que tu voulais
        # -------------------------------
        if categorie == "nom":
            print("nom trouvé :", mot)
            nom = analyser_mot(mot)

        if categorie == "article":
            print("article trouvé :", mot)
            article = analyser_mot(mot)

        if categorie == "verbe":
            print("verbe trouvé :", mot)
            vrb = analyser_mot(mot)

        if categorie == "interjection":
            print("interjection trouvée :", mot)
            inter = analyser_mot(mot)

        # affichage final
        print(f" → {mot} : catégorie = {categorie} | classe grammaticale = {pos_tag}")
    categories_depart = list(set(categories_depart))
    print("Catégories de départ :", categories_depart)

    # correction utilisateur
    correction = input("\nEst-ce correct ? (o/n) : ").lower()

    if correction == "n":
        mot_cible = input("Quel mot est incorrect ? : ").lower()
        bonne_cat = input("Sa vraie catégorie ? : ").lower()
        bonne_pos = input("Sa classe grammaticale ? : ").lower()

        nouveau = pd.DataFrame({
            "texte": [mot_cible],
            "classe": [bonne_cat],
            "pos_tag": [bonne_pos]
        })

        data = pd.concat([data, nouveau], ignore_index=True)
        data.to_csv("dataset.csv", index=False)

        print("🔄 Ré-entraînement...")
        charger_modele()
        print("✔ Correction enregistrée !")

    else:
        print("OOOKKKKAI")
        gen = True

    while gen == True:
        print("generation")

        with open("noms.csv", newline="", encoding="utf-8") as f:
            mots = [row[0] for row in csv.reader(f)]
        nom = random.choice(mots)
        
        with open("pron.csv", newline="", encoding="utf-8") as f:
            mots = [row[0] for row in csv.reader(f)]
        arti = random.choice(mots)

        with open("pron.csv", newline="", encoding="utf-8") as f:
            mots = [row[0] for row in csv.reader(f)]
        articl = random.choice(mots)

        with open("verbes.csv", newline="", encoding="utf-8") as f:
            mots = [row[0] for row in csv.reader(f)]
        verbe = random.choice(mots)

        phrase = arti + " " + verbe + " " + articl + " " + nom
        print("Phrase générée :", phrase)

        decomp = decouper_phrase(phrase)

        compteur = 0

        for mot in decomp:
            categorie = analyser_mot(mot)
            print(mot, "→", categorie)

            if categorie in categories_depart:
                compteur += 1

        print("Correspondances trouvées :", compteur)

        if compteur >= 3:
            print("✅ Phrase acceptée")
            gen = False
        else:
            print("❌ Pas assez de correspondances, on regénère...")
