import pandas as pd
import sys

# ==========================================
# 1. Charger le dataset
# ==========================================

df = pd.read_csv("dataset.csv", encoding='utf-8', on_bad_lines='skip', keep_default_na=False)

print("\n--- Nettoyage des lignes avec '?' ---")
rows_before = len(df)
question_mask = df.apply(lambda col: col.astype(str).str.strip().eq("?")).any(axis=1)
if question_mask.any():
    print("Lignes contenant '?' :")
    print(df.loc[question_mask].index.tolist())
    df = df.loc[~question_mask].copy()

# Nettoyage supplémentaire des valeurs vides au cas où certaines colonnes auraient été lues comme NaN
for col in df.columns:
    df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
df = df.dropna().copy()
rows_after = len(df)
print(f"Lignes avant nettoyage : {rows_before}")
print(f"Lignes supprimées : {rows_before - rows_after}")
print(f"Lignes après nettoyage : {rows_after}")

# Nettoyage des valeurs anormales (ex. #Params (B) négatif)
print("\n--- Nettoyage des valeurs aberrantes ---")
negative_params_mask = df["#Params (B)"] < 0
if negative_params_mask.any():
    abnormal_rows = df.loc[negative_params_mask].index.tolist()
    print(f"Lignes avec #Params (B) négatif : {abnormal_rows}")
    df = df.loc[~negative_params_mask].copy()
else:
    print("Aucune valeur anormale détectée sur #Params (B).")

# Sauvegarder le dataset nettoyé sur le fichier CSV original
output_path = "dataset.csv"
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\nDataset nettoyé enregistré dans : {output_path}")

print("Taille du dataset :", df.shape)
print("\nColonnes (raw) :")
try:
    for i, col in enumerate(df.columns):
        col_repr = repr(col)
        try:
            sys.stdout.write(f"{i}: {col_repr}\n")
        except UnicodeEncodeError:
            sys.stdout.write(f"{i}: {col_repr.encode('utf-8').decode('unicode-escape')}\n")
except Exception as e:
    print(f"Erreur : {e}")
    print("Colonnes (raw) :", df.columns.tolist())


# ==========================================
# 2. Vérifier les valeurs manquantes
# ==========================================

print("\n--- Valeurs manquantes ---")
missing = df.isnull().sum()
print(missing)
if missing.sum() > 0:
    print("\n--- Cellules manquantes (ligne, colonne) ---")
    for row_index, row in df.iterrows():
        missing_cols = [col for col in df.columns if pd.isna(row[col])]
        if missing_cols:
            print(f"Ligne {row_index}: colonnes manquantes -> {missing_cols}")


# ==========================================
# 3. Afficher les informations du dataset
# ==========================================

print("\n--- Informations ---")
df.info()


# ==========================================
# 4. Afficher les valeurs possibles
#    des colonnes catégorielles
# ==========================================

colonnes_categorielles = [
    "Precision",
    "Type",
    "Weight type",
    "Architecture",
    "MoE"
]

for colonne in colonnes_categorielles:
    print(f"\n--- {colonne} ---")
    print("Nombre de valeurs différentes :", df[colonne].nunique())
    print(df[colonne].value_counts(dropna=False).head(20))


# ==========================================
# 5. Vérifier les colonnes numériques
# ==========================================

colonnes_numeriques = [
    "#Params (B)",
    "CO₂ cost (kg)",
    "Generation",
    "Average ⬆️"
]

print("\n--- Statistiques numériques ---")
print(df[colonnes_numeriques].describe())


# ==========================================
# 6. Séparer les entrées et la cible
# ==========================================

colonnes_entrees = [
    "Precision",
    "Type",
    "Weight type",
    "Architecture",
    "#Params (B)",
    "MoE",
    "CO₂ cost (kg)",
    "Generation"
]

X = df[colonnes_entrees]
y = df["Average ⬆️"]

print("\n--- X ---")
print(X.head())

print("\n--- y ---")
print(y.head())


# ==========================================
# 7. Vérifier les types de données
# ==========================================

print("\n--- Types de X ---")
print(X.dtypes)

print("\n--- Type de y ---")
print(y.dtype)