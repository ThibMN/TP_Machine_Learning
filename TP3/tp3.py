"""
TP 3 : Nettoyer les trous (Data Cleaning)
Données : Titanic
"""

import pandas as pd

# 1. Chargez le dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

print("Dataset Titanic chargé")
print(f"Nombre de lignes : {len(df)}")
print(f"Nombre de colonnes : {len(df.columns)}")
print("\n")

# 2. Affichez le nombre de valeurs manquantes
print("Nombre de valeurs manquantes par colonne :")
print(df.isna().sum())
print("\n")

# 3. La colonne age a des trous. Remplacez les vides par la moyenne des âges
print("Remplacement des valeurs manquantes dans 'age' par la moyenne...")
df['age'] = df['age'].fillna(df['age'].mean())

# 4. Vérifiez avec isna().sum() que la colonne age est propre
print("\nVérification après nettoyage :")
print("Nombre de valeurs manquantes dans 'age' :", df['age'].isna().sum())
print("\nNombre total de valeurs manquantes par colonne :")
print(df.isna().sum())

