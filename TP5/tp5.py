"""
TP 5 : Encoder du texte (Pour que l'IA comprenne)
Données : Titanic
Problème : La colonne sex contient "male" et "female". L'IA ne veut que des chiffres.
"""

import pandas as pd

# Chargement des données
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

# 1. Affichez la colonne sex
print("Colonne 'sex' avant encodage :")
print(df['sex'].head(10))
print(f"\nValeurs uniques : {df['sex'].unique()}")
print("\n")

# 2. Utilisez la méthode get_dummies de Pandas
df = pd.get_dummies(df, columns=['sex'], drop_first=True)

# 3. Affichez df.head(). Que s'est-il passé avec la colonne sex ?
print("Dataset après encodage (5 premières lignes) :")
print(df.head())
print("\n")
print("Explication : La colonne 'sex' a été remplacée par 'sex_male' qui contient :")
print("- 1 si le passager est un homme (male)")
print("- 0 si le passager est une femme (female)")
print("\nCela permet à l'IA de travailler avec des nombres au lieu de texte.")

