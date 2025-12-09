"""
TP 4 : Train / Test Split (La base de la robustesse)
Données : Titanic (version nettoyée du TP3)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement et nettoyage des données (comme TP3)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

# Nettoyage de la colonne age
df['age'] = df['age'].fillna(df['age'].mean())

# 1. Importez la fonction train_test_split (fait ci-dessus)

# 2. Préparez vos variables : X (âge) et y (survived)
X = df[['age']]
y = df['survived']

# 3. Copiez cette ligne exacte
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Affichez la taille des données d'entraînement
print("Taille des données d'entraînement :", X_train.shape)
print("Taille des données de test :", X_test.shape)
print(f"\nDonnées d'entraînement : {len(X_train)} lignes")
print(f"Données de test : {len(X_test)} lignes")

# 5. Question : Pourquoi coupe-t-on les données en deux ?
print("\nQuestion : Pourquoi coupe-t-on les données en deux ?")
print("Réponse : On sépare les données pour évaluer la performance du modèle sur des données")
print("qu'il n'a jamais vues pendant l'entraînement. Cela permet de détecter le surapprentissage")
print("et de s'assurer que le modèle généralise bien à de nouvelles données.")

