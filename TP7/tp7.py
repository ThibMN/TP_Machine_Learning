"""
TP 7 : Classification Simple (Les fleurs Iris)
Objectif : Prédire l'espèce d'une fleur
Données : Iris
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Chargement des données
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# X = toutes les colonnes numériques (sepal_length, etc.). y = species
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# N'oubliez pas le train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utilisez une LogisticRegression (Attention : augmentez le paramètre max_iter=1000 si ça plante)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Affichez le score de précision
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Modèle de classification Iris entraîné")
print(f"Score de précision sur les données d'entraînement : {train_score:.4f} ({train_score*100:.2f}%)")
print(f"Score de précision sur les données de test : {test_score:.4f} ({test_score*100:.2f}%)")

