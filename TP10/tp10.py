"""
TP 10 : Sauvegarder son travail
Objectif : Exporter le modèle Iris
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Création et entraînement du modèle Iris (comme TP 7)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Modèle entraîné avec succès")
print(f"Score : {model.score(X_test, y_test):.4f}")
print("\n")

# Importez la librairie joblib (fait ci-dessus)

# Utilisez joblib.dump(votre_modele, 'iris_model.pkl')
model_filename = 'iris_model.pkl'
joblib.dump(model, model_filename)
print(f"Modèle sauvegardé dans '{model_filename}'")

# Vérifiez que le fichier existe
if os.path.exists(model_filename):
    file_size = os.path.getsize(model_filename)
    print(f"Fichier créé avec succès (taille : {file_size} octets)")
else:
    print("Erreur : le fichier n'a pas été créé")
print("\n")

# Essayez de le recharger dans une nouvelle variable avec joblib.load()
loaded_model = joblib.load(model_filename)
print("Modèle rechargé avec succès")

# Vérification : testez le modèle rechargé
test_score_loaded = loaded_model.score(X_test, y_test)
print(f"Score du modèle rechargé : {test_score_loaded:.4f}")
print("Le modèle rechargé fonctionne correctement !")

