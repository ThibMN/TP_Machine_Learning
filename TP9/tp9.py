"""
TP 9 : Évaluation - Matrice de Confusion
Objectif : Comprendre les erreurs du TP 7
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Reprenez le modèle Iris du TP 7
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Faites des prédictions sur le Test set : y_pred = model.predict(X_test)
y_pred = model.predict(X_test)

# Importez confusion_matrix depuis sklearn.metrics (fait ci-dessus)

# Affichez la matrice (print(confusion_matrix(y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)
print("\n")
print("Légende :")
print("Les lignes représentent les vraies espèces")
print("Les colonnes représentent les espèces prédites")
print(f"\nEspèces : {model.classes_}")

# Question : Combien d'erreurs le modèle a-t-il faites ? (Regardez les chiffres hors de la diagonale)
total_predictions = cm.sum()
correct_predictions = cm.trace()  # Somme de la diagonale
errors = total_predictions - correct_predictions

print(f"\nNombre total de prédictions : {total_predictions}")
print(f"Prédictions correctes (diagonale) : {correct_predictions}")
print(f"Nombre d'erreurs (hors diagonale) : {errors}")
print(f"Taux d'erreur : {errors/total_predictions*100:.2f}%")

