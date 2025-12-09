"""
TP 6 : Régression Multiple
Objectif : Prédire le pourboire en fonction de l'addition ET de la taille de la table
Données : Tips
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

# Chargement des données
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# Pour X, sélectionnez deux colonnes : ['total_bill', 'size']
X = df[['total_bill', 'size']]
y = df['tip']

# Entraînez une LinearRegression
model = LinearRegression()
model.fit(X, y)

print("Modèle de régression multiple entraîné")
print(f"Coefficients : total_bill = {model.coef_[0]:.4f}, size = {model.coef_[1]:.4f}")
print(f"Intercept : {model.intercept_:.4f}")
print("\n")

# Prédisez le pourboire pour une addition de 100$ et une table de 6 personnes
input_df = pd.DataFrame({'total_bill': [100], 'size': [6]})
prediction = model.predict(input_df)
print(f"Pour une addition de 100$ et une table de 6 personnes, le pourboire prédit est : ${prediction[0]:.2f}")

