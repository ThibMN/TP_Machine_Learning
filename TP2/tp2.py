"""
TP 2 : Ma première Régression (Le "Hello World")
Données : Mêmes données que le TP1
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

# Chargement des données
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# 1. Définissez X = df[['total_bill']] (Notez les doubles crochets)
X = df[['total_bill']]

# 2. Définissez y = df['tip']
y = df['tip']

# 3. Importez LinearRegression depuis sklearn.linear_model (fait ci-dessus)

# 4. Créez le modèle et entraînez-le
model = LinearRegression()
model.fit(X, y)

print(f"Coefficient (pente) : {model.coef_[0]:.4f}")
print(f"Intercept (ordonnée à l'origine) : {model.intercept_:.4f}")
print("\n")

# 5. Faites une prédiction : Quel pourboire pour une addition de 50$ ?
prediction = model.predict(pd.DataFrame({'total_bill': [50]}))
print(f"Pour une addition de 50$, le pourboire prédit est : ${prediction[0]:.2f}")

