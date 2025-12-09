"""
TP 11 : L'Archipel des Pingouins (Standardisation & Classification)
Mission : Prédire l'espèce du pingouin (Adelie, Chinstrap ou Gentoo) en fonction de ses mesures corporelles
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Chargement & Nettoyage
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)

print("Dataset chargé")
print(f"Nombre de lignes avant nettoyage : {len(df)}")

# Attention : Ce dataset contient des lignes vides
df = df.dropna()  # Supprime les lignes avec des valeurs manquantes

print(f"Nombre de lignes après nettoyage : {len(df)}")
print("\n")

# 2. Sélection des Features
# X = bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
# y = species
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

print("Features sélectionnées :")
print(X.head())
print(f"\nEspèces à prédire : {y.unique()}")
print("\n")

# 3. Le Piège (Scaling) - Split d'abord, puis scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regardez vos données : le poids est en grammes (ex: 4000) et le bec en mm (ex: 40)
print("Exemple de données brutes (première ligne) :")
print(X_train.iloc[0])
print("\nLes échelles sont très différentes !")
print("body_mass_g est en milliers, tandis que les autres mesures sont en dizaines.")
print("\n")

# Vous devez utiliser StandardScaler sur X (après le split Train/Test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir en DataFrame pour meilleure lisibilité
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Données après standardisation (première ligne) :")
print(X_train_scaled.iloc[0])
print("\nToutes les features sont maintenant à la même échelle (moyenne ≈ 0, écart-type ≈ 1)")
print("\n")

# 4. Modélisation
# Entraînez une LogisticRegression (avec max_iter=1000 pour être sûr qu'elle converge)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Affichez votre score de précision
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print("Modèle entraîné avec succès")
print(f"Score de précision sur les données d'entraînement : {train_score:.4f} ({train_score*100:.2f}%)")
print(f"Score de précision sur les données de test : {test_score:.4f} ({test_score*100:.2f}%)")

