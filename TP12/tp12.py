"""
TP 12 : Estimation Immobilière (Régression)
Mission : Prédisez la "median_house_value"
Contraintes :
1. Vous devez gérer les valeurs manquantes (s'il y en a)
2. Vous devez utiliser au moins 3 colonnes en entrée
3. Votre modèle doit avoir un score (R²) sur le jeu de test supérieur à 0.5
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Supprimer les warnings RuntimeWarning pour une sortie plus propre
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Chargement des données
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print("Dataset chargé")
print(f"Nombre de lignes : {len(df)}")
print(f"Nombre de colonnes : {len(df.columns)}")
print("\n")

# 1. Gérer les valeurs manquantes
print("Valeurs manquantes par colonne :")
print(df.isna().sum())
print("\n")

# Remplacer les valeurs manquantes par la médiane (numérique uniquement)
if df.isna().sum().sum() > 0:
    print("Remplacement des valeurs manquantes par la médiane (colonnes numériques)...")
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print("Nettoyage terminé")
    print("\n")

# 2. Utiliser au moins 3 colonnes en entrée
# Exemple : latitude, longitude, median_income
# On peut aussi ajouter d'autres colonnes numériques
X = df[['latitude', 'longitude', 'median_income', 'housing_median_age', 'total_rooms']].astype(float)
y = df['median_house_value'].astype(float)

# Nettoyer les valeurs infinies et NaN qui pourraient causer des problèmes
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
y = y.replace([np.inf, -np.inf], np.nan)
y = y.fillna(y.median())

print("Features sélectionnées :")
print(X.columns.tolist())
print("\n")

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entraîner le modèle avec standardisation et obtenir un score R² > 0.5
# StandardScaler évite les warnings d'overflow en mettant les features à la même échelle
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

# Prédictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Scores R²
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Modèle de régression linéaire entraîné")
print(f"R² sur les données d'entraînement : {train_r2:.4f}")
print(f"R² sur les données de test : {test_r2:.4f}")
print("\n")

if test_r2 > 0.5:
    print(f"✓ Objectif atteint ! Le score R² ({test_r2:.4f}) est supérieur à 0.5")
else:
    print(f"✗ Le score R² ({test_r2:.4f}) est inférieur à 0.5. Essayez d'ajouter plus de features.")

print("\nCoefficients du modèle :")
# Extraire les coefficients du modèle dans l'échelle d'origine
scaler = model.named_steps["standardscaler"]
lin_reg = model.named_steps["linearregression"]

# Reprojetter les coefficients dans l'échelle des features d'origine
coef_original = lin_reg.coef_ / scaler.scale_
intercept_original = lin_reg.intercept_ - np.sum(scaler.mean_ * coef_original)

for feature, coef in zip(X.columns, coef_original):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {intercept_original:.4f}")

