"""
TP 13 : Vision par Ordinateur (Reconnaître des chiffres)
Données : Images de chiffres manuscrits (8x8 pixels)
Mission : Entraînez un modèle pour reconnaître quel chiffre est écrit
Contraintes : Essayez d'afficher une image avec plt.imshow avant de faire le modèle. Obtenez plus de 90% de précision
"""

import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Supprimer les warnings RuntimeWarning pour une sortie plus propre
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Code chargement
digits = load_digits()
X = digits.data
y = digits.target

print("Dataset de chiffres manuscrits chargé")
print(f"Nombre d'images : {len(X)}")
print(f"Taille de chaque image : {digits.images[0].shape} pixels")
print(f"Chiffres disponibles : {digits.target_names}")
print("\n")

# Afficher une image avec plt.imshow avant de faire le modèle
print("Affichage de quelques exemples d'images :")
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Chiffre : {digits.target[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDonnées d'entraînement : {len(X_train)} images")
print(f"Données de test : {len(X_test)} images")
print("\n")

# Entraîner un modèle pour reconnaître quel chiffre est écrit
# Utilisation de LogisticRegression avec max_iter augmenté pour convergence
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calcul de la précision
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Modèle entraîné")
print(f"Précision sur les données d'entraînement : {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Précision sur les données de test : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("\n")

# Obtenez plus de 90% de précision
if test_accuracy > 0.90:
    print(f"✓ Objectif atteint ! La précision ({test_accuracy*100:.2f}%) est supérieure à 90%")
else:
    print(f"✗ La précision ({test_accuracy*100:.2f}%) est inférieure à 90%")

print("\nRapport de classification détaillé :")
print(classification_report(y_test, y_test_pred))

