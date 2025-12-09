"""
TP 15 : Le Challenge Final - Titanic Complet
Mission : Prédire qui survit
Difficulté : Vous devez tout combiner
1. Nettoyer : Gérer les âges vides
2. Encoder : Transformer le sexe ET la classe d'embarquement (string -> int)
3. Sélectionner : Choisir les bonnes colonnes (pas le nom du passager !)
4. Entraîner : Régression Logistique ou Arbre de décision
5. Score : Tentez de battre 78% de précision
"""

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Supprimer les warnings RuntimeWarning pour une sortie plus propre
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Chargement des données
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

print("Dataset Titanic chargé")
print(f"Nombre de lignes : {len(df)}")
print("\n")

# 1. Nettoyer : Gérer les âges vides
print("1. Nettoyage des données...")
print(f"Valeurs manquantes avant nettoyage :")
print(df.isna().sum())
print("\n")

# Remplacer les âges vides par la moyenne
df['age'] = df['age'].fillna(df['age'].mean())

# Gérer les autres valeurs manquantes si nécessaire
# Pour embarked, on peut remplacer par le mode (valeur la plus fréquente)
if df['embarked'].isna().sum() > 0:
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

print("Nettoyage terminé")
print(f"Valeurs manquantes après nettoyage :")
print(df.isna().sum())
print("\n")

# 2. Encoder : Transformer le sexe ET la classe d'embarquement (string -> int)
print("2. Encodage des variables catégorielles...")
print("Avant encodage :")
print(f"  sex: {df['sex'].unique()}")
print(f"  embarked: {df['embarked'].unique()}")
print("\n")

# Utiliser get_dummies pour encoder sex et embarked
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

print("Après encodage :")
print("Colonnes créées :")
encoded_cols = [col for col in df.columns if col.startswith('sex_') or col.startswith('embarked_')]
print(encoded_cols)
print("\n")

# 3. Sélectionner : Choisir les bonnes colonnes (pas le nom du passager !)
print("3. Sélection des features...")
# Colonnes à exclure : name, ticket (trop spécifique), cabin (trop de valeurs manquantes)
# Colonnes à inclure : age, pclass, fare, sex_male, embarked_Q, embarked_S, sibsp, parch
features = ['age', 'pclass', 'fare', 'sibsp', 'parch']
# Ajouter les colonnes encodées
features.extend([col for col in df.columns if col.startswith('sex_') or col.startswith('embarked_')])

# S'assurer que toutes les colonnes existent
features = [f for f in features if f in df.columns]

X = df[features]
y = df['survived']

print(f"Features sélectionnées : {features}")
print(f"Nombre de features : {len(features)}")
print("\n")

# Gérer les valeurs manquantes dans fare si nécessaire
if X['fare'].isna().sum() > 0:
    X['fare'] = X['fare'].fillna(X['fare'].median())

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Données d'entraînement : {len(X_train)} lignes")
print(f"Données de test : {len(X_test)} lignes")
print("\n")

# 4. Entraîner : Régression Logistique ou Arbre de décision
print("4. Entraînement des modèles...")
print("\n")

# Modèle 1 : Régression Logistique
print("Modèle 1 : Régression Logistique")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Précision : {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print("\n")

# Modèle 2 : Arbre de décision
print("Modèle 2 : Arbre de décision")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Précision : {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
print("\n")

# 5. Score : Tentez de battre 78% de précision
print("5. Évaluation finale :")
print("\n")
target_accuracy = 0.78

if lr_accuracy > target_accuracy:
    print(f"✓ Régression Logistique : Objectif atteint ! ({lr_accuracy*100:.2f}% > 78%)")
else:
    print(f"✗ Régression Logistique : {lr_accuracy*100:.2f}% < 78%")

if dt_accuracy > target_accuracy:
    print(f"✓ Arbre de décision : Objectif atteint ! ({dt_accuracy*100:.2f}% > 78%)")
else:
    print(f"✗ Arbre de décision : {dt_accuracy*100:.2f}% < 78%")

print("\n" + "="*50)
print("RAPPORT DÉTAILLÉ DU MEILLEUR MODÈLE")
print("="*50)

# Afficher le rapport du meilleur modèle
if dt_accuracy >= lr_accuracy:
    best_model = dt_model
    best_pred = dt_pred
    best_name = "Arbre de décision"
else:
    best_model = lr_model
    best_pred = lr_pred
    best_name = "Régression Logistique"

print(f"\nMeilleur modèle : {best_name}")
print(f"Précision : {accuracy_score(y_test, best_pred):.4f} ({accuracy_score(y_test, best_pred)*100:.2f}%)")
print("\nRapport de classification :")
print(classification_report(y_test, best_pred))

