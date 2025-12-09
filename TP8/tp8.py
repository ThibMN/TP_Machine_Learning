"""
TP 8 : Clustering (Grouper des clients)
Contexte : On ne veut pas prédire, on veut faire des groupes automatiques
Données : Iris (sans la colonne species)
"""

import pandas as pd
from sklearn.cluster import KMeans

# Chargement des données
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Données : Iris (sans la colonne species)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Importez KMeans depuis sklearn.cluster (fait ci-dessus)

# Initialisez model = KMeans(n_clusters=3)
model = KMeans(n_clusters=3, random_state=42)

# Faites juste un model.fit(X)
model.fit(X)

# Ajoutez une colonne cluster au dataframe avec model.labels_
df['cluster'] = model.labels_

# Affichez le tableau. L'IA a-t-elle retrouvé les groupes ?
print("Résultats du clustering K-Means (3 clusters) :")
print("\nPremières lignes avec les clusters assignés :")
print(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species', 'cluster']].head(20))
print("\n")
print("Comparaison entre les vraies espèces et les clusters :")
comparison = pd.crosstab(df['species'], df['cluster'])
print(comparison)
print("\nL'IA a-t-elle retrouvé les groupes ?")
print("Réponse : L'IA a retrouvé les groupes approximativement, il reste une marge d'erreur à prendre en compte.")

