"""
TP 14 : Clustering de Géographie
Données : Housing (TP 12)
Mission : Utilisez uniquement la Latitude et la Longitude
Action : Demandez à un algorithme K-Means de trouver 5 "zones" géographiques distinctes en Californie
Visualisation : Faites un scatterplot de la latitude/longitude en coloriant les points selon leur cluster
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans

# Supprimer les warnings RuntimeWarning pour une sortie plus propre
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Chargement des données
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print("Dataset chargé")
print(f"Nombre de lignes : {len(df)}")
print("\n")

# Utilisez uniquement la Latitude et la Longitude
X = df[['latitude', 'longitude']]

print("Features utilisées : latitude et longitude")
print("\n")

# Demandez à un algorithme K-Means de trouver 5 "zones" géographiques distinctes
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Ajouter les clusters au dataframe
df['cluster'] = kmeans.labels_

print("Clustering K-Means effectué avec 5 clusters")
print(f"Centres des clusters :")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"  Cluster {i}: latitude={center[0]:.2f}, longitude={center[1]:.2f}")
print("\n")

# Visualisation : Faites un scatterplot de la latitude/longitude en coloriant les points selon leur cluster
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], 
                     cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter, label='Cluster')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], 
           marker='x', s=200, c='red', linewidths=3, label='Centres des clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustering géographique de la Californie (5 zones)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Visualisation créée : cela devrait dessiner une carte approximative de la Californie")
print("avec 5 zones géographiques distinctes coloriées différemment.")

