"""
TP 1 : Hello Pandas (Chargement et Visualisation)
Données : Pourboires au restaurant
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Importez pandas et seaborn (fait ci-dessus)

# 2. Chargez le CSV dans une variable df
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# 3. Affichez les 5 premières lignes avec df.head()
print("Les 5 premières lignes du dataset :")
print(df.head())
print("\n")

# 4. Affichez un nuage de points
sns.scatterplot(data=df, x="total_bill", y="tip")
plt.title("Relation entre le montant de l'addition et le pourboire")
plt.xlabel("Montant de l'addition ($)")
plt.ylabel("Pourboire ($)")
plt.show()

# 5. Question : Que remarquez-vous sur la relation entre le montant de l'addition et le pourboire ?
print("\nQuestion : Que remarquez-vous sur la relation entre le montant de l'addition et le pourboire ?")
print("Réponse : On observe généralement une relation positive : plus l'addition est élevée, plus le pourboire tend à être élevé.")

