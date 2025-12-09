### **NIVEAU 1 : TRÈS GUIDÉS** 

#### **TP 1 : Hello Pandas (Chargement et Visualisation)**

* **Données :** Pourboires au restaurant.  
* **Lien :** https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv  
* **Consigne :**  
  1. Importez pandas et seaborn.  
  2. Chargez le CSV dans une variable df.  
  3. Affichez les 5 premières lignes avec df.head().  
  4. Copiez ce code pour afficher un nuage de points :   
     sns.scatterplot(data=df, x="total\_bill", y="tip").  
  5. **Question :** Que remarquez-vous sur la relation entre le montant de l'addition et le pourboire ?


#### **TP 2 : Ma première Régression (Le "Hello World")**

* **Données :** Mêmes données que le TP1.  
* **Consigne :**  
  1. Définissez X \= df\[\['total\_bill'\]\]         (Notez les doubles crochets).  
  2. Définissez y \= df\['tip'\].  
  3. Importez   LinearRegression    depuis     sklearn.linear\_model.  
  4. Créez le modèle (model \= LinearRegression()) et entraînez-le (fit).  
  5. Faites une prédiction : Quel pourboire pour une addition de 50$ ? (model.predict(\[\[50\]\])).


#### **TP 3 : Nettoyer les trous (Data Cleaning)**

* **Données :** Titanic.  
* **Lien :** https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv  
* **Consigne :**  
  1. Chargez le dataset.  
  2. Affichez le nombre de valeurs manquantes avec df.isna().sum().

3.La colonne age a des trous. Remplacez les vides par la moyenne des âges :  
df\['age'\] \= df\['age'\].fillna(df\['age'\].mean()) 

3. Vérifiez avec isna().sum() que la colonne age est propre.

   

#### **TP 4 : Train / Test Split (La base de la robustesse)**

* **Données :** Titanic (version nettoyée du TP3).  
* **Consigne :**  
  1. Importez la fonction : from sklearn.model\_selection import train\_test\_split.  
  2. Préparez vos variables : X (âge) et y (survived).

3.Copiez cette ligne exacte :  
    X\_train, X\_test, y\_train, y\_test \= train\_test\_split(X, y, test\_size=0.2, random\_state=42) 

3. Affichez la taille des données d'entraînement avec print(X\_train.shape).  
   4. **Question :** Pourquoi coupe-t-on les données en deux ?

#### **TP 5 : Encoder du texte (Pour que l'IA comprenne)**

* **Données :** Titanic.  
* **Problème :** La colonne sex contient "male" et "female". L'IA ne veut que des chiffres.  
* **Consigne :**  
  1. Affichez la colonne sex.

2.Utilisez la méthode get\_dummies de Pandas :

    df \= pd.get\_dummies(df, columns=\['sex'\], drop\_first=True) 

2. Affichez df.head(). Que s'est-il passé avec la colonne sex ? (Elle est devenue sex\_male avec des 0 et des 1).

###  **NIVEAU 2 : MOYENNEMENT GUIDÉS**

#### **TP 6 : Régression Multiple**

* **Objectif :** Prédire le pourboire en fonction de l'addition ET de la taille de la table.  
* **Données :** Tips (https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv).  
* **Indices :**  
  * Pour X, sélectionnez deux colonnes : \['total\_bill', 'size'\].  
  * Entraînez une LinearRegression.  
  * Prédisez le pourboire pour une addition de 100$ et une table de 6 personnes.

#### **TP 7 : Classification Simple (Les fleurs Iris)**

* **Objectif :** Prédire l'espèce d'une fleur.  
* **Données :** Iris.  
* **Lien :** https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv  
* **Indices :**  
  * X \= toutes les colonnes numériques (sepal\_length, etc.). y \= species.  
  * N'oubliez pas le train\_test\_split.  
  * Utilisez une LogisticRegression (Attention : augmentez le paramètre max\_iter=1000 si ça plante).  
  * Affichez le score de précision (model.score(...)).


#### **TP 8 : Clustering (Grouper des clients)** 

* **Contexte :** On ne veut pas prédire, on veut faire des groupes automatiques.  
* **Données :** Iris (sans la colonne species).  
* **Indices :**  
  * Importez KMeans depuis sklearn.cluster.  
  * Initialisez model \= KMeans(n\_clusters=3).  
  * Faites juste un model.fit(X).  
  * Ajoutez une colonne cluster au dataframe avec model.labels\_.  
  * Affichez le tableau. L'IA a-t-elle retrouvé les groupes ?

#### **TP 9 : Évaluation \- Matrice de Confusion**

* **Objectif :** Comprendre les erreurs du TP 7\.  
* **Indices :**  
  * Reprenez le modèle Iris du TP 7\.  
  * Faites des prédictions sur le Test set : y\_pred \= model.predict(X\_test).  
  * Importez confusion\_matrix depuis sklearn.metrics.  
  * Affichez la matrice (print(confusion\_matrix(y\_test, y\_pred))).  
  * **Question :** Combien d'erreurs le modèle a-t-il faites ? (Regardez les chiffres hors de la diagonale).

#### **TP 10 : Sauvegarder son travail**

* **Objectif :** Exporter le modèle Iris.  
* **Indices :**  
  * Importez la librairie joblib.  
  * Utilisez joblib.dump(votre\_modele, 'iris\_model.pkl').  
  * Vérifiez dans l'onglet "Fichiers" de gauche de Google Colab que le fichier est bien là.  
  * Essayez de le recharger dans une nouvelle variable avec joblib.load().

### **NIVEAU 3 : NON GUIDÉS** 

### **TP 11 : L'Archipel des Pingouins (Standardisation & Classification)**

* ### **Lien des données :** [https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv)

* ### **Mission :** Prédire l'espèce du pingouin (Adelie, Chinstrap ou Gentoo) en fonction de ses mesures corporelles. 

1. #### **Chargement & Nettoyage :**

   * #### Chargez le CSV.

   * #### Attention : Ce dataset contient des lignes vides

2. #### **Sélection des Features :**

   * #### Nous n'utilisons que les colonnes numériques pour prédire l'espèce.

   * #### X \= bill\_length\_mm, bill\_depth\_mm, flipper\_length\_mm, body\_mass\_g.

   * #### y \= species.

3. #### **Le Piège (Scaling) :**

   * #### Regardez vos données : le poids est en grammes (ex: 4000\) et le bec en mm (ex: 40). L'écart est trop grand.

   * #### Vous devez utiliser StandardScaler sur X (après le split Train/Test) pour tout mettre à la même échelle.

4. #### **Modélisation :**

   * #### Entraînez une LogisticRegression (avec max\_iter=1000 pour être sûr qu'elle converge).

   #### Affichez votre score de précision.

#### **TP 12 : Estimation Immobilière (Régression)**

* **Lien :** https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv  
* **Mission :** Prédisez la "median\_house\_value".  
* **Contraintes :**  
  1. Vous devez gérer les valeurs manquantes (s'il y en a).  
  2. Vous devez utiliser au moins 3 colonnes en entrée (ex: latitude, longitude, median\_income).  
  3. Votre modèle doit avoir un score (R²) sur le jeu de test supérieur à 0.5

#### 

#### **TP 13 : Vision par Ordinateur (Reconnaître des chiffres)**

* **Données :** Images de chiffres manuscrits (8x8 pixels).

**Code chargement :**

    from sklearn.datasets import load\_digits  
digits \= load\_digits()  
X \= digits.data  
y \= digits.target

* **Mission :** Entraînez un modèle pour reconnaître quel chiffre est écrit.  
* **Contraintes :** Essayez d'afficher une image avec plt.imshow(digits.images\[0\]) avant de faire le modèle. Obtenez plus de 90% de précision.

#### **TP 14 : Clustering de Géographie**

* **Données :** Housing (TP 11).  
* **Mission :** Utilisez uniquement la Latitude et la Longitude.  
* **Action :** Demandez à un algorithme K-Means de trouver 5 "zones" géographiques distinctes en Californie.  
* **Visualisation :** Faites un scatterplot de la latitude/longitude en coloriant les points selon leur cluster. Cela devrait dessiner une carte approximative de la Californie.

#### **TP 15 : Le Challenge Final \- Titanic Complet**

* **Lien :** https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv  
* **Mission :** Prédire qui survit.  
* **Difficulté :** Vous devez tout combiner.  
  1. **Nettoyer** : Gérer les âges vides.  
  2. **Encoder** : Transformer le sexe ET la classe d'embarquement (string \-\> int).  
  3. **Sélectionner** : Choisir les bonnes colonnes (pas le nom du passager \!).  
  4. **Entraîner** : Régression Logistique ou Arbre de décision (DecisionTreeClassifier).  
  5. **Score** : Tentez de battre 78% de précision.