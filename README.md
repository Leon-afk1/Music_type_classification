# Classification de Genres Musicaux (Projet DS52)

Ce projet nous a été donné dans le cadre de mon cursus scolaire à l'UTBM.
J'étais en groupe avec Alban MORIN et Rémi  BONNET.

Ce projet explore la classification de genres musicaux en utilisant des caractéristiques audio extraites de la base de données GTZAN. L'objectif est de comparer les performances d'un modèle classique (Support Vector Machine) et d'un réseau de neurones (construit avec Keras) pour prédire le genre d'un morceau de musique à partir de ses caractéristiques.

Ce notebook utilise également [Weights & Biases (Wandb)](https://wandb.ai/) pour le suivi des expérimentations, y compris la journalisation des métriques, des graphiques, et même des échantillons audio.

## Données

Le projet utilise le **GTZAN Dataset (Music Genre Classification)**, accessible via Kaggle.

* **Source :** [GTZAN Dataset sur Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
* **Fichiers utilisés :**
    * `Data/features_3_sec.csv` : Caractéristiques audio extraites de segments de 3 secondes. C'est le jeu de données principal utilisé pour l'entraînement des modèles.
    * `Data/features_30_sec.csv` : Caractéristiques extraites de segments de 30 secondes (chargé mais moins utilisé pour la modélisation dans ce notebook).
    * `Data/genres_original/` : Contient les fichiers audio bruts (ex: `.wav`) utilisés pour l'analyse exploratoire (visualisation de spectrogrammes, etc.).

Les données comprennent 10 genres musicaux :
`blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.

## Méthodologie

Le notebook suit les étapes suivantes :

### 1. Chargement et Exploration des Données (EDA)

* Importation des bibliothèques nécessaires (`pandas`, `librosa`, `sklearn`, `tensorflow`, `wandb`).
* Téléchargement des données via `kagglehub`.
* Analyse initiale des dataframes (vérification des valeurs nulles, description des caractéristiques, distribution des labels).
* Analyse audio exploratoire :
    * Chargement d'échantillons audio bruts (rock et classique) avec `librosa`.
    * Visualisation des formes d'onde (Waveforms).
    * Calcul et affichage des STFT (Short-Time Fourier Transform).
    * Calcul et affichage des Mel Spectrogrammes.

### 2. Prétraitement des Données

* Utilisation du jeu de données `features_3_sec.csv`.
* Encodage des labels textuels (genres) en valeurs numériques (`LabelEncoder`).
* Mise à l'échelle des caractéristiques (features) à l'aide de `StandardScaler`.
* Séparation des données en ensembles d'entraînement (70%) et de test (30%) (`train_test_split`).

### 3. Modélisation et Évaluation

Deux modèles ont été entraînés et comparés :

#### Modèle A : Support Vector Machine (SVM)

1.  **SVM Basique :** Un premier modèle `SVC` de Scikit-learn est entraîné, atteignant une précision d'environ **76.3%**.
2.  **SVM avec GridSearchCV :** Les hyperparamètres du `SVC` (spécifiquement `C` et `gamma`) sont optimisés à l'aide de `GridSearchCV`.
    * *Meilleurs paramètres trouvés :* `{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}`
    * Le modèle optimisé atteint une précision d'environ **89.9%**.

#### Modèle B : Réseau de Neurones (Keras)

1.  **Architecture :** Un réseau de neurones profond (MLP) est construit avec Keras `Sequential`.
    * Couches `Dense` (1024, 512, 256, 128, 64) avec activation `relu`.
    * Couches `Dropout` (0.3) pour la régularisation.
    * Couche de sortie `Dense` (10 neurones) avec activation `softmax` pour la classification multi-classe.
2.  **Compilation :** Optimiseur `adam` et fonction de perte `sparse_categorical_crossentropy`.
3.  **Entraînement :** Le modèle est entraîné sur 100 époques.
4.  **Évaluation :** Les performances (accuracy, loss) sont tracées pour l'entraînement et la validation. Un rapport de classification et une matrice de confusion sont générés pour évaluer les résultats sur l'ensemble de test.

## ⚙️ Dépendances

Pour exécuter ce notebook, les principales bibliothèques suivantes sont requises :

```bash
pandas
kaggle
tensorflow
wandb
scikit-learn
librosa
matplotlib
seaborn
```

Vous pouvez les installer via pip :
```bash
pip install pandas kaggle tensorflow wandb scikit-learn librosa matplotlib seaborn
```

## Résultats 

La comparaison des performances a montré que :
- Le SVM optimisé (GridSearchCV) a atteint une précision d'environ 89.9% sur l'ensemble de test.
- Le Réseau de Neurones (Keras) a atteint une précision de test d'environ 86.7% après 100 époques.
La matrice de confusion du modèle Keras montre que certaines confusions persistent, notamment entre des genres musicalement proches comme le rock et le disco . Le SVM optimisé s'est avéré légèrement plus performant sur ces données tabulaires de caractéristiques.
