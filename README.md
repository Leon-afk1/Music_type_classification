# Music Genre Classification (DS52 Project)

This project was given as part of my curriculum at UTBM. I was in a group with Alban MORIN and Rémi BONNET.

This project explores the classification of music genres using audio features extracted from the GTZAN database. The objective is to compare the performance of a classic model (Support Vector Machine) and a neural network (built with Keras) to predict the genre of a music track from its features.

This notebook also uses [Weights & Biases (Wandb)](https://wandb.ai/) for experiment tracking, including logging metrics, graphs, and even audio samples.

## 🎵 Data

The project uses the **GTZAN Dataset (Music Genre Classification)**, accessible via Kaggle.

* **Source:** [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
* **Files used:**
    * `Data/features_3_sec.csv`: Audio features extracted from 3-second segments. This is the main dataset used for training the models.
    * `Data/features_30_sec.csv`: Features extracted from 30-second segments (loaded but less used for modeling in this notebook).
    * `Data/genres_original/`: Contains the raw audio files (e.g., `.wav`) used for exploratory analysis (spectrogram visualization, etc.).

The data includes 10 music genres:
`blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.

## 🛠️ Methodology

The notebook follows these steps:

### 1. Data Loading and Exploration (EDA)

* Importing necessary libraries (`pandas`, `librosa`, `sklearn`, `tensorflow`, `wandb`).
* Downloading data via `kagglehub`.
* Initial analysis of dataframes (checking for null values, feature description, label distribution).
* Exploratory audio analysis:
    * Loading raw audio samples (rock and classical) with `librosa`.
    * Visualizing waveforms.
    * Calculating and displaying STFT (Short-Time Fourier Transform).
    * Calculating and displaying Mel Spectrograms.

### 2. Data Preprocessing

* Using the `features_3_sec.csv` dataset.
* Encoding textual labels (genres) into numerical values (`LabelEncoder`).
* Scaling the features using `StandardScaler`.
* Splitting the data into training (70%) and test (30%) sets (`train_test_split`).

### 3. Modeling and Evaluation

Two models were trained and compared:

#### Model A: Support Vector Machine (SVM)

1.  **Basic SVM:** A first Scikit-learn `SVC` model is trained, achieving an accuracy of about **76.3%**.
2.  **SVM with GridSearchCV:** The `SVC` hyperparameters (specifically `C` and `gamma`) are optimized using `GridSearchCV`.
    * *Best parameters found:* `{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}`
    * The optimized model achieves an accuracy of about **89.9%**.

#### Model B: Neural Network (Keras)

1.  **Architecture:** A deep neural network (MLP) is built with Keras `Sequential`.
    * `Dense` layers (1024, 512, 256, 128, 64) with `relu` activation.
    * `Dropout` layers (0.3) for regularization.
    * Output `Dense` layer (10 neurons) with `softmax` activation for multi-class classification.
2.  **Compilation:** `adam` optimizer and `sparse_categorical_crossentropy` loss function.
3.  **Training:** The model is trained for 100 epochs.
4.  **Evaluation:** Performance (accuracy, loss) is plotted for training and validation. A classification report and confusion matrix are generated to evaluate the results on the test set.

## ⚙️ Dependencies

To run this notebook, the following main libraries are required:

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

You can install them via pip:
```bash
pip install pandas kaggle tensorflow wandb scikit-learn librosa matplotlib seaborn
```

## Résults

The performance comparison showed that:
- The optimized SVM (GridSearchCV) achieved an accuracy of approximately 89.9% on the test set.
- The Neural Network (Keras) achieved a test accuracy of approximately 86.7% after 100 epochs.

The confusion matrix for the Keras model shows that some confusion persists, particularly between musically similar genres like rock and disco. The optimized SVM proved to be slightly more performant on this tabular feature data.
