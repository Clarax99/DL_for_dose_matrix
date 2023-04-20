 # Prédiction de maladies cardiaques suite à un traitement de cancer de l'enfant
 Comparaison de méthodes de Machine Learning et de Deep Learning
 Auteurs : Thomas Ménard et Clara Cousteix
 
 ## Description du projet
 
Ce projet est porté par deux acteurs principaux, le laboratoire MICS de CentraleSupélec et l'équipe INSERM de l'Institut Gustave Roussy, Villejuif. L'enjeu de ce sujet est de prédire l'apparition de maladies cardiaques suite à un traitement ant-cancéreux chez l'enfant ou l'adolescent. 

Pour réaliser ces prédictions, nous avons à disposition une base de données de **7670 patients**, issus de la cohorte FCCSS, French Childhood Cancer Survivor Study. Parmi les patients de la cohorte, nous possédons les matrices de doses issues de traitements en radiothérapie 3D de près de **4000 patients**. De plus, nous possédons des données cliniques sur les traitements par chimiothérapie. Après sélection des patients, nous avons une population d'étude de **1378 patients**.

Nous avons entraînés les algorithmes de Machine Learning **(Random Forest, XGBoost, LightGBM)** sur les indicateurs dose-volume issus des matrices de doses. Nous avons entraînés les algorithmes de Deep Learning **(réseau linéaire, réseau convolutionnel, réseau à chemins multiples)** sur les matrices de doses 3D. Concernant les algorithmes de Machine Learning, nous obtenons des résultats d'environ **67%** pour la balanced accuracy  et d'environ **50%** pour le score de rappel. Concernant les algorithmes de Deep Learning, nous obtenons des résultats d'environ **69%** pour la balanced accuracy et d'environ **60%** pour le score de rappel, jusqu'à **67%** pour le réseau à chemins multiples.

Il apparaît donc que les réseaux de neurones parviennent mieux à prédire les individus positifs que les algorithmes de Deep Learning, ce qui est primordial en prédiction de maladies. Les résultats obtenus permettront d'améliorer le suivi des patients présents dans cette base de données en augmentant ou réduisant le nombre de séance de suivie suite au cancer. Plus de de travaux sont nécessaires pour améliorer la robustesse des modèles et améliorer les performances.

## Sélection des patients pour l'étude

![image](https://user-images.githubusercontent.com/124738526/233184048-87b450e5-813b-4b24-92e7-a4df069dc79d.png)

## Organisation du code

Le prétraitement des données de survie, les analyses de survie, l'analyse en composantes principales sont disponibles dans le dossier "notebooks".

Les analyses en Machine Learning sont disponibles dans le notebook ML_dosevolume.ipynb dans le dossier "notebooks".

Les fichiers python (dataset.py, model.py, loss.py, main.py) sont destinés l'entraînement des réseau de neurones. Pour les lancer, il faut exécuter main.py, avec deux arguments. Le premier prend en valeur 40 ou 50, et correspond au temps de censure minimal choisi. Le second correpond à la L2-régularisation (weight decay dans l'optimizr Adam). Il prend des entiers. Celui-ci peut être désactivé en commentant certaines sections dans main.py.
```
python main.py 40 4    # avec de la L2 régularisation de 10**-4
python main.py 40      # sans L2 régularisation
```
Le ficher main.py est le fichier de configuration. Il regroupe les principaux hyperparamètres des modèles.

Le fichier predict.py est destiné à tester les réseaux de neurones avec le test set prévu à cet effet. Il est prévu pour tester tous les modèles d'un dossier.

```
python predict.py
```

Les fichiers shell sont destinés à l'exécution du code sur le supercalculteur de l'Université Paris Saclay, le Mésocentre

## Organisation des données

Le chemin "path_to_dir" doit être l'emplacement d'un dossier data, organisé ainsi qu'illustré dans l'image ci-dessous. Il est nécessaire d'avoir : 
* un dossier nii, contenant toutes les images
* un ficher labels.csv, dont le contenu sera détaillé plus tard
* un fichier clinical_features.csv , avec les colonnes "ctr", "numcent", "file_name" et les données cliniques
* un dossier saved models, lui-même contenant : 
   * les modèles entraînés et enregistrés, ainsi qu'un fichier texte par modèle. Par défaut, exécuter le code enregistrera deux poids : ceux qui minimisent la validation_loss et ceux qui minimisent le la validation_balanced_accuracy. Empiriquement, on remarque que les poids qui minimisent la validation loss donnent de meilleurs résultats sur le test set. Les fichiers textes enregistrent les hyperparamètres du modèle, ainsi que les epochs avec enregistrement des poids.
   * le dossier saved_plots. Une fois toutes les epochs effectuées, on enregistre des plots des loss d'entraînement, ainsi que les valeurs de balanced accuracy et de recall.
   * un dossier saved_plots_test : par modèle testé, on a une matrice de confusion sur le test set, ainsi qu'un fichier texte avec les valeurs des métriques sur le test set
* un dossier csv (optionnel) : les fichiers CSV correspondent aux doses par séance de radiothérapie par patient. Les csv de chaque patient dans un intervalle de 6 mois ont été sommés pour former l'image résultante en format NIFTI, enregistrée dans le dossier nii. Ce dossier est optionnel, nécessaire seulement si on souhaite modifier les images NIFTI.

![image](https://user-images.githubusercontent.com/124738526/233300459-8ce7f4af-8ea0-42cc-ac55-696a34a68ab5.png)

## Fichier labels.csv

| Nom de la colonne | Signification | Valeur attendue |
|-------------------|---------------|-----------------|
| ctr  | Numéro du centre de traitement   | [1, 3, 8, 10, 12]   |
| numcent   | Identifiant du patient dans son centre   | 199600389 par ex|
| file_location  | chemin absolu vers l'image nii| `D:\data\dose_matrices\nii\heart_cropped\CURIE\newdosi_3_199600389_ID2013A.nii.gz` par ex|
| file_name   | Nom de l'image nii   | newdosi_3_199600389_ID2013A.nii.gz par ex|
| Pathologie_cardiaque_3_new   | label | 0 ou 1   |
| do_ANTHRA/do_ALKYL/do_VINCA   | doses de famille de chimiothérapie reçue   | flottant positif  |
| card_age   | Temps à l'apparition d'une maladie cardiaque ou à défaut temps de censure | flottant positif |
| card_age_40/card_age_50   | 1 si [(apparition d'une maladie cardiaque) OU (si tps_censure>40, resp 50)], 0 sinon  | 0 ou 1  |
| train_40, resp train_50 | Patients d'entraînement pour card_age_40 = 1, resp card_age_50 = 1, 60% du dataset d'étude  | 0 ou 1   |
| val_40, resp val_50 | Patients de validation pour card_age_40 = 1, resp card_age_50 = 1, 20% du dataset d'étude  | 0 ou 1   |
| test_40, resp test_50 | Patients de test pour card_age_40 = 1, resp card_age_50 = 1, 20% du dataset d'étude  | 0 ou 1   |
