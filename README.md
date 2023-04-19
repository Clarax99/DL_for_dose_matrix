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
