# Introduction
Bienvenue au concours de programmation CEC 2025 ! Tout ce dont vous avez besoin se trouve dans ce GitHub, et nous vous recommandons de suivre les étapes suivantes pour commencer :

1. Veuillez lire le reste de ce README.md
2. Veuillez lire le tutoriel lié à `githubHowToUseGithubDesktop.txt`
3. Veuillez lire le tutoriel lié à `githubHowToFork.txt`
4. Veuillez créer un repo (Fork)
5. Veuillez cloner votre repo forké localement sur votre machine, ou utilisez un autre répertoire.
6. Ajoutez-nous comme collaborateurs.
7. Veuillez lire commencer à coder !

# Obtenir de l'aide
Veuillez suivre les étapes ci-dessous si vous avez besoin d'aide :

1. Veuillez consulter la documentation incluse pour les questions de logistique et de notation
2. Veuillez consulter le dossier du concours de programmation pour les questions relatives au concours
3. Veuillez consulter le Discord aux questions qui ont été déjà répondu
4. Si vous avez fait tout cela, veuillez demander de l'aide aux directeurs. Si nous pouvons répondre à votre question, elle sera postée sur le Discord en anglais.

# Notes importantes
Dans votre README.md, veuillez spécifier :

- Comment exécuter votre code
- Le langage et la version que votre code utilise (par exemple, Python 3.11)
- Une liste des paquets requis (par exemple, Pandas, NumPy), avec les versions si nécessaire (par exemple, pytorch==2.1.116). Un `requirements.txt` peut également suffire.
- Les fichiers modèles (avec les liens s'ils ne sont pas situés sur la branche git) qui doivent être téléchargés pour exécuter votre code. Voir ``Testing Information\specific_model_file_download.md`` pour plus d'informations à ce sujet.
- Si nécessaire, le système d'exploitation sur lequel votre code doit être exécuté. Toute spécification de ce type qui n'est pas incluse dans votre README ne peut être considérée comme étant sur les machines des directeurs.

### Instructions pour le téléchargement des fichiers de modèle :

Si votre code nécessite des fichiers spécifiques pour les données du modèle (par exemple, les poids du modèle stockés dans HDF5, Pickle, le format JSON, etc.), veuillez spécifier ces dépendances dans le fichier README. Le téléchargement de fichiers volumineux directement sur GitHub peut entraîner des retards. Si c'est le cas, vous pouvez utiliser un lien public OneDrive, Google Drive ou un autre service pour y accéder. Les équipes seront sanctionnées si les fichiers ne sont pas téléchargés à temps (les dates de modification des fichiers seront vérifiées pour OneDrive et Google Drive). 

Les équipes se verront infliger des pénalités allant de -5pts à -20pts en cas de soumission tardive des fichiers de modèles. La raison de la soumission tardive doit être indiquée et confirmée par les directeurs. Dans les cas les plus graves, les directeurs de compétition n'utiliseront pas ces données (les dates de modification des fichiers seront vérifiées sur OneDrive et Google Drive). Si votre fichier est en retard de plus d'une heure (en temps de téléchargement), les directeurs choisiront de ne pas utiliser les données du modèle, ce qui n'entraînera pas de pénalité. Cette règle ne s'applique pas aux codes et aux présentations. Les directeurs détermineront ce qui est considéré comme un « fichier modèle » après avoir examiné les soumissions des concurrents.

D'autres méthodes de téléchargement sont autorisées, à condition que les fichiers soient téléchargés dans les délais impartis par le défi et que le temps de téléchargement soit visible.



# Fichiers d'information
Les fichiers suivants fournissent toutes les informations relatives au concours. Il s'agit des dossiers suivants:
- Le document de référence
- La présentation
- L'exemple de production
- Informations sur les tests
- Info GitHub

# Accès à l'ensemble des données
Nous vous avons donné accès aux images via le lien OneDrive ici :

https://dalu-my.sharepoint.com/:u:/g/personal/or942416_dal_ca/EUw8w3GiFVlIkPkwHZGgUhoBnqsPx9cRXZrVvGMeSh_VjA?e=Iuqni0

Vous remarquerez que le jeu de données doit être sauvegardé sous la forme d'un fichier 7z. Veuillez installer 7-zip, Keka ou p7zip pour extraire le jeu de données.

### Windows
7-zip peut être téléchargé pour Windows en utilisant le lien suivant:
https://www.7-zip.org

### macOS
Keka peut être téléchargé pour MacOS en utilisant le lien suivant:
https://www.keka.io/en/

### Ubuntu
Sur Ubuntu, veuillez installer p7zip (ou tout autre logiciel supportant 7z) en lançant les commandes suivantes dans votre terminal:
```
sudo apt update
sudo apt install p7zip-full
```


Lors de l'extraction du fichier, assurez-vous d'utiliser 7-zip, p7zip ou Keka.

### MOT DE PASSE

Le mot de passe pour le fichier 7z est :

```
gT5&dK9zR2wQ!aP0eY3B#6vL1zXhF8j
```

### Structure du répertoire

Le dossier que vous avez téléchargé contient trois sous-dossiers : un dossier « yes », un dossier « no » et un dossier « CEC_test ». Les dossiers 'yes' et 'no' contiennent des images correspondant à des IRM saines et malsaines, tandis que le dossier 'CEC_test' sera utilisé par les directeurs à des fins de test.
```
/CEC_2025
│
├── /yes/
│ ├─── yes__1.png
├─── yes__2.png
│ └── ...
│
├─── /no/
│ ├─── no__1.png
├─── no__2.png
│ └── ...
│
└─── /CEC_test/
    ├── test__1.png
    ├─── test__2.png
    └── ...
```

# Potentiellement utile :

## Récupérer le jeu de données
Veuillez trouver ci-dessous un exemple de code (en Python, mais il n'est pas obligatoire d'utiliser Python) qui accède à ce dossier et le parcourt en utilisant le répertoire racine. 

Note : Ceci fait référence à la variable d'environnement CEC_2025_dataset que nous vous encourageons à configurer sur vos ordinateurs locaux. Dans ce dépôt, vous pouvez consulter ``Testing Information\setting_up_environment_variable.md`` pour plus d'informations sur l'utilisation des variables d'environnement. 

```Ruby
import os
from os import path

dataset_dir = os.getenv('CEC_2025_dataset') # Récupère le chemin de la variable d'environnement

# Initialise les listes pour contenir les données
image_paths = []
targets = []

total_images = 0

# Carte pour les étiquettes des cibles
label_map = {'no' : 0, 'yes' : 1, 'CEC_test' : 2} 

# Boucle sur les sous-répertoires du répertoire du jeu de données
for subdir in os.listdir(dataset_dir) :
    subdir_path = path.join(dataset_dir, subdir)

    if os.path.isdir(subdir_path) :
        subdir_path_list = os.listdir(subdir_path)
        for image in subdir_path_list :
            image_paths.append(path.join(subdir_path, image))
            targets.append(label_map[subdir])

        total_images += len(subdir_path_list)
        print(f « Nombre d'images dans le répertoire “{subdir}” : {len(subdir_path_list)}")

        print(f'Nombre total d'images : {total_images}')
```
Cela devrait donner quelque chose comme ceci :

```
Nombre d'images dans le répertoire 'no' : 8205
Nombre d'images dans le répertoire 'CEC_test' : 1
Nombre d'images dans le répertoire 'yes' : 9310
Nombre total d'images : 17516
```
## Vérification des chemins d'accès aux fichiers
Vous pouvez également vérifier les chemins d'accès aux fichiers de vos images :
```Ruby
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'image_path' : image_paths,
    'target' : targets
}, index=np.arange(0, total_images))

print(df.head())
```
Ce qui devrait donner quelque chose comme ceci :
```
                                          image_path target
0 /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
1 /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
2 /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
3 /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
4 /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
```
