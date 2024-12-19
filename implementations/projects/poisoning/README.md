# Projets d'empoisonnement de modèles simples

Prérequis : [Pytorch basics](https://pytorch.org/tutorials/beginner/basics/)

Nous avons vu le perceptron multi-couches appliqué à plusieurs problèmes : régression linéaire, logistique, classification. Il est temps de **tester l'empoisonnement sur ces modèles**. Pour chacun de ces modèles, il faut effectuer deux tâches :
1. **Implémenter une attaque** par empoisonnement des données dans un notebook Jupyter. On pourra faire varier les dimensions, les paramètres d'entraînement, les amplitudes et proportions d'empoisonnement, et donner les **résultats** sous forme de **figures** `matplotlib`. 
2. Rédiger **l'analyse mathématique** des résultats avec les figures obtenues. **Ce n'est pas forcément une démonstration**, mais au moins une explication.

Pour l'implémentation, ne pas hésiter à adapter le code du tutoriel de [cette page](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

L'analyse mathématique sera idéalement rédigée dans un document Markdown ou LaTeX séparé afin de présenter proprement les résultats. On peut la faire directement dans le notebook, mais c'est moins lisible.

> [!NOTE]
> Ce document pourra être enrichi avec plus de précisions.
> Je n'ai pas toutes les réponses aux questions posées dans les sujets ci-dessous.
> **Ce n'est pas un TD**, sentez vous libres de prendre des initiatives.
> Pour toute demande au sujet de ce document, utiliser [Github Discussions](https://github.com/orgs/AI-Safety-X2023/discussions).

Certaines fonctions du notebook sur le Machine Unlearning sont réutilisables, voir le fichier[`gaussian_poisoning.ipynb`](../../implementations/gaussian_poisoning.ipynb).

## Régression linéaire

_Fichier à modifier_ : [`linear_regression.ipynb`](./linear_regression.ipynb)

### Implémentation

1. Créer les données :
   1. Générer un jeu de données `S` de couples $(x, y)$ où $y = w x + b + \varepsilon$ où $\varepsilon$ est une gaussienne centrée de variance modérée. Garder les paramètres $\theta := (w, b)$ pour la suite.
   2. Visualiser les données.
   3. Séparer les données en deux datasets : un jeu d'entraînement (`S_train`, 80 %) et un jeu de validation (`S_test`, 20 %).
2. Entraîner un modèle $\theta_0$ de régression linéaire (`nn.Linear`) avec la perte des moindres carrés (`torch.nn.MSELoss`) et la descente de gradient stochastique (`torch.optim.SGD`).
3. Créer un nouveau jeu de données corrompu `S_corr` à partir `S` en empoisonnant une fraction $\alpha \in [0, 1]$ des données $(x, y)$ : pour cette partie à empoisonner, on remplace $y$ par $y + \eta$ où $\eta$ un bruit gaussien centré de variance modérée.
4. Visualiser les données corrompues.
5. Créer un nouveau modèle $\theta_1$ et l'entraîner sur ce nouveau jeu de données corrompu. Afficher la précision de $\theta_1$ sur `S_test`.
6. Dans les sous-questions suivantes, on va créer de **nouveaux** modèles avec des paramètres différents (écrire une fonction pour éviter de répéter le code de la question précédente). Afficher les différentes courbes de précision lorsque l'on fait varier un paramètre :
   1. Que se passe-t-il si $\eta$ n'est pas centré ? À partir de maintenant, $\eta$ sera biaisé et de variance élevée, et $\alpha$ élevé.
   2. La variance de $\eta$
   3. La valeur de $\alpha$
   4. _Optionnel_ : les valeurs `epochs` et `learning_rate` (essayer des valeurs plus faibles)
   5. La dimension de `x` et de `y`
   6. La fonction de perte (essayer en priorité `torch.nn.L1Loss`, puis `torch.nn.HuberLoss`)
7. _Bonus_ : que se passe-t-il si l'on rajoute des couches (_hidden layers_) ?
8. _Bonus_ : que se passe-t-il avec peu de données ?
9. _Bonus_ : que se passe-t-il si au lieu de créer un nouveau modèle $\theta_1$, on continue l'entraînement du modèle $\theta$ sur `S_corr` ?

## Analyse

1. Montrer que le modèle $\theta_0$ est l'estimateur des moindres carrés de $\theta$ et rappeler ses propriétés d'optimalité (il faut utiliser le cours de statistiques).
2. Expliquer pourquoi les résultats sont différents selon que $\eta$ est centré ou non.
3. À partir des courbes, établir un résultat (quantitatif) empirique sur l'imprécision du modèle lorsqu'on fait varier $\eta$ ou $\alpha$. Si possible, le démontrer.
4. Établir une relation empirique entre l'imprécision du modèle et la dimension de `x` et de `y`.
5. Expliquer pourquoi la fonction de perte $L^1$ est moins vulnérable aux valeurs extrêmes (_outliers_) que la perte des moindres carrés $L^2$. Faire le lien avec la médiane et avec approches de _data cleaning_ (élimination de valeurs extrêmes). Discuter de l'utilité des approches hybrides comme `HuberLoss`.

## Régression logistique

_Fichier à modifier_ : [`logistic_regression.ipynb`](./logistic_regression.ipynb)

La régression logistique fonctionne un peu comme la régression linéaire, cependant c'est un problème de classification (ici binaire).

## Implémentation

1. Créer les données :
   1. Générer un jeu de données `S` de couples $(x, y)$ où $x \in \mathbb{R}^2$ et $y = \mathbf{1}_{w x + b > 0}$. Garder les paramètres $\theta := (w, b)$ pour la suite.
   2. Visualiser les données et la frontière $w x + b = 0$.
   3. Séparer les données en deux datasets : un jeu d'entraînement (`S_train`, 80 %) et un jeu de validation (`S_test`, 20 %).
2. Entraîner un modèle $\theta_0$ de régression logistique (`nn.Linear` + `nn.Sigmoid`) avec la perte _binary cross-entropy_ (`torch.nn.BCELoss`) et la descente de gradient stochastique (`torch.optim.SGD`).
3. Créer un nouveau jeu de données corrompu `S_corr` à partir `S` en appliquant du _label flipping_ à une fraction $\alpha \in [0, 1]$ des données $(x, y)$ : pour cette partie, on remplace le label $y$ par son opposé, $1 - y$.
4. Visualiser les données corrompues.
5. Créer un nouveau modèle  $\theta_1$ et l'entraîner sur ce nouveau jeu de données corrompu. Afficher la précision de $\theta_1$ sur `S_test`.
6. Dans les sous-questions suivantes, on va créer de **nouveaux** modèles avec des paramètres différents (écrire une fonction pour éviter de répéter le code de la question précédente). Afficher les différentes courbes de précision lorsque l'on fait varier un paramètre.
   1. Faire varier la valeur de $\alpha$.
   2. Créer un nouveau jeu de données de sorte que les valeurs de $x$ ne sont pas réparties uniformément, et refaire l'expérience. Par exemple, on concentre les valeurs de $x$ autour de $1$ ou plusieurs clusters (avec un mélange gaussien par exemple). Faire en sorte qu'au moins un des clusters soit proche de la frontière de classification. Que se passe-t-il ? On utilisera ce nouveau jeu de données dans la suite.
   3. Que se passe-t-il si l'on applique le _label flipping_ spécifiquement sur les points $x$ très proches de la frontière ?
   4. Faire varier la dimension de `x`.
7. _Bonus_ : que se passe-t-il si l'on rajoute des couches (_hidden layers_) ?
8. _Bonus_ : que se passe-t-il avec peu de données ?
9.  _Bonus_ : que se passe-t-il si au lieu de créer un nouveau modèle $\theta_1$, on continue l'entraînement du modèle $\theta$ sur `S_corr` ?

## Analyse
   1. Faire le lien avec les chapitres sur la classification dans le cours de Statistiques.
   2. Expliquer brièvement le concept du mélange gaussien.
   3. Établir une relation empirique entre l'imprécision du modèle et la proportion de points "proches" de la frontière.
   4. Faire le lien avec le concept du jailbreaking. Pour illustration, voir la slide 26 : _Évasion (jailbreaking)_ de la [conférence de Lê Nguyên Hoang en décembre 2024](https://science4all.org/wp-content/uploads/2024/12/piaf.pdf)
   5. Établir une relation empirique entre l'imprécision du modèle et la dimension de `x`.
   6. Peut-on imaginer un moyen d'identifier l'empoisonnement, de filtrer les données suspicieuses ou d'agréger les gradients d'un mini-batch de manière robuste ? Exemple d'approche sur [Scikit-Learn](https://scikit-learn-extra.readthedocs.io/en/stable/modules/robust.html)

## Classification d'images

_Fichier à modifier_ : [`image_classification.ipynb`](./image_classification.ipynb)

Comme pour la régression logistique, on appliquera du label flipping, mais sur un jeu de données avec plusieurs classes (labels). On peut le générer ou utiliser des données réelles comme celles du [MNIST digits](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST), [Fashion MNIST](https://pytorch.org/tutorials//beginner/basics/data_tutorial.html) ou encore [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10). Il faut cette fois enlever la fonction d'activation et utiliser `torch.nn.CrossEntropyLoss` par exemple.

Cette fois, il est plus difficile de visualiser les données et de déterminer les frontières de classification. On pourra essayer d'effectuer un _label flipping_ ciblé sur les données où la fonction de perte est maximale (ou la probabilité de prédiction est minimale).