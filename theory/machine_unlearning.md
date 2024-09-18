
# Machine Unlearning Fails to Remove Data Poisoning Attacks


Voici des compléments d'explication de l'article [Machine Unlearning Fails to Remove Data Poisoning Attacks
](https://arxiv.org/abs/2406.17216).

## Un exemple simple

Considérons un modèle d'intelligence artificielle qui tente d'apprendre la relation linéaire (vectorielle)
$$y = \bold{x} \bold{\theta}$$
en optimisant selon l'estimateur des moindres carrés :
$$\bold{\theta}^\star = \argmin_{\hat{\bold{\theta}}} l(\hat{\bold{\theta}}, (\bold{x}, y))$$

où l'on définit $\hat y := \hat{\bold{\theta}}(\bold{x})$, $y^\star := \bold{\theta}^\star(\bold{x})$ et $l(\hat{\bold{\theta}}, (\bold{x}, y)) := ||y - \hat y||^2$.

On obtient $\bold{\theta}^\star = X^\sharp \bold{y} = (\bold{X}^T \bold{X})^{-1} \bold{X}^T \bold{y}$ où $\bold{X}$ est la matrice dont les lignes sont les données explicatives $\bold{x}$.

Après empoisonnement, le modèle cherche à apprendre la relation
$$y = (\bold{x} + \xi) \bold{\theta}$$
Pour simplifier, imaginons que tout le jeu d'entraînement soit empoisonné de cette manière. D'après l'article, $\xi$ est centré et de faible variance, donc les paramètres $\bold{\theta}$ sont peu biaisés par l'empoisonnement.

Le modèle apprend cette fois les paramètres $\bold{\theta}^\star = (X + \Xi)^\sharp \bold{y} = (\bold{(X + \Xi)}^T \bold{(X + \Xi)})^{-1} \bold{(X + \Xi)}^T \bold{y}$ où $\Xi$ est la matrice correspondant aux perturbations $\xi_z$.


### Fuite des paramètres du modèle

L'article mentionne brièvement que l'attaquant, muni des données d'empoisonnement, pourrait faire fuiter les paramètres du modèle. Cependant, l'article ne donne pas de méthode pour en récupérer une partie.

#### Régression linéaire - un exemple simplifié

Voici une illustration d'une telle fuite de données. L'attquant calcule la quantité suivante sur $S_{\mathrm{poison}}$ :

$$
\begin{align*}
\mathrm{\widehat{Cov}}(y^\star, \xi) &= \underbrace{\mathrm{\widehat{Cov}}(\bold{\theta}^\star \bold{x}, \xi)}_{\text{$0$ car $\bold{x} \bot \xi$}} + \mathrm{\widehat{Cov}}(\bold{\theta}^\star \xi, \xi) \\
    &= \bold{\theta}^\star \widehat{\mathbb{V}}(\xi) \\
    &= \varepsilon^2 \bold{\theta}^\star
\end{align*}
$$
Dans le cas du modèle linéaire, l'attaquant connaît $S_{\mathrm{poison}}$, $\xi$ et $\varepsilon$, il peut donc en déduire les paramètres $\bold{\theta}^\star$.

Pour le modèle considéré, ce n'est pas très spectaculaire. Pour une attaque de type black-box, il est facile d'estimer ce paramètre en recalculant la régression linéaire de la manière suivante :
- Générer des données $\bold{x}$
- Collecter les valeurs renvoyées par le modèle $y^\star = \bold{\theta}^\star(\bold{x})$
- Effectuer la régression linéaire sur les couples $(\bold{x}, y^\star)$
Cela consiste à ré-entraîner soi-même un modèle à partir des sorties de $\bold{\theta}^\star$.

Toutefois, pour des modèles plus grands, cette méthode n'est pas aussi simple, pour les raisons suivantes :
- À partir d'un modèle uniquement accessible via une plateforme (car l'attaque est supposée black-box), il n'est pas toujours
faisable de générer massivement des données d'entraînement représentatives et de bonne qualité.
- L'attaquant n'a pas nécessairement des ressources suffisantes afin d'entraîner un grand modèle à partir de zéro.

En comparaison, l'estimation des paramètres $\bold{\theta}^\star$ du modèle consiste principalement en une étape d'inférence sur un sous-ensemble réduit $S_{\mathrm{poison}}$ des données d'entraînement. Aucune étape d'entraînement n'est nécessaire.
