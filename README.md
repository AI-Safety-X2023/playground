# Playground

Un dépôt centralisé pour implémenter des techniques en sécurité de l'IA

> [!WARNING] Disclaimer
> Dans sa forme actuelle, ce code n'a pas pour vocation à être présenté ou réutilisé.

## Structure du dépôt

- `theory` : contient les articles de recherche dans le sous-dossier `references`. Les notes de lectures doivent être déposées dans le sous-dossier `literature_review`.
- `implementations` : contient des courtes implémentations de techniques illustrées dans les articles, et nos propres méthodes.

## Exécuter les implémentations

### Installer les dépendances Python

Deux méthodes décrites ci-dessous

#### Installer les dépendances avec Anaconda

```bash
conda install -c conda-forge -r implementations/requirements.txt
```

#### Installer les dépendances avec `pip`

```bash
cd implementations
pip install -r implementations/requirements.txt
```