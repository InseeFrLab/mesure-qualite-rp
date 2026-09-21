# mesure-qualite-rp

Production du fichier Fiqual du recensement de la population par des moyens
automatiques, en remplacement du prestataire de mesure de la qualité.

## Principe

Le Fiqual n'est pas une correction du Ficod : c'est une **seconde mesure,
obtenue indépendamment**, à partir des mêmes images et du même document de
consignes. Cette indépendance est ce qui donne un sens à la comparaison des
deux fichiers.

**Le dispositif ne doit donc jamais accéder au Ficod en production.** Celui-ci
n'est utilisé qu'en phase d'études, comme source d'exemples annotés et comme
moyen de vérifier la lecture des fichiers.

## Architecture

    image → découpe → perception (modèles) → règles (consignes) → codification → Fiqual

Deux étages séparés, pour une raison de fond : la perception est apprise, les
règles sont déterministes. Cette séparation rend les deux sources d'erreur
séparément mesurables, ce qu'une architecture monolithique ne permet pas.

## Organisation

    src/mesure_qualite/
        io/             lecture du bucket, manifeste des images
        parsing/        Ficod, Fiqual, divergences, dessins de fichier
        regles/         application du document de consignes
        codification/   communes, pays, nationalités
        perception/     classifieurs de cases, de chiffres, écriture manuscrite
        assemblage/     écriture du fichier Fiqual
        evaluation/     taux d'erreur, seuils, corrélation avec le Ficod
    notebooks/          exploration (non testé, non exécuté en production)
    configs/            dessins de fichier et positions, par millésime
    tests/              tests unitaires, dont les exemples du document de consignes
    docs/               notes de cadrage et spécifications

## Règles de contribution

- Aucune donnée dans le dépôt. Tout vient du bucket.
- Les notebooks sont commités **sans leurs sorties** (`nbstripout`), qui
  contiennent des images de questionnaires et des données individuelles.
- Les positions, dessins de fichier et modalités admises vivent dans
  `configs/`, jamais dans le code : le questionnaire change d'un millésime à
  l'autre.
- Le code qui se stabilise quitte les notebooks pour `src/`, avec ses tests.

## Installation
    uv sync --all-extras
    uv run nbstripout --install
# Installer la librairie systéme dont OpenCV a besoin
    sudo apt update && sudo apt install -y libgl1

# Claude code

J'utilise claude code pour debug et génération de certains codes que j'adapte. ça reste totalement sous contrôle.  
    
