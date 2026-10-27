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
        decoupage/      étape 1 : gabarit du questionnaire, recalage des scans, découpe en blocs
        lecture/        étape 2 : lecture des blocs par des modèles (internes, Eden AI), jugée contre le SeRN
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

## Étape 1 : découpage en blocs

Le masque (gabarit) d'un questionnaire est calculé à partir de son **PDF vierge** :
positions exactes des cadres, numéros de questions, cases, champs et noms de
variables. Chaque scan est ensuite recalé sur la page vierge (points ORB, RANSAC,
transformation affine) puis découpé bloc par bloc. Aucun modèle appris : le résultat
est déterministe et reproductible. Les chemins viennent de `configs/campagnes.yaml`.

    # gabarit d'une nouvelle version du questionnaire (JSON + pages de référence + image de contrôle)
    python -m mesure_qualite.decoupage construire --pdf "s3://projet-mesure-qualite-rp/documentation/Questionnaires/2025-imprime-3-BI avec VAR.pdf"

    # découpe des scans d'une campagne (essai sur 20 scans, puis tout ; reprise automatique)
    python -m mesure_qualite.decoupage decouper --campagne RG25 --formulaire BI --limite 20
    python -m mesure_qualite.decoupage decouper --campagne RG25 --formulaire BI

    # ce qui a changé entre deux versions d'un questionnaire
    python -m mesure_qualite.decoupage comparer FLRM_2022 FLR_2027

Sortie : `data/decoupes/<campagne>/<formulaire>/<bloc>/<scan>.jpg` et un index
`index_<formulaire>_<campagne>.csv` (scan, bloc, chemin, méthode, qualité du recalage),
point de départ des étapes suivantes. Dans un notebook :

    from mesure_qualite import decoupage as dc
    r = dc.Recaleur.depuis_dossier("s3://projet-mesure-qualite-rp/data/decoupes/gabarits", "BI_2025")
    image = dc.decouper_selection(scan, r, variables=["NUMVOI", "NOMVOI"], avec_intitule=True)

## Étape 2 : lecture des blocs

Un **sujet** (`configs/lecture.yaml`) dit quel bloc lire et quels champs attendre :
`naissance` (question 2 : jour, mois, année, commune, département, pays) ou
`adresse_travail` (question 23). Le même appel sert pour les modèles internes du
SSP Cloud et pour ceux d'Eden AI ; les lectures sont gardées en cache (jamais
repayées) et jugées contre la vérité arbitrée par le SeRN.

    from mesure_qualite import lecture as lc
    s = lc.sujet("naissance")
    r = lc.lire(lc.Lecteur("gemma4-26b-moe", "interne"), octets_jpeg, lc.consigne(s, "V1"), s.noms_champs)

Le notebook `lecture_modeles_internes_et_eden.ipynb` enchaîne les deux étapes et
compare les lecteurs (justes, erreurs silencieuses, McNemar, barre du PMQ, coût).

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

# Site et présentations

Le site (`docs/`) présente les objectifs, le pipeline, les méthodes et les résultats. Les
diaporamas (`docs/presentations/`) sont un sous-projet Quarto séparé, dans la charte Insee.

    bash scripts/installer_extensions.sh        # charte Insee + icônes (44 Mo, non versionnées)
    quarto render docs/presentations            # d'abord les diaporamas
    quarto render docs                          # puis le site, dans docs/_site
    quarto preview docs                         # pour travailler une page

Les liens vers les expériences de l'instance MLflow du projet (`fiqual-banc-essai` = 1, `fiqual-typl` = 59)
sont dans `docs/_variables.yml` ; le diaporama a les siens écrits en clair.

**Publication.** Le workflow `.github/workflows/site.yml` publie sur GitHub Pages à chaque push
qui touche `docs/`. À activer une fois : *Settings > Pages > Source : GitHub Actions*.

# Claude code

Claude code est utilisé pour debug et génération de certains codes que j'adapte.
