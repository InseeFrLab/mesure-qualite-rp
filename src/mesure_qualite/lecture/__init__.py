"""Etape 2 du pipeline : lire les decoupes de l'etape 1 avec des modeles qui voient les
images, et juger ces lectures contre la verite arbitree par le SeRN.

    sujets      : quoi lire (bloc du gabarit, champs attendus) ; configs/lecture.yaml
    consignes   : le texte donne aux modeles (V0 simple, V1 detaillee)
    modeles     : un appel a un modele interne (SSP Cloud) ou du marche (Eden AI)
    campagne    : toutes les decoupes d'un sujet, en parallele, avec cache et garde-fous
    verite      : Fiqual + divergences du SeRN -> la verite, bulletin par bulletin
    comparaison : formes comparables et verdicts (juste, avouee, silencieuse)
    mesures     : tableaux, intervalles de confiance, McNemar, barre du PMQ
    rapport     : page HTML des erreurs, image de la decoupe a cote de chaque lecture

Exemple :

    from mesure_qualite import lecture as lc
    s = lc.sujet("naissance")
    texte = lc.consigne(s, "V1")
    r = lc.lire(lc.Lecteur("gemma4-26b-moe", "interne"), octets_jpeg, texte, s.noms_champs)
"""

from .campagne import Bilan, Compteurs, chemin_cache, en_long, lire_tout
from .comparaison import comparable, type_erreur, verdict, verdict_groupe
from .consignes import VERSIONS, consigne
from .mesures import barre_pmq, comparer, juger, mcnemar, resumer, wilson
from .modeles import Lecteur, Reponse, cle_api, extraire_json, lire, lister_internes
from .rapport import comparaison_verite, galerie_erreurs, lignes_en_erreur
from .sujets import Champ, Sujet, lire_configuration, sujet
from .verite import (
    cle,
    cle_image,
    colonnes_date_par_contenu,
    colonnes_par_votes,
    forme,
    lire_divergences,
    lire_fiqual,
    table_verite,
)

__all__ = [
    "VERSIONS",
    "Bilan",
    "Champ",
    "Compteurs",
    "Lecteur",
    "Reponse",
    "Sujet",
    "barre_pmq",
    "chemin_cache",
    "cle",
    "cle_api",
    "cle_image",
    "colonnes_date_par_contenu",
    "colonnes_par_votes",
    "comparable",
    "comparaison_verite",
    "comparer",
    "consigne",
    "en_long",
    "extraire_json",
    "forme",
    "galerie_erreurs",
    "juger",
    "lignes_en_erreur",
    "lire",
    "lire_configuration",
    "lire_divergences",
    "lire_fiqual",
    "lire_tout",
    "lister_internes",
    "mcnemar",
    "resumer",
    "sujet",
    "table_verite",
    "type_erreur",
    "verdict",
    "verdict_groupe",
    "wilson",
]
