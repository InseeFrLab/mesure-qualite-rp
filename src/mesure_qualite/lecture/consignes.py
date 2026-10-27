"""Les consignes donnees aux modeles : le meme texte pour tous, seul le modele change.

Deux versions, construites a partir du sujet (configs/lecture.yaml) :
  V0 « simple »    : ce qu'il faut lire, en JSON ;
  V1 « detaillee » : les memes champs, plus les pieges connus (chiffres proches, cases,
                     abreviations a ne pas developper, rien a completer de memoire).
"""

from __future__ import annotations

import json

from .sujets import Sujet

VERSIONS = {"V0": "simple", "V1": "detaillee"}

_REGLES_COMMUNES = """Regles :
- Recopie seulement l'ecriture manuscrite, pas les libelles ni les exemples imprimes.
- En MAJUSCULES, sans accent.
- Ne corrige rien, ne devine pas, ne complete rien de memoire.
- Si une case est vide, mets VIDE. Si c'est ecrit mais illisible, mets ILLISIBLE."""

_PIEGES = {
    "entier": "chiffres proches (1/7, 4/9, 0/6, 3/8, 5/6) : regarde-les un par un ; "
    "recopie les chiffres, sans les mettre en lettres",
    "mois": "si le mois est ecrit en lettres, recopie-le en lettres ; en chiffres, en chiffres",
    "departement": "recopie le numero ou le nom tel qu'il est ecrit (2A, 2B, 971...)",
    "lieu": "ne remplace pas un nom rare par un nom plus courant ; garde ST, STE tels qu'ecrits",
    "voie": "ne developpe pas les abreviations (BD reste BD, AV reste AV)",
    "texte": "ne remplace pas un mot rare par un mot plus courant",
}


def _json_attendu(sujet: Sujet) -> str:
    return json.dumps({c: "..." for c in sujet.noms_champs})


def consigne(sujet: Sujet, version: str = "V1") -> str:
    if version not in VERSIONS:
        raise ValueError(f"consigne {version!r} inconnue ({sorted(VERSIONS)})")
    liste = "\n".join(f"- {c.nom} : {c.aide}" for c in sujet.champs)
    entete = (
        f"Cette image montre {sujet.document} du recensement francais.\n"
        f"Elle contient {sujet.cible}, ecrite a la main a cote de libelles imprimes."
    )
    if version == "V0":
        return (
            f"{entete}\n\nRecopie ce qui est ecrit a la main pour chaque champ :\n{liste}\n\n"
            f"{_REGLES_COMMUNES}\n\nReponds uniquement en JSON : {_json_attendu(sujet)}"
        )
    pieges = sorted({c.type for c in sujet.champs})
    attention = "\n".join(f"- {_PIEGES[t]}" for t in pieges)
    return (
        f"{entete}\n\nRecopie l'ECRITURE MANUSCRITE, champ par champ :\n{liste}\n\n"
        f"{_REGLES_COMMUNES}\n\nAttention particuliere :\n{attention}\n"
        "- Si le texte deborde d'une case a l'autre, rattache chaque morceau a son libelle.\n\n"
        f"Reponds uniquement en JSON, avec exactement ces cles : {_json_attendu(sujet)}"
    )
