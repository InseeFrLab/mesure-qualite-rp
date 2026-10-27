"""Mettre une valeur sous une forme comparable, puis juger une lecture.

Les deux cotes (lecture du modele et verite) passent par la meme normalisation : on ne
juge pas un modele sur un accent, un zero devant un jour ou « BD » ecrit pour
« BOULEVARD ».

Verdicts (les memes que les notebooks 10 a 13) :
  juste       : meme valeur que la verite ;
  avouee      : le modele ne lit rien ou dit ILLISIBLE alors qu'il y a quelque chose ;
  relue       : fausse, mais un signal l'envoie a un humain ;
  silencieuse : fausse et personne ne le voit ;
  ecartee     : verite inconnue (divergence non arbitree par le SeRN).
"""

from __future__ import annotations

import math
import re
import unicodedata

ILLISIBLE = "ILLISIBLE"
VIDES = {"", "VIDE", "NEANT", "AUCUN", "AUCUNE", "NULL", "NONE", "N A", "NA"}

MOIS = {
    "JANVIER": 1,
    "JANV": 1,
    "JAN": 1,
    "FEVRIER": 2,
    "FEVR": 2,
    "FEV": 2,
    "MARS": 3,
    "MAR": 3,
    "AVRIL": 4,
    "AVR": 4,
    "MAI": 5,
    "JUIN": 6,
    "JUILLET": 7,
    "JUIL": 7,
    "AOUT": 8,
    "SEPTEMBRE": 9,
    "SEPT": 9,
    "SEP": 9,
    "OCTOBRE": 10,
    "OCT": 10,
    "NOVEMBRE": 11,
    "NOV": 11,
    "DECEMBRE": 12,
    "DEC": 12,
}

# Abreviations de voies et de lieux : appliquees des deux cotes avant de comparer.
ABREVIATIONS = {
    "R": "RUE",
    "BD": "BOULEVARD",
    "BLD": "BOULEVARD",
    "BOUL": "BOULEVARD",
    "AV": "AVENUE",
    "AVE": "AVENUE",
    "CH": "CHEMIN",
    "CHE": "CHEMIN",
    "CHEM": "CHEMIN",
    "IMP": "IMPASSE",
    "PL": "PLACE",
    "RTE": "ROUTE",
    "ALL": "ALLEE",
    "LOT": "LOTISSEMENT",
    "RES": "RESIDENCE",
    "RESID": "RESIDENCE",
    "QUA": "QUARTIER",
    "QRT": "QUARTIER",
    "SQ": "SQUARE",
    "CRS": "COURS",
    "FG": "FAUBOURG",
    "FBG": "FAUBOURG",
    "HAM": "HAMEAU",
    "PAS": "PASSAGE",
    "PROM": "PROMENADE",
    "SENT": "SENTIER",
    "LD": "LIEU DIT",
    "LIEUDIT": "LIEU DIT",
    "B": "BIS",
    "T": "TER",
    "Q": "QUATER",
    "BAT": "BATIMENT",
    "BT": "BATIMENT",
    "ESC": "ESCALIER",
    "APT": "APPARTEMENT",
    "APP": "APPARTEMENT",
    "IMM": "IMMEUBLE",
    "ST": "SAINT",
    "STE": "SAINTE",
}
ABREVIATIONS_LIEU = {"ST": "SAINT", "STE": "SAINTE", "S": "SUR", "SS": "SOUS"}


def manquant(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def normaliser(t: object) -> str:
    """Majuscules, sans accents ni ponctuation, espaces simples."""
    if manquant(t):
        return ""
    t = unicodedata.normalize("NFKD", str(t).strip())
    t = "".join(c for c in t if not unicodedata.combining(c)).upper()
    t = re.sub(r"[^A-Z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _entier(v: str) -> str:
    s = v.replace(" ", "")
    return str(int(s)) if s.isdigit() else v


def comparable(valeur: object, type_: str) -> str:
    """La forme sur laquelle on compare. « » = rien d'ecrit ; ILLISIBLE garde tel quel."""
    v = normaliser(valeur)
    if v in VIDES:
        return ""
    if v == ILLISIBLE:
        return v
    if type_ == "entier":
        return _entier(v)
    if type_ == "mois":
        return str(MOIS[v]) if v in MOIS else _entier(v)
    if type_ == "departement":
        s = v.replace(" ", "")
        if s.isdigit():
            return str(int(s))
        return {"2A": "2A", "2B": "2B"}.get(
            s, " ".join(ABREVIATIONS_LIEU.get(m, m) for m in v.split())
        )
    if type_ == "lieu":
        return " ".join(ABREVIATIONS_LIEU.get(m, m) for m in v.split())
    if type_ == "voie":
        return " ".join(ABREVIATIONS.get(m, m) for m in v.split())
    return v


def verdict(lu: str | None, vrai: str | None, signale: bool = False) -> str:
    """Juge une lecture deja rendue comparable. vrai=None : verite inconnue."""
    if vrai is None or manquant(vrai):
        return "ecartee"
    lu = "" if lu is None or manquant(lu) else lu
    if lu == vrai:
        return "juste"
    if lu in ("", ILLISIBLE):
        return "avouee"
    return "relue" if signale else "silencieuse"


def verdict_groupe(verdicts: list[str]) -> str:
    """Un groupe (ex. la DATE = jour, mois, annee) est juste si tous ses champs le sont."""
    if any(v == "ecartee" for v in verdicts):
        return "ecartee"
    if all(v == "juste" for v in verdicts):
        return "juste"
    if "silencieuse" in verdicts:
        return "silencieuse"
    if "relue" in verdicts:
        return "relue"
    return "avouee"


def distance(a: str, b: str) -> int:
    """Distance d'edition (Levenshtein)."""
    if a == b:
        return 0
    prec = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cour = [i]
        for j, cb in enumerate(b, 1):
            cour.append(min(prec[j] + 1, cour[j - 1] + 1, prec[j - 1] + (ca != cb)))
        prec = cour
    return prec[-1]


def type_erreur(lu: str, vrai: str) -> str:
    """Le genre d'erreur, sans montrer les valeurs (pour un tableau partageable)."""
    if lu in ("", ILLISIBLE):
        return "rien lu ou illisible"
    if vrai == "":
        return "lu alors que rien n'est ecrit"
    if lu.replace(" ", "") == vrai.replace(" ", ""):
        return "espace ou tiret"
    if lu.isdigit() and vrai.isdigit():
        if len(lu) != len(vrai):
            return "chiffre en trop ou en moins"
        return "1 chiffre" if distance(lu, vrai) == 1 else "plusieurs chiffres"
    if set(lu.split()) < set(vrai.split()) or set(vrai.split()) < set(lu.split()):
        return "mot en trop ou en moins"
    d = distance(lu, vrai)
    return (
        "1 lettre" if d == 1 else "2 lettres" if d == 2 else "autre (3 lettres ou plus)"
    )
