"""Mettre une lecture au format du Fiqual, selon les consignes de saisie du PMQ.

Le Fiqual n'est pas une transcription brute : c'est l'ecriture passee par les consignes
de saisie (annexe 2, versions 2026 et 2027). Un modele qui recopie fidelement peut donc
etre juge faux a tort. On applique ici les memes regles aux lectures avant de comparer :

  JOUR, MOIS   : 2 chiffres (5 -> 05), mois en lettres -> numero, date impossible -> blanc
  ANNEE        : 4 chiffres ; 2 chiffres -> siecle complete (85 -> 1985) ; hors de
                 [collecte - 125 ; collecte] -> blanc
  DEPARTEMENT  : 01 a 95, 2A, 2B, 971 a 978, 984 a 989, 98, 99 ; nom -> numero ;
                 code postal -> 2 premiers chiffres ; sinon blanc
  COMMUNE      : ST -> SAINT ; arrondissements de Paris, Lyon, Marseille au libelle du
                 dictionnaire (PARIS 18E ARRONDISSEMENT) ; departement ecrit dans le
                 libelle retire
  PAYS         : FRANCE avec un departement metropolitain -> blanc ; IRAK -> IRAQ
  NUMERO (Q23) : le plus petit numero (10-12 -> 10) ; INDICE : BIS -> B, TER -> T...
  NOM, PRENOM  : civilite retiree (M, MME...) ; EPOUSE, NEE, VEUVE gardes

ILLISIBLE et VIDE sont gardes tels quels : ce sont des abstentions du modele, jugees
comme telles (envoi a un humain).
"""

from __future__ import annotations

import calendar
import re

import pandas as pd

from .comparaison import ILLISIBLE, VIDES, normaliser
from .sujets import Sujet

DEPARTEMENTS = {
    "AIN": "01", "AISNE": "02", "ALLIER": "03", "ALPES DE HAUTE PROVENCE": "04",
    "HAUTES ALPES": "05", "ALPES MARITIMES": "06", "ARDECHE": "07", "ARDENNES": "08",
    "ARIEGE": "09", "AUBE": "10", "AUDE": "11", "AVEYRON": "12", "BOUCHES DU RHONE": "13",
    "CALVADOS": "14", "CANTAL": "15", "CHARENTE": "16", "CHARENTE MARITIME": "17",
    "CHER": "18", "CORREZE": "19", "CORSE DU SUD": "2A", "HAUTE CORSE": "2B",
    "COTE D OR": "21", "COTES D ARMOR": "22", "COTES DU NORD": "22", "CREUSE": "23",
    "DORDOGNE": "24", "DOUBS": "25", "DROME": "26", "EURE": "27", "EURE ET LOIR": "28",
    "FINISTERE": "29", "GARD": "30", "HAUTE GARONNE": "31", "GERS": "32", "GIRONDE": "33",
    "HERAULT": "34", "ILLE ET VILAINE": "35", "INDRE": "36", "INDRE ET LOIRE": "37",
    "ISERE": "38", "JURA": "39", "LANDES": "40", "LOIR ET CHER": "41", "LOIRE": "42",
    "HAUTE LOIRE": "43", "LOIRE ATLANTIQUE": "44", "LOIRET": "45", "LOT": "46",
    "LOT ET GARONNE": "47", "LOZERE": "48", "MAINE ET LOIRE": "49", "MANCHE": "50",
    "MARNE": "51", "HAUTE MARNE": "52", "MAYENNE": "53", "MEURTHE ET MOSELLE": "54",
    "MEUSE": "55", "MORBIHAN": "56", "MOSELLE": "57", "NIEVRE": "58", "NORD": "59",
    "OISE": "60", "ORNE": "61", "PAS DE CALAIS": "62", "PUY DE DOME": "63",
    "PYRENEES ATLANTIQUES": "64", "HAUTES PYRENEES": "65", "PYRENEES ORIENTALES": "66",
    "BAS RHIN": "67", "HAUT RHIN": "68", "RHONE": "69", "HAUTE SAONE": "70",
    "SAONE ET LOIRE": "71", "SARTHE": "72", "SAVOIE": "73", "HAUTE SAVOIE": "74",
    "PARIS": "75", "SEINE MARITIME": "76", "SEINE ET MARNE": "77", "YVELINES": "78",
    "DEUX SEVRES": "79", "SOMME": "80", "TARN": "81", "TARN ET GARONNE": "82", "VAR": "83",
    "VAUCLUSE": "84", "VENDEE": "85", "VIENNE": "86", "HAUTE VIENNE": "87", "VOSGES": "88",
    "YONNE": "89", "TERRITOIRE DE BELFORT": "90", "ESSONNE": "91", "HAUTS DE SEINE": "92",
    "SEINE SAINT DENIS": "93", "VAL DE MARNE": "94", "VAL D OISE": "95",
    "GUADELOUPE": "971", "MARTINIQUE": "972", "GUYANE": "973", "LA REUNION": "974",
    "REUNION": "974", "SAINT PIERRE ET MIQUELON": "975", "MAYOTTE": "976",
    "SAINT BARTHELEMY": "977", "SAINT MARTIN": "978", "TERRES AUSTRALES": "984",
    "WALLIS ET FUTUNA": "986", "POLYNESIE FRANCAISE": "987", "NOUVELLE CALEDONIE": "988",
    "CLIPPERTON": "989",
}  # fmt: skip
OUTRE_MER = {v for v in DEPARTEMENTS.values() if len(v) == 3}
DEPARTEMENTS_ADMIS = (
    {f"{i:02d}" for i in range(1, 96) if i != 20}
    | {"2A", "2B", "98", "99"}
    | {str(i) for i in range(971, 979)}
    | {str(i) for i in range(984, 990)}
)
ARRONDISSEMENTS = {"PARIS": ("75", 20), "LYON": ("69", 9), "MARSEILLE": ("13", 16)}
ROMAINS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9,
    "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16,
    "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
}  # fmt: skip
MOIS_LETTRES = {
    "JANVIER": 1, "JANV": 1, "FEVRIER": 2, "FEV": 2, "FEVR": 2, "MARS": 3, "AVRIL": 4,
    "AVR": 4, "MAI": 5, "JUIN": 6, "JUILLET": 7, "JUIL": 7, "AOUT": 8, "SEPTEMBRE": 9,
    "SEPT": 9, "OCTOBRE": 10, "OCT": 10, "NOVEMBRE": 11, "NOV": 11, "DECEMBRE": 12, "DEC": 12,
}  # fmt: skip
PAYS_EQUIVALENTS = {"IRAK": "IRAQ", "IRAKIEN": "IRAQ", "IRAKIENNE": "IRAQ"}
INDICES = {"BIS": "B", "TER": "T", "QUATER": "Q", "QUINQUIES": "C", "QUINQUESE": "C"}


def _abstention(v: str) -> bool:
    return v in VIDES or v == ILLISIBLE


# --- date de naissance ---------------------------------------------------------------
def jour_ou_mois(valeur: object, maximum: int) -> str:
    """'5' -> '05', 'mars' -> '03' ; hors de 1..maximum -> blanc."""
    v = normaliser(valeur)
    if _abstention(v):
        return v
    if v in MOIS_LETTRES and maximum == 12:
        return f"{MOIS_LETTRES[v]:02d}"
    s = v.replace(" ", "")
    if not s.isdigit() or len(s) > 2:
        return ""
    n = int(s)
    return f"{n:02d}" if 1 <= n <= maximum else ""


def annee(valeur: object, collecte: int) -> str:
    """4 chiffres dans [collecte - 125 ; collecte] ; 2 chiffres -> siecle complete en
    privilegiant un age de 0 a 100 ans (85 -> 1985, 08 -> 2008 en 2025, 30 -> 1930)."""
    v = normaliser(valeur)
    if _abstention(v):
        return v
    s = v.replace(" ", "")
    if not s.isdigit():
        return ""
    bas = collecte - 125
    if len(s) == 4:
        return s if bas <= int(s) <= collecte else ""
    if len(s) == 2:
        candidats = [
            c * 100 + int(s)
            for c in (20, 19, 18)
            if bas <= c * 100 + int(s) <= collecte
        ]
        plausibles = [a for a in candidats if collecte - a <= 100] or candidats
        return str(plausibles[0]) if plausibles else ""
    return ""


def date_valide(jour: str, mois: str, an: str) -> bool:
    """Le jour existe dans ce mois (29 fevrier : annee bissextile si elle est connue)."""
    if not (jour.isdigit() and mois.isdigit()):
        return True
    j, m = int(jour), int(mois)
    if an.isdigit() and len(an) == 4:
        return 1 <= j <= calendar.monthrange(int(an), m)[1]
    return 1 <= j <= (29 if m == 2 else calendar.monthrange(2001, m)[1])


# --- lieu ------------------------------------------------------------------------------
def departement(valeur: object) -> str:
    """Numero ou nom de departement -> code admis par le Fiqual, sinon blanc."""
    v = normaliser(valeur)
    if _abstention(v):
        return v
    if v in DEPARTEMENTS:
        return DEPARTEMENTS[v]
    s = v.replace(" ", "")
    if s in ("2A", "2B"):
        return s
    if s.isdigit():
        if len(s) == 5:  # code postal
            s = s[:3] if s[:2] in ("97", "98") else s[:2]
        if len(s) == 1:
            s = "0" + s
        return s if s in DEPARTEMENTS_ADMIS else ""
    return ""


def _numero_arrondissement(mot: str) -> int | None:
    m = re.fullmatch(r"(\d{1,2})(ER|E|EME|IEME)?", mot)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"([IVX]+)(ER|E|EME)?", mot)
    return ROMAINS.get(m.group(1)) if m else None


def commune(valeur: object) -> str:
    """Libelle de commune comme le restitue le dictionnaire des communes."""
    v = normaliser(valeur)
    if _abstention(v):
        return v
    mots = v.split()
    # code postal seul ou en tete/fin : 75018 -> PARIS 18E ARRONDISSEMENT
    for i, m in enumerate(mots):
        if re.fullmatch(r"\d{5}", m):
            dep, n = m[:2], int(m[3:])
            ville = next(
                (c for c, (d, mx) in ARRONDISSEMENTS.items() if d == dep), None
            )
            reste = [x for j, x in enumerate(mots) if j != i]
            if (
                ville
                and 1 <= n <= ARRONDISSEMENTS[ville][1]
                and (not reste or reste[0] == ville)
            ):
                mots = [ville, str(n)]
            else:
                mots = reste
            break
    mots = [{"ST": "SAINT", "STE": "SAINTE"}.get(m, m) for m in mots]
    if mots and mots[0] in ARRONDISSEMENTS:
        ville = mots[0]
        dep, maximum = ARRONDISSEMENTS[ville]
        reste = [
            m for m in mots[1:] if m not in ("ARRONDISSEMENT", "ARR", "ARRDT", dep)
        ]
        n = _numero_arrondissement(reste[0]) if len(reste) == 1 else None
        if n and 1 <= n <= maximum:
            return f"{ville} {n}{'ER' if n == 1 else 'E'} ARRONDISSEMENT"
        return ville if not reste else " ".join([ville, *reste])
    # departement ecrit a la fin du libelle (BREST 29) : retire
    if len(mots) > 1 and departement(mots[-1]) and mots[-1].isdigit():
        mots = mots[:-1]
    return " ".join(mots)


def pays(valeur: object, dep: str) -> str:
    """FRANCE avec un departement metropolitain -> blanc ; equivalences (IRAK -> IRAQ)."""
    v = normaliser(valeur)
    if _abstention(v):
        return v
    v = PAYS_EQUIVALENTS.get(v, v)
    if v == "FRANCE" and dep and dep not in OUTRE_MER and dep not in ("98", "99"):
        return ""
    return v


# --- adresse (question 23) ---------------------------------------------------------------
def numero_voie(valeur: object) -> str:
    """Le plus petit numero de l'adresse principale (10-12 -> 10)."""
    v = normaliser(valeur)
    if _abstention(v):
        return v
    nombres = [int(n) for n in re.findall(r"\d+", v)]
    return str(min(nombres)) if nombres else ""


def indice(valeur: object) -> str:
    v = normaliser(valeur)
    if _abstention(v):
        return v
    return INDICES.get(v, v)


# --- nom et prenom ----------------------------------------------------------------------
CIVILITES = {"M", "MR", "MME", "MLLE", "MELLE", "MONSIEUR", "MADAME", "MADEMOISELLE"}


def nom_ou_prenom(valeur: object) -> str:
    """Civilite retiree (M, MME, MONSIEUR...) ; EPOUSE, NEE, VEUVE... gardes tels quels ;
    contenu des parentheses garde (normaliser retire seulement les signes)."""
    v = normaliser(valeur)
    if _abstention(v):
        return v
    mots = v.split()
    while len(mots) > 1 and mots[0] in CIVILITES:
        mots = mots[1:]
    return " ".join(mots)


# --- application a une table de lectures -------------------------------------------------
def au_format_fiqual(
    lectures: pd.DataFrame, sujet: Sujet, collecte: int
) -> pd.DataFrame:
    """Table (cle, champ, lu) -> la meme table, chaque lecture mise au format du Fiqual.
    Ajoute la colonne « lu_brut » (la lecture telle que le modele l'a rendue)."""
    d = lectures.copy()
    d["lu_brut"] = d["lu"]
    large = d.pivot_table(index="cle", columns="champ", values="lu", aggfunc="first")
    large = large.reindex(columns=sujet.noms_champs).fillna("")
    sortie = pd.DataFrame(index=large.index)
    if sujet.nom == "identite":
        sortie["NOM"] = large["NOM"].map(nom_ou_prenom)
        sortie["PRENOM"] = large["PRENOM"].map(nom_ou_prenom)
    elif sujet.nom == "naissance":
        sortie["JOUR"] = large["JOUR"].map(lambda v: jour_ou_mois(v, 31))
        sortie["MOIS"] = large["MOIS"].map(lambda v: jour_ou_mois(v, 12))
        sortie["ANNEE"] = large["ANNEE"].map(lambda v: annee(v, collecte))
        faux = [
            not date_valide(j, m, a)
            for j, m, a in zip(sortie.JOUR, sortie.MOIS, sortie.ANNEE)
        ]
        sortie.loc[faux, ["JOUR", "MOIS"]] = ""
        sortie["DEPARTEMENT"] = large["DEPARTEMENT"].map(departement)
        sortie["COMMUNE"] = large["COMMUNE"].map(commune)
        sortie["PAYS"] = [
            pays(p, dep) for p, dep in zip(large["PAYS"], sortie.DEPARTEMENT)
        ]
    elif sujet.nom == "adresse_travail":
        sortie = large.copy()
        sortie["NUMERO"] = large["NUMERO"].map(numero_voie)
        sortie["INDICE"] = large["INDICE"].map(indice)
        sortie["DEPARTEMENT"] = large["DEPARTEMENT"].map(departement)
        sortie["COMMUNE"] = large["COMMUNE"].map(commune)
        sortie["PAYS"] = [
            pays(p, dep) for p, dep in zip(large["PAYS"], sortie.DEPARTEMENT)
        ]
    else:
        return d
    formate = sortie.stack().rename("lu_fiqual").reset_index()
    formate.columns = ["cle", "champ", "lu_fiqual"]
    d = d.merge(formate, on=["cle", "champ"], how="left")
    d["lu"] = d["lu_fiqual"].fillna(d["lu"])
    return d.drop(columns="lu_fiqual")


def colonne_departement_par_contenu(
    fiqual: dict[str, list[str]],
    pres_de: int,
    rayon: int = 4,
    exclues: set[int] | frozenset[int] = frozenset(),
) -> int | None:
    """La colonne voisine de « pres_de » (la commune) dont les valeurs remplies sont des
    codes de departement (au moins 90 %), la plus proche en cas d'egalite. « exclues » :
    colonnes deja attribuees (jour, mois : leurs valeurs 01 a 12 ressemblent aussi a des
    departements). A VERIFIER : le dire en l'utilisant."""
    lignes = list(fiqual.values())
    meilleure, score = None, 0.0
    voisines = sorted(
        range(max(0, pres_de - rayon), pres_de + rayon + 1),
        key=lambda i: abs(i - pres_de),
    )
    for i in voisines:
        if i == pres_de or i in exclues:
            continue
        vals = [c[i].strip() for c in lignes if len(c) > i and c[i].strip()]
        if len(vals) < 0.3 * len(lignes):
            continue
        part = sum(v.upper() in DEPARTEMENTS_ADMIS for v in vals) / len(vals)
        if part >= 0.9 and part > score:
            meilleure, score = i, part
    return meilleure
