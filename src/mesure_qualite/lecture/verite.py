"""La verite contre laquelle on juge les lectures : le Fiqual (saisie du PMQ) corrige par
les arbitrages du SeRN.

Pour chaque bulletin et chaque champ :
  PMQ    : la valeur du Fiqual ;
  PAD    : la valeur de l'autre saisie, s'il y a eu divergence ;
  SeRN   : la valeur attendue, si le SeRN a arbitre la divergence ;
  verite : SeRN si arbitree, PMQ s'il n'y a pas eu de divergence, inconnue sinon ;
  groupe : A (pas de divergence), B (divergence arbitree), C (non arbitree).

Ou est chaque champ dans le Fiqual ? On ne le suppose pas : pour chaque divergence, on
cherche dans quelle colonne de la ligne du Fiqual se trouve la VALEUR_PMQ de ce bulletin.
La colonne qui revient a chaque fois est celle de la variable (« votes »).
"""

from __future__ import annotations

import io
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..decoupage import stockage
from .comparaison import comparable, normaliser
from .sujets import Sujet

POS_UT, POS_CAB = 2, 4  # la cle d'une ligne du Fiqual : colonnes 2 (UT) et 4 (numero)
COLONNES_DIVERGENCES = [
    "VARIABLE",
    "NOM_IMAGE",
    "VALEUR_PAD",
    "VALEUR_PMQ",
    "VALEUR_ATTENDUE",
]


def _sans_zeros(v: str) -> str:
    v = str(v).strip()
    return (v.lstrip("0") or "0") if v.isdigit() else v


def cle(ut: str, numero: str) -> str:
    """La cle d'un bulletin, sans zeros devant (0407 et 407 sont le meme lot)."""
    return f"{_sans_zeros(ut)}_{_sans_zeros(numero)}"


def cle_image(chemin: str) -> str:
    """'.../0407_9761_5300000184_BI.jpg' -> '9761_5300000184' (les deux derniers numeros)."""
    m = Path(str(chemin)).stem.split("_")
    if m and not m[-1].isdigit():
        m = m[:-1]
    return cle(*m[-2:]) if len(m) >= 2 else str(chemin)


# --- lecture des fichiers ------------------------------------------------------------------
def _decoder(octets: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return octets.decode(enc)
        except UnicodeDecodeError:
            continue
    return octets.decode("latin-1", errors="replace")


def lire_fiqual(chemins: Iterable[str], type_ligne: str = "3") -> dict[str, list[str]]:
    """Les lignes d'un type (3 = bulletin individuel), par cle de bulletin."""
    lignes: dict[str, list[str]] = {}
    for chemin in chemins:
        for brut in _decoder(stockage.lire_octets(chemin)).splitlines():
            c = brut.split("|")
            if len(c) > POS_CAB and c[0].strip() == type_ligne:
                lignes.setdefault(cle(c[POS_UT], c[POS_CAB]), c)
    return lignes


def lire_divergences(chemins: Iterable[str]) -> pd.DataFrame:
    """Les fichiers Excel des divergences du SeRN, colonnes normalisees + cle + arbitree."""
    morceaux = [
        pd.read_excel(io.BytesIO(stockage.lire_octets(f)), dtype=str) for f in chemins
    ]
    if not morceaux:
        return pd.DataFrame(columns=[*COLONNES_DIVERGENCES, "cle", "arbitree"])
    d = pd.concat(morceaux, ignore_index=True).fillna("")
    d.columns = [re.sub(r"[\s\-]+", "_", c.strip().upper()) for c in d.columns]
    manque = [c for c in COLONNES_DIVERGENCES if c not in d.columns]
    if manque:
        raise ValueError(f"colonnes absentes du fichier des divergences : {manque}")
    d["VARIABLE"] = d.VARIABLE.str.strip().str.upper()
    d["cle"] = d.NOM_IMAGE.map(cle_image)
    d["arbitree"] = d.VALEUR_ATTENDUE.str.strip() != ""
    return d


# --- ou est chaque champ dans le Fiqual ---------------------------------------------------
def _egal(a: str, b: str) -> bool:
    a, b = normaliser(a), normaliser(b)
    return bool(a) and (a == b or (a.isdigit() and b.isdigit() and int(a) == int(b)))


def colonnes_par_votes(
    fiqual: dict[str, list[str]], divergences: pd.DataFrame, sujet: Sujet
) -> tuple[dict[str, int], pd.DataFrame]:
    """{champ: colonne} et le detail des votes (pour verifier).

    Une colonne est retenue si elle contient la VALEUR_PMQ dans au moins 60 % des
    divergences de la variable, sans ex aequo."""
    votes, presents = Counter(), Counter()
    variable_du_champ: dict[str, Counter] = {}
    for r in divergences.itertuples():
        champ = sujet.champ_de_variable(r.VARIABLE)
        ligne = fiqual.get(r.cle)
        if champ is None or ligne is None or not normaliser(r.VALEUR_PMQ):
            continue
        presents[champ] += 1
        variable_du_champ.setdefault(champ, Counter())[r.VARIABLE] += 1
        for i, x in enumerate(ligne):
            if i > POS_CAB and _egal(x, r.VALEUR_PMQ):
                votes[(champ, i)] += 1
    retenues, detail = {}, []
    for champ, sur in presents.items():
        classe = sorted(
            ((n, i) for (c, i), n in votes.items() if c == champ), reverse=True
        )
        if not classe:
            continue
        n, i = classe[0]
        ex_aequo = len(classe) > 1 and classe[1][0] == n
        ok = n >= max(1, 0.6 * sur) and not ex_aequo
        if ok:
            retenues[champ] = i
        detail.append(
            {
                "champ": champ,
                "variables": ", ".join(variable_du_champ[champ]),
                "divergences": sur,
                "colonne": i,
                "votes": n,
                "second": classe[1] if len(classe) > 1 else None,
                "retenue": ok,
            }
        )
    return retenues, pd.DataFrame(detail)


def colonnes_date_par_contenu(
    fiqual: dict[str, list[str]], annee_max: int
) -> dict[str, int]:
    """Repli pour la date de naissance si les divergences ne suffisent pas : trois colonnes
    voisines jour (1-31), mois (1-12), annee (4 chiffres). A VERIFIER (le dire)."""
    lignes = list(fiqual.values())
    n = max((len(c) for c in lignes), default=0)

    def part(i: int, test) -> float:
        vals = [c[i].strip() for c in lignes if len(c) > i and c[i].strip()]
        rempli = len(vals) / max(1, len(lignes))
        return (sum(map(test, vals)) / len(vals)) if vals and rempli >= 0.5 else 0.0

    def est_an(v: str) -> bool:
        return v.isdigit() and len(v) == 4 and 1900 <= int(v) <= annee_max

    def est_mois(v: str) -> bool:
        return v.isdigit() and 1 <= int(v) <= 12

    def est_jour(v: str) -> bool:
        return v.isdigit() and 1 <= int(v) <= 31

    for i in range(POS_CAB + 3, n):
        if part(i, est_an) >= 0.8:
            if part(i - 1, est_mois) >= 0.9 and part(i - 2, est_jour) >= 0.9:
                return {"JOUR": i - 2, "MOIS": i - 1, "ANNEE": i}
            if (
                i + 2 < n
                and part(i + 1, est_mois) >= 0.9
                and part(i + 2, est_jour) >= 0.9
            ):
                return {"ANNEE": i, "MOIS": i + 1, "JOUR": i + 2}
    return {}


# --- la table de verite -------------------------------------------------------------------
def table_verite(
    fiqual: dict[str, list[str]],
    divergences: pd.DataFrame,
    sujet: Sujet,
    colonnes: dict[str, int],
    cles: Iterable[str],
) -> pd.DataFrame:
    """Une ligne par (bulletin, champ) : PMQ, PAD, SeRN, verite (comparables), groupe."""
    div = {}
    for r in divergences.itertuples():
        champ = sujet.champ_de_variable(r.VARIABLE)
        if champ:
            div[(r.cle, champ)] = r
    lignes = []
    for k in cles:
        ligne = fiqual.get(k)
        if ligne is None:
            continue
        for champ, i in colonnes.items():
            t = sujet.champ(champ).type
            pmq = comparable(ligne[i] if len(ligne) > i else "", t)
            d = div.get((k, champ))
            if d is None:
                pad, sern, vrai, groupe = pmq, "", pmq, "A"
            elif d.arbitree:
                pad, sern = (
                    comparable(d.VALEUR_PAD, t),
                    comparable(d.VALEUR_ATTENDUE, t),
                )
                vrai, groupe = sern, "B"
            else:
                pad, sern, vrai, groupe = comparable(d.VALEUR_PAD, t), "", None, "C"
            lignes.append(
                {
                    "cle": k,
                    "champ": champ,
                    "PMQ": pmq,
                    "PAD": pad,
                    "SeRN": sern,
                    "verite": vrai,
                    "groupe": groupe,
                }
            )
    return pd.DataFrame(
        lignes, columns=["cle", "champ", "PMQ", "PAD", "SeRN", "verite", "groupe"]
    )


def forme(v: object) -> str:
    """La forme d'une valeur SANS la valeur : chiffres -> 9, lettres -> A."""
    return re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", normaliser(v)))[:18]
