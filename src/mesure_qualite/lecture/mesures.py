"""Juger et comparer les lecteurs, sur les memes lignes, avec la verite arbitree.

juger      : la table ligne par ligne (bulletin x champ) avec le verdict de chaque lecteur ;
resumer    : par lecteur et par champ (ou groupe de champs) : justes, IC a 95 %,
             avouees, silencieuses ;
comparer   : deux lecteurs sur les memes lignes, test exact de McNemar ;
barre_pmq  : la fourchette du taux d'erreur du PMQ, a laquelle on compare chaque lecteur.
"""

from __future__ import annotations

import math

import pandas as pd
from scipy import stats

from .comparaison import comparable, verdict, verdict_groupe
from .sujets import Sujet


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance a 95 % d'une proportion k / n (Wilson)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - m), min(1.0, c + m)


def mcnemar(a_ok: pd.Series, b_ok: pd.Series) -> tuple[int, int, float]:
    """(lignes justes pour a seul, justes pour b seul, p exact bilateral)."""
    a_ok, b_ok = a_ok.astype(bool), b_ok.astype(bool)
    n10, n01 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = stats.binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def juger(
    verite: pd.DataFrame, lectures: dict[str, pd.DataFrame], sujet: Sujet
) -> pd.DataFrame:
    """verite : table_verite(...) ; lectures : nom du lecteur -> (cle, champ, lu).

    Seules les lignes lues par TOUS les lecteurs sont gardees : chacun est juge sur les
    memes lignes. Ajoute aussi les lignes des groupes de champs (ex. DATE)."""
    t = verite.copy()
    t["verdict PMQ"] = [verdict(p, v) for p, v in zip(t.PMQ, t.verite)]
    for nom, lus in lectures.items():
        types = {c.nom: c.type for c in sujet.champs}
        lu = {
            (k, c): comparable(v, types[c])
            for k, c, v in zip(lus.cle, lus.champ, lus.lu)
            if c in types
        }
        t[f"lu {nom}"] = [lu.get((k, c)) for k, c in zip(t.cle, t.champ)]
        t[f"verdict {nom}"] = [
            verdict(x, v) if x is not None else None
            for x, v in zip(t[f"lu {nom}"], t.verite)
        ]
    if lectures:
        lus_partout = t[[f"lu {n}" for n in lectures]].notna().all(axis=1)
        t = t[lus_partout].reset_index(drop=True)
    for groupe, membres in sujet.groupes.items():
        sous = t[t.champ.isin(membres)]
        complets = sous.groupby("cle").champ.nunique() == len(membres)
        sous = sous[sous.cle.isin(complets[complets].index)]
        lignes = []
        for k, d in sous.groupby("cle", sort=False):
            ligne = {"cle": k, "champ": groupe, "groupe": max(d.groupe)}
            ligne["verdict PMQ"] = verdict_groupe(list(d["verdict PMQ"]))
            for nom in lectures:
                ligne[f"verdict {nom}"] = verdict_groupe(list(d[f"verdict {nom}"]))
            lignes.append(ligne)
        if lignes:
            t = pd.concat([t, pd.DataFrame(lignes)], ignore_index=True)
    return t


def _resume(v: pd.Series) -> dict:
    v = v[v != "ecartee"].dropna()
    n, k = len(v), int((v == "juste").sum())
    lo, hi = wilson(k, n)

    def pc(x: int) -> float:
        return round(x / n * 100, 1) if n else float("nan")

    return {
        "lignes": n,
        "justes": k,
        "justes_%": pc(k),
        "IC95": f"[{lo * 100:.1f} ; {hi * 100:.1f}]",
        "avouees_%": pc(int((v == "avouee").sum())),
        "silencieuses": int((v == "silencieuse").sum()),
        "silencieuses_%": pc(int((v == "silencieuse").sum())),
    }


def resumer(
    t: pd.DataFrame, lecteurs: list[str], champs: list[str] | None = None
) -> pd.DataFrame:
    """Une ligne par (champ, lecteur), plus le PMQ, contre la verite arbitree."""
    champs = champs or list(dict.fromkeys(t.champ))
    lignes = []
    for champ in champs:
        s = t[t.champ == champ]
        for nom in ["PMQ", *lecteurs]:
            lignes.append(
                {"champ": champ, "lecteur": nom, **_resume(s[f"verdict {nom}"])}
            )
    return pd.DataFrame(lignes)


def comparer(
    t: pd.DataFrame, a: str, b: str, champs: list[str] | None = None
) -> pd.DataFrame:
    """a contre b, champ par champ, sur les memes lignes (verite connue)."""
    lignes = []
    for champ in champs or list(dict.fromkeys(t.champ)):
        s = t[(t.champ == champ) & (t[f"verdict {a}"] != "ecartee")]
        n10, n01, p = mcnemar(
            s[f"verdict {a}"] == "juste", s[f"verdict {b}"] == "juste"
        )
        verdict_ = (
            f"{a} meilleur"
            if p < 0.05 and n10 > n01
            else f"{b} meilleur"
            if p < 0.05 and n01 > n10
            else "pas d'ecart etabli"
        )
        lignes.append(
            {
                "champ": champ,
                "lignes": len(s),
                f"{a} seul juste": n10,
                f"{b} seul juste": n01,
                "p": round(p, 4),
                "conclusion": verdict_,
            }
        )
    return pd.DataFrame(lignes)


def barre_pmq(t: pd.DataFrame, champ: str) -> tuple[float, float]:
    """Taux d'erreur du PMQ : au minimum (erreurs prouvees par le SeRN), au maximum (si le
    PMQ avait tort sur toutes les divergences non arbitrees)."""
    s = t[t.champ == champ]
    n, non_arb = len(s), int((s.groupe == "C").sum())
    err = int((s["verdict PMQ"] == "silencieuse").sum())
    return err / max(1, n - non_arb), (err + non_arb) / max(1, n)
