"""Une etude complete en trois appels, pour les notebooks.

    d  = preparer("naissance", "RG25")            # Fiqual, verite, decoupes (etape 1)
    lu = lire_tous(d, lecteurs, ["V0", "V1"], ...)  # essai, puis lectures (cache)
    ev = evaluer(d, lu.lectures)                  # format Fiqual, verdicts, tableaux

Chaque fonction ne fait qu'enchainer les modules du paquet : verite, decoupage,
campagne (lectures), saisie (consignes du PMQ), mesures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from ..decoupage import stockage
from ..decoupage.campagne import (
    decouper_campagne,
    parametres,
    racine_donnees,
    scans_du_formulaire,
)
from ..decoupage.campagne import lire_configuration as config_campagnes
from . import saisie
from .campagne import Compteurs, chemin_cache, en_long, lire_tout
from .consignes import consigne
from .mesures import comparer, juger, resumer
from .modeles import Lecteur
from .sujets import Sujet, lire_configuration, sujet
from .verite import (
    cle_image,
    colonnes_date_par_contenu,
    colonnes_par_votes,
    lire_divergences,
    lire_fiqual,
    table_verite,
)


# --- 1. preparer : la verite et les decoupes ----------------------------------------------
@dataclass
class Donnees:
    sujet: Sujet
    campagne: str
    collecte: int
    fiqual: dict[str, list[str]]
    divergences: pd.DataFrame
    colonnes: dict[str, int]
    sources: dict[str, str]
    votes: pd.DataFrame
    index: pd.DataFrame
    images: dict[str, Callable[[], bytes]]
    verite: pd.DataFrame
    gabarit: str


def annee_collecte(campagne: str) -> int:
    """RG25 -> 2025."""
    return 2000 + int("".join(c for c in campagne if c.isdigit())[:2] or 25)


def preparer(
    nom_sujet: str,
    campagne: str,
    racine: str | None = None,
    colonnes: dict[str, int] | None = None,
    limite: int | None = None,
    paralleles: int = 8,
    dire: Callable[[str], None] = print,
) -> Donnees:
    """Fiqual, divergences, colonnes du Fiqual, decoupes du bloc (etape 1), verite."""
    s = sujet(nom_sujet)
    cfg = config_campagnes()
    camp = cfg["campagnes"][campagne]
    dossier = stockage.joindre(racine_donnees(cfg, racine), camp["racine"])
    nom_fiqual = camp.get("fiqual")
    f_fiqual = (
        [stockage.joindre(dossier, nom_fiqual)]
        if nom_fiqual
        else [
            f
            for f in stockage.lister(dossier, (".txt",))
            if "FIQUAL" in Path(f).name.upper()
        ]
    )
    if not f_fiqual or not all(stockage.existe(f) for f in f_fiqual):
        raise FileNotFoundError(f"Fiqual introuvable sous {dossier}")
    f_div = stockage.lister(
        stockage.joindre(dossier, camp.get("divergences", "Divergences")),
        (".xlsx", ".xls"),
    )
    fiqual = lire_fiqual(f_fiqual, s.type_ligne_fiqual)
    div = lire_divergences(f_div)

    # ou est chaque champ dans le Fiqual
    cols, votes = colonnes_par_votes(fiqual, div, s)
    sources = {c: "divergences (votes)" for c in cols}
    collecte = annee_collecte(campagne)
    if s.nom == "naissance" and not {"JOUR", "MOIS", "ANNEE"} <= set(cols):
        for c, i in colonnes_date_par_contenu(fiqual, collecte).items():
            if c not in cols:
                cols[c], sources[c] = i, "contenu des colonnes (A VERIFIER)"
    if (
        "DEPARTEMENT" in s.noms_champs
        and "DEPARTEMENT" not in cols
        and "COMMUNE" in cols
    ):
        i = saisie.colonne_departement_par_contenu(
            fiqual, cols["COMMUNE"], exclues=set(cols.values())
        )
        if i is not None:
            cols["DEPARTEMENT"], sources["DEPARTEMENT"] = (
                i,
                "contenu des colonnes (A VERIFIER)",
            )
    for c, i in (colonnes or {}).items():
        cols[c], sources[c] = int(i), "reglage (colonnes)"
    cols = {c: cols[c] for c in s.noms_champs if c in cols}
    if not cols:
        raise RuntimeError(
            f"{s.titre} : aucun champ trouve dans le Fiqual ; donner les colonnes a la main."
        )
    manquants = [c for c in s.noms_champs if c not in cols]
    dire(
        f"{s.titre} : Fiqual {len(fiqual)} bulletins, {len(div)} divergences | colonnes "
        f"{cols}"
        + (f" | non trouves (lus, pas juges) : {manquants}" if manquants else "")
    )

    # etape 1 : decoupe du bloc sur les bulletins qui ont une ligne du Fiqual
    p = replace(parametres(campagne, s.formulaire, cfg, racine=racine), blocs=(s.bloc,))
    if not stockage.existe(stockage.joindre(p.gabarits, f"{p.gabarit}.json")):
        raise FileNotFoundError(
            f"Gabarit {p.gabarit} absent de {p.gabarits} : le construire d'abord "
            "(python -m mesure_qualite.decoupage construire --pdf ...)"
        )
    scans = {cle_image(x): x for x in scans_du_formulaire(p)}
    cles = sorted(k for k in scans if k in fiqual)
    cles = cles[:limite] if limite else cles
    index = decouper_campagne(p, scans=[scans[k] for k in cles], paralleles=paralleles)
    index = index[
        (index.bloc == s.bloc) & index.scan.isin([scans[k] for k in cles])
    ].copy()
    index["cle"] = index.scan.map(cle_image)
    decoupes = dict(zip(index.cle, index.decoupe))
    images = {
        k: (lambda c=decoupes[k]: stockage.lire_octets(c))
        for k in cles
        if k in decoupes
    }
    non_recales = int((index.methode != "recalage").sum())
    dire(
        f"  {len(scans)} scans {s.formulaire}, {len(images)} decoupes du bloc {s.bloc}"
        + (f", dont {non_recales} non recales" if non_recales else ", tous recales")
    )
    verite = table_verite(fiqual, div, s, cols, list(images))
    return Donnees(
        s,
        campagne,
        collecte,
        fiqual,
        div,
        cols,
        sources,
        votes,
        index,
        images,
        verite,
        p.gabarit,
    )


# --- 2. lire : essai puis lectures completes ------------------------------------------------
@dataclass
class Lectures:
    lectures: dict[str, tuple[Lecteur, str, pd.DataFrame]]
    essai: pd.DataFrame
    bilans: pd.DataFrame
    retenus: list[Lecteur]
    textes: dict[str, str] = field(default_factory=dict)


def lire_tous(
    d: Donnees,
    lecteurs: list[Lecteur],
    versions: list[str],
    dossier_cache: str | Path,
    compteurs: Compteurs,
    plafond_dollars: float = 30.0,
    duree_max_minutes: float = 60.0,
    max_jetons: int = 2000,
    n_essai: int = 3,
    dire: Callable[[str], None] = print,
) -> Lectures:
    """Chaque lecteur lit d'abord n_essai bulletins (consigne la plus longue) ; ceux qui
    repondent au bon format lisent ensuite tout, avec chaque consigne. Cache : rien n'est
    relu ni repaye."""
    textes = {v: consigne(d.sujet, v) for v in versions}
    conf = lire_configuration()

    def lire(
        lecteur: Lecteur, version: str, images: dict
    ) -> tuple[pd.DataFrame, object]:
        cache = chemin_cache(
            str(dossier_cache), d.campagne, d.sujet.nom, lecteur, version
        )
        par = conf["fournisseurs"][lecteur.fournisseur].get("paralleles", 4)
        return lire_tout(
            lecteur,
            version,
            textes[version],
            d.sujet.noms_champs,
            images,
            cache,
            compteurs,
            paralleles=par,
            plafond_dollars=plafond_dollars,
            duree_max_minutes=duree_max_minutes,
            max_jetons=max_jetons,
            dire=dire,
        )

    essai_images = {k: d.images[k] for k in list(d.images)[:n_essai]}
    essai, retenus = [], []
    for lecteur in lecteurs:
        lus, bilan = lire(lecteur, versions[-1], essai_images)
        json_ok = pd.to_numeric(lus.json_ok, errors="coerce").fillna(0).astype(bool)
        ok = len(lus) == len(essai_images) and bool(json_ok.all())
        cout = pd.to_numeric(lus.cout, errors="coerce").mean()
        essai.append(
            {
                "lecteur": lecteur.nom,
                "modele": lecteur.modele,
                "lus": len(lus),
                "json_ok": int(json_ok.sum()),
                "s_par_appel": round(
                    pd.to_numeric(lus.secondes, errors="coerce").mean(), 1
                ),
                "version_servie": lus.version.dropna().iloc[-1] if len(lus) else "",
                "cout_estime_tout_$": round(cout * len(d.images) * len(versions), 2)
                if pd.notna(cout)
                else None,
                "retenu": ok,
                "erreur": bilan.premiere_erreur[:150],
            }
        )
        if ok:
            retenus.append(lecteur)
    lectures, bilans = {}, []
    for lecteur in retenus:
        for v in versions:
            lus, bilan = lire(lecteur, v, d.images)
            lectures[f"{lecteur.nom} · {v}"] = (lecteur, v, lus)
            bilans.append(bilan.__dict__)
    return Lectures(
        lectures, pd.DataFrame(essai), pd.DataFrame(bilans), retenus, textes
    )


# --- 3. evaluer : format Fiqual, verdicts, tableaux ------------------------------------------
@dataclass
class Evaluation:
    table: pd.DataFrame  # ligne par ligne : verite, lectures, verdicts
    synthese: pd.DataFrame  # par champ et par lecteur
    principal: str  # le champ (ou groupe) a citer, ex. DATE
    vue: pd.DataFrame  # la synthese du champ principal, triee
    meilleure: dict[str, str]  # modele -> son meilleur « modele · consigne »
    ecartes: list[str]


def evaluer(
    d: Donnees,
    lectures: dict[str, tuple[Lecteur, str, pd.DataFrame]],
    consignes_de_saisie: bool = True,
    couverture_min: float = 0.95,
) -> Evaluation:
    """Met les lectures au format du Fiqual (consignes de saisie du PMQ), les juge contre
    la verite arbitree sur les memes lignes, et resume."""
    longues = {}
    for nom, (_, _, lus) in lectures.items():
        if len(lus) < couverture_min * len(d.images):
            continue
        lg = en_long(lus, d.sujet.noms_champs)
        longues[nom] = (
            saisie.au_format_fiqual(lg, d.sujet, d.collecte)
            if consignes_de_saisie
            else lg
        )
    ecartes = sorted(set(lectures) - set(longues))
    t = juger(d.verite, longues, d.sujet)
    ordre = [*d.sujet.groupes, *[c for c in d.sujet.noms_champs if c in d.colonnes]]
    ordre = [c for c in ordre if c in set(t.champ)]
    synthese = resumer(t, list(longues), ordre)
    principal = next((g for g in d.sujet.groupes if g in set(t.champ)), ordre[0])
    vue = (
        synthese[synthese.champ == principal]
        .sort_values(["justes_%", "silencieuses_%"], ascending=[False, True])
        .reset_index(drop=True)
    )
    j = vue.set_index("lecteur")["justes_%"]
    meilleure: dict[str, str] = {}
    for nom, (lecteur, _, _) in lectures.items():
        if nom in j.index and (
            lecteur.nom not in meilleure or j[nom] > j[meilleure[lecteur.nom]]
        ):
            meilleure[lecteur.nom] = nom
    return Evaluation(t, synthese, principal, vue, meilleure, ecartes)


def duel(ev: Evaluation, a: str, b: str) -> dict:
    """a contre b sur le champ principal (McNemar), en une ligne lisible."""
    m = comparer(ev.table, a, b, [ev.principal]).iloc[0]
    n_a, n_b = int(m[f"{a} seul juste"]), int(m[f"{b} seul juste"])
    return {
        "a": a,
        "b": b,
        "a seul juste": n_a,
        "b seul juste": n_b,
        "p": float(m.p),
        "conclusion": (
            f"{a} meilleur"
            if m.p < 0.05 and n_a > n_b
            else f"{b} meilleur"
            if m.p < 0.05 and n_b > n_a
            else "pas d'ecart etabli"
        ),
    }
