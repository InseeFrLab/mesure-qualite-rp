"""Decoupage d'une campagne : tous les scans d'un formulaire, bloc par bloc.

Entree  : les scans de la campagne (ex. s3://.../data/raw/RG25/**/*_BI.jpg) et le
          gabarit du formulaire (ex. BI_2025).
Sortie  : une image par scan et par bloc, <sortie>/<bloc>/<scan>.jpg, et un INDEX
          <sortie>/index_<formulaire>_<campagne>.csv : pour chaque scan et chaque bloc,
          le chemin de la decoupe, la methode (recalage, secours, proportions) et la
          qualite du recalage. Les etapes suivantes du pipeline partent de l'index.

Reprise : un scan deja present dans l'index (avec le meme gabarit) n'est pas refait.
Tous les chemins et reglages viennent de configs/campagnes.yaml : rien n'est ecrit en
dur dans le code.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import stockage
from .gabarit import blocs_a_decouper, selectionner
from .recalage import Recaleur

journal = logging.getLogger(__name__)

EXTENSIONS_SCANS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
COLONNES_INDEX = [
    "scan",
    "gabarit",
    "bloc",
    "question",
    "theme",
    "variables",
    "decoupe",
    "methode",
    "points_recalage",
    "angle",
    "octets",
]

# Secours pour une unite non recalee : (image du scan, bornes des unites, bloc) -> decoupe
# ou None. Exemple : un modele de detection. Le nom apparait dans la colonne « methode ».
Secours = Callable[[np.ndarray, list[tuple[int, int]], dict], "np.ndarray | None"]


# --- configuration ---------------------------------------------------------------------
def racine_depot() -> Path:
    """Racine du depot (src/mesure_qualite/decoupage/campagne.py -> 3 niveaux plus haut)."""
    return Path(__file__).resolve().parents[3]


def lire_configuration(fichier: str | Path | None = None) -> dict:
    fichier = (
        Path(fichier) if fichier else racine_depot() / "configs" / "campagnes.yaml"
    )
    return yaml.safe_load(fichier.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ParametresDecoupage:
    """Tout ce qu'il faut pour decouper un formulaire d'une campagne."""

    campagne: str
    formulaire: str
    gabarit: str
    scans: str
    suffixe: str
    gabarits: str
    sortie: str
    format: str = "jpg"
    blocs: str | tuple[str, ...] = "toutes"

    @property
    def index(self) -> str:
        return stockage.joindre(
            self.sortie, f"index_{self.formulaire}_{self.campagne}.csv"
        )


def s3(bucket: str, chemin: str) -> str:
    return f"{stockage.PREFIXE_S3}{bucket}/{chemin.strip('/')}"


def racine_donnees(configuration: dict | None = None, racine: str | None = None) -> str:
    """La racine des donnees : le bucket du projet (s3://...), ou un dossier local."""
    c = configuration or lire_configuration()
    return racine or s3(c["bucket"], "")


def parametres(
    campagne: str,
    formulaire: str,
    configuration: dict | None = None,
    racine: str | None = None,
) -> ParametresDecoupage:
    """Lit configs/campagnes.yaml et renvoie les parametres de decoupage.

    racine : None = le bucket du projet ; sinon un dossier local de meme organisation
    (essais hors du SSP Cloud, tests)."""
    c = configuration or lire_configuration()
    general = c.get("decoupage") or {}
    try:
        camp = c["campagnes"][campagne]
        form = camp["decoupage"][formulaire]
    except KeyError as e:
        raise KeyError(
            f"Pas de decoupage « {formulaire} » pour la campagne « {campagne} » "
            "dans configs/campagnes.yaml (campagnes.<campagne>.decoupage.<formulaire>)."
        ) from e
    base = racine_donnees(c, racine)

    def chemin(relatif: str) -> str:
        return stockage.joindre(base, relatif.strip("/"))

    return ParametresDecoupage(
        campagne=campagne,
        formulaire=formulaire,
        gabarit=form["gabarit"],
        scans=chemin(form.get("scans", camp["racine"])),
        suffixe=form.get("suffixe", formulaire),
        gabarits=chemin(general.get("gabarits", "data/decoupes/gabarits")),
        sortie=chemin(
            f"{general.get('decoupes', 'data/decoupes')}/{campagne}/{formulaire}"
        ),
        format=general.get("format", "jpg"),
    )


# --- un scan ---------------------------------------------------------------------------
@dataclass
class Decoupe:
    """Une decoupe : l'image d'un bloc, la methode qui l'a produite, le recalage."""

    bloc: dict
    image: np.ndarray
    methode: str
    recalage: dict = field(default_factory=dict)


def decouper_scan(
    img: np.ndarray,
    recaleur: Recaleur,
    blocs: Iterable[dict],
    secours: tuple[str, Secours] | None = None,
) -> list[Decoupe]:
    """Recale un scan et decoupe chaque bloc demande.

    Methode : « recalage » si l'unite est recalee ; sinon « secours:<nom> » si un secours
    est fourni et trouve le bloc ; sinon « proportions » (unite prise aux proportions de
    la reference, signalee)."""
    toile, infos = recaleur.toile(img)
    bornes = (
        recaleur.bornes_unites(img) if any(not i["recalee"] for i in infos) else None
    )
    sortie = []
    for b in blocs:
        inf = infos[b["unite"] - 1]
        crop, methode = None, "recalage" if inf["recalee"] else "proportions"
        if not inf["recalee"] and secours is not None:
            nom, fonction = secours
            crop = fonction(img, bornes, b)
            if crop is not None and crop.size:
                methode = f"secours:{nom}"
        if crop is None or not crop.size:
            crop = recaleur.decouper(toile, b["zone"], b.get("marge"))
        sortie.append(Decoupe(b, crop, methode, inf))
    return sortie


def decouper_selection(img: np.ndarray, recaleur: Recaleur, **criteres) -> np.ndarray:
    """Recale un scan et decoupe une selection (voir gabarit.selectionner), par exemple
    decouper_selection(img, r, variables=["NUMVOI", "NOMVOI"], avec_intitule=True)."""
    sel = selectionner(recaleur.cfg, **criteres)
    toile, _ = recaleur.toile(img)
    return recaleur.decouper(toile, sel["zone"], sel["marge"])


# --- une campagne ----------------------------------------------------------------------
def scans_du_formulaire(p: ParametresDecoupage) -> list[str]:
    """Les scans dont le nom se termine par _<suffixe> (ex. 6901_5300000403_BI.jpg)."""
    fin = "_" + p.suffixe.upper()
    return [
        f
        for f in stockage.lister(p.scans, EXTENSIONS_SCANS)
        if Path(f).stem.upper().endswith(fin)
    ]


def lire_index(p: ParametresDecoupage) -> pd.DataFrame:
    if not stockage.existe(p.index):
        return pd.DataFrame(columns=COLONNES_INDEX)
    return pd.read_csv(io.BytesIO(stockage.lire_octets(p.index)))


def decouper_campagne(
    p: ParametresDecoupage,
    limite: int | None = None,
    refaire: bool = False,
    paralleles: int = 8,
    duree_max_minutes: float | None = None,
    secours: tuple[str, Secours] | None = None,
    sauvegarde_tous: int = 50,
    scans: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Decoupe tous les scans d'un formulaire et met l'index a jour. Renvoie l'index.

    limite            : ne traiter que les N premiers scans (essai) ;
    refaire           : ignorer l'index existant (apres un nouveau gabarit) ;
    duree_max_minutes : arret propre au-dela ; relancer reprend la ou on s'est arrete ;
    scans             : ne traiter que ces scans (ex. les bulletins qui ont une verite).

    Reprise bloc par bloc : un scan est refait seulement s'il lui manque un des blocs
    demandes (p.blocs). Decouper d'abord q2, puis tout le bulletin, ne perd rien."""
    recaleur = Recaleur.depuis_dossier(p.gabarits, p.gabarit)
    blocs = blocs_a_decouper(
        recaleur.cfg, p.blocs if isinstance(p.blocs, str) else list(p.blocs)
    )
    deja = pd.DataFrame(columns=COLONNES_INDEX) if refaire else lire_index(p)
    voulus = {b["id"] for b in blocs}
    blocs_faits: dict[str, set] = {}
    if len(deja):
        for s_, b_ in deja.loc[deja.gabarit == p.gabarit, ["scan", "bloc"]].itertuples(
            index=False
        ):
            blocs_faits.setdefault(s_, set()).add(b_)
    faits = {s_ for s_, bs in blocs_faits.items() if voulus <= bs}
    tous = scans_du_formulaire(p) if scans is None else sorted(set(scans))
    a_faire = [s for s in (tous[:limite] if limite else tous) if s not in faits]
    journal.info(
        "%s %s (%s) : %d blocs par scan, %d scans, %d deja faits, %d a faire",
        p.campagne,
        p.formulaire,
        p.gabarit,
        len(blocs),
        len(tous),
        len(faits),
        len(a_faire),
    )

    lignes: list[dict] = []
    verrou = threading.Lock()
    arret = threading.Event()

    def un_scan(chemin: str) -> list[dict]:
        if arret.is_set():
            return []
        out = []
        for d in decouper_scan(stockage.lire_image(chemin), recaleur, blocs, secours):
            dest = stockage.joindre(
                p.sortie, d.bloc["id"], f"{Path(chemin).stem}.{p.format}"
            )
            out.append(
                {
                    "scan": chemin,
                    "gabarit": p.gabarit,
                    "bloc": d.bloc["id"],
                    "question": d.bloc["question"],
                    "theme": d.bloc.get("theme"),
                    "variables": " ".join(d.bloc["variables_formulaire"]),
                    "decoupe": dest,
                    "methode": d.methode,
                    "points_recalage": d.recalage.get("retenus"),
                    "angle": d.recalage.get("angle"),
                    "octets": stockage.ecrire_image(dest, d.image, p.format),
                }
            )
        return out

    def sauver() -> pd.DataFrame:
        with verrou:
            nouveaux = pd.DataFrame(lignes, columns=COLONNES_INDEX)
            refaits = set(zip(nouveaux.scan, nouveaux.bloc))
            anciens = (
                deja[[(s_, b_) not in refaits for s_, b_ in zip(deja.scan, deja.bloc)]]
                if len(deja)
                else deja
            )
            index = (
                pd.concat([anciens, nouveaux], ignore_index=True)
                if len(anciens)
                else nouveaux
            )
        stockage.ecrire_texte(p.index, index.to_csv(index=False))
        return index

    debut = time.time()
    index = deja
    with ThreadPoolExecutor(paralleles) as ex:
        futurs = [ex.submit(un_scan, s) for s in a_faire]
        try:
            for k, fu in enumerate(as_completed(futurs), 1):
                with verrou:
                    lignes.extend(fu.result())
                if k % sauvegarde_tous == 0 or k == len(futurs):
                    ecoule = time.time() - debut
                    journal.info(
                        "%d/%d scans en %.1f min (reste ~%.0f min)",
                        k,
                        len(futurs),
                        ecoule / 60,
                        ecoule / k * (len(futurs) - k) / 60,
                    )
                    index = sauver()
                if duree_max_minutes and time.time() - debut > duree_max_minutes * 60:
                    journal.warning(
                        "duree maximale atteinte apres %d scans : relancer pour continuer",
                        k,
                    )
                    break
        finally:
            arret.set()
            for fu in futurs:
                fu.cancel()
    if lignes:
        index = sauver()
    return index


def resume_index(index: pd.DataFrame) -> dict:
    """Chiffres de controle d'un index : scans, decoupes, part de chaque methode (%)."""
    if not len(index):
        return {"scans": 0, "decoupes": 0}
    u = index.drop_duplicates(["scan", "bloc"])
    parts = u.methode.value_counts(normalize=True).mul(100).round(1).to_dict()
    return {
        "scans": int(u.scan.nunique()),
        "blocs_par_scan": int(u.bloc.nunique()),
        "decoupes": len(u),
        "methode_pct": parts,
    }


def formulaire_et_millesime(nom_pdf: str) -> tuple[str | None, int | None]:
    """Deduit le formulaire et le millesime du nom d'un PDF d'imprime.
    « 2025-imprime-3-BI avec VAR.pdf » -> ("BI", 2025)."""
    an = re.search(r"(20\d\d)", nom_pdf)
    form = re.search(r"imprime-\w+-([A-Za-z]+)", nom_pdf)
    return (form.group(1).upper() if form else None, int(an.group(1)) if an else None)
