"""Faire lire toutes les decoupes d'un sujet par un lecteur, sans jamais repayer.

- parallele (n appels a la fois) ;
- cache : chaque lecture reussie est ecrite tout de suite (un CSV par sujet, lecteur et
  consigne). Relancer reprend la ou on s'est arrete ; rien n'est relu ni repaye ;
- garde-fous : plafond de depense (cout renvoye par Eden AI), duree maximale, arret si
  les premiers appels echouent tous (nom de modele faux, cle, service indisponible).
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import pandas as pd

from ..decoupage import stockage
from .modeles import Lecteur, Reponse, lire

journal = logging.getLogger(__name__)

COLONNES_FIXES = ["cle", "secondes", "jetons", "cout", "version", "json_ok"]


def nom_fichier(texte: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", texte.lower()).strip("_")


def chemin_cache(
    dossier: str, campagne: str, sujet: str, lecteur: Lecteur, version: str
) -> str:
    nom = f"lectures_{campagne}_{sujet}_{nom_fichier(lecteur.fournisseur + '_' + lecteur.modele)}_{version}.csv"
    return stockage.joindre(dossier, nom)


def lire_cache(chemin: str, champs: list[str]) -> pd.DataFrame:
    colonnes = COLONNES_FIXES + [f"lu_{c}" for c in champs]
    if not stockage.existe(chemin):
        return pd.DataFrame(columns=colonnes)
    d = pd.read_csv(io.BytesIO(stockage.lire_octets(chemin)), dtype={"cle": str})
    for c in colonnes:
        if c not in d:
            d[c] = None
    for c in champs:
        d[f"lu_{c}"] = d[f"lu_{c}"].fillna("").astype(str)
    return d[colonnes].drop_duplicates("cle", keep="last")


@dataclass
class Compteurs:
    """Partages entre les lecteurs d'une session : cout cumule, parametres refuses."""

    cout: float = 0.0
    adaptations: dict[str, list[str]] = field(default_factory=dict)
    verrou: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class Bilan:
    lecteur: str
    version: str
    deja: int
    lus: int
    echecs: int
    arret: str
    premiere_erreur: str
    secondes_par_appel: float | None


def lire_tout(
    lecteur: Lecteur,
    version: str,
    consigne: str,
    champs: list[str],
    images: dict[str, Callable[[], bytes]],
    cache: str,
    compteurs: Compteurs,
    paralleles: int = 4,
    plafond_dollars: float = 50.0,
    duree_max_minutes: float = 60.0,
    max_jetons: int = 2000,
    delai: float = 180,
    dire: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, Bilan]:
    """Lit toutes les images (cle -> fonction qui renvoie les octets). Renvoie le cache
    restreint a ces cles (une ligne par image) et un bilan."""
    deja = lire_cache(cache, champs)
    faits = set(deja.cle)
    a_faire = [k for k in images if k not in faits]
    nom = f"{lecteur.nom} · {version}"
    if not a_faire:
        dire(f"{nom} : {len(images)} deja lus, rien a faire")
        return deja[deja.cle.isin(images)], Bilan(
            nom, version, len(images), 0, 0, "", "", None
        )
    if lecteur.externe and compteurs.cout > plafond_dollars:
        dire(f"{nom} : plafond atteint ({compteurs.cout:.2f} $), lecture non lancee")
        return deja[deja.cle.isin(images)], Bilan(
            nom, version, len(faits & set(images)), 0, 0, "plafond", "", None
        )
    dire(f"{nom} : {len(images) - len(a_faire)} deja lus, {len(a_faire)} a lire")

    stop = threading.Event()
    lignes, echecs, temps = [], [], []
    arret, k = "", 0
    t_debut = dernier = time.time()

    def une(k_: str):
        if stop.is_set():
            return k_, None
        try:
            return k_, lire(
                lecteur,
                images[k_](),
                consigne,
                champs,
                max_jetons=max_jetons,
                delai=delai,
                adaptations=compteurs.adaptations,
            )
        except Exception as e:  # noqa: BLE001 (image abimee, stockage : un echec, pas un arret)
            return k_, Reponse(
                {c: "" for c in champs}, False, erreur=f"{type(e).__name__}: {e}"[:200]
            )

    def sauver():
        nonlocal deja, lignes
        if lignes:
            deja = pd.concat([deja, pd.DataFrame(lignes)], ignore_index=True)
            stockage.ecrire_texte(cache, deja.to_csv(index=False))
            lignes = []

    ex = ThreadPoolExecutor(paralleles)
    en_cours = {ex.submit(une, k_) for k_ in a_faire}
    try:
        while en_cours:
            finis, en_cours = wait(en_cours, timeout=30, return_when=FIRST_COMPLETED)
            for f in finis:
                k_, r = f.result()
                if r is None:
                    continue
                k += 1
                if r.ok:
                    lignes.append(
                        {
                            "cle": k_,
                            "secondes": round(r.secondes, 2),
                            "jetons": r.jetons,
                            "cout": r.cout,
                            "version": r.version,
                            "json_ok": r.json_ok,
                            **{f"lu_{c}": r.champs.get(c, "") for c in champs},
                        }
                    )
                    temps.append(r.secondes)
                    if isinstance(r.cout, (int, float)):
                        with compteurs.verrou:
                            compteurs.cout += r.cout
                else:
                    echecs.append(r.erreur)
            if not arret:
                if lecteur.externe and compteurs.cout > plafond_dollars:
                    arret = f"plafond de depense atteint ({compteurs.cout:.2f} $)"
                elif k >= 5 and len(echecs) == k:
                    arret = "les 5 premiers appels ont echoue (nom du modele, cle ou service)"
                elif k >= 20 and len(echecs) > k / 2:
                    arret = f"plus de la moitie des appels echouent ({len(echecs)}/{k})"
                elif (time.time() - t_debut) / 60 > duree_max_minutes:
                    arret = f"duree maximale atteinte ({duree_max_minutes:.0f} min)"
                if arret and en_cours:
                    stop.set()
                    for f in en_cours:
                        f.cancel()
                    en_cours = {f for f in en_cours if not f.cancelled()}
                    dire(f"  ARRET de {nom} : {arret}. Ce qui est lu est garde.")
            if len(lignes) >= 10:
                sauver()
            if time.time() - dernier >= 30 or not en_cours:
                dernier = time.time()
                vitesse = (
                    f"{sum(temps[-20:]) / len(temps[-20:]):.1f} s/appel"
                    if temps
                    else "-"
                )
                dire(
                    f"  {nom} : {k}/{len(a_faire)} | {(time.time() - t_debut) / 60:.1f} min | "
                    f"{vitesse} | echecs {len(echecs)} | cout cumule {compteurs.cout:.2f} $"
                )
    finally:
        sauver()
        stop.set()
        ex.shutdown(wait=False, cancel_futures=True)
    lus = deja[deja.cle.isin(images)]
    bilan = Bilan(
        lecteur=nom,
        version=version,
        deja=len(images) - len(a_faire),
        lus=k - len(echecs),
        echecs=len(echecs),
        arret=arret,
        premiere_erreur=echecs[0] if echecs else "",
        secondes_par_appel=round(sum(temps) / len(temps), 2) if temps else None,
    )
    if echecs:
        dire(
            f"  {len(echecs)} echecs (relancer pour les retenter). Premier : {echecs[0][:200]}"
        )
    return lus, bilan


def en_long(lus: pd.DataFrame, champs: list[str]) -> pd.DataFrame:
    """Une ligne par image -> une ligne par (image, champ) : cle, champ, lu."""
    return lus.melt(
        id_vars=["cle"],
        value_vars=[f"lu_{c}" for c in champs],
        var_name="champ",
        value_name="lu",
    ).assign(
        champ=lambda d: d.champ.str.removeprefix("lu_"), lu=lambda d: d.lu.fillna("")
    )
