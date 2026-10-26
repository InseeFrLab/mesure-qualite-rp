"""Ligne de commande de l'etape 1 (decoupage).

python -m mesure_qualite.decoupage construire --pdf "s3://.../2025-imprime-3-BI avec VAR.pdf"
python -m mesure_qualite.decoupage decouper --campagne RG25 --formulaire BI --limite 20
python -m mesure_qualite.decoupage comparer FLRM_2022 FLR_2027
python -m mesure_qualite.decoupage catalogue BI_2025 --sortie catalogue_BI_2025.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import stockage
from .campagne import (
    decouper_campagne,
    formulaire_et_millesime,
    lire_configuration,
    parametres,
    resume_index,
    s3,
)
from .construction import construire
from .gabarit import (
    ajouter_themes_par_regles,
    catalogue,
    charger,
    comparer_versions,
    enregistrer,
    image_controle_jpeg,
    nom_gabarit,
)

journal = logging.getLogger("mesure_qualite.decoupage")


def _dossier_gabarits(arg: str | None) -> str:
    if arg:
        return arg
    c = lire_configuration()
    return s3(
        c["bucket"],
        (c.get("decoupage") or {}).get("gabarits", "data/decoupes/gabarits"),
    )


def cmd_construire(a: argparse.Namespace) -> int:
    form, an = formulaire_et_millesime(Path(a.pdf).name)
    form, an = a.formulaire or form, a.millesime or an
    if not form or not an:
        journal.error(
            "formulaire ou millesime introuvable dans le nom du PDF : --formulaire et --millesime"
        )
        return 2
    cfg, refs = construire(
        a.pdf, form, an, dpi=a.dpi, ordre=a.ordre, source_nom=Path(a.pdf).name
    )
    ajouter_themes_par_regles(cfg)
    dossier = _dossier_gabarits(a.sortie)
    chemin = enregistrer(cfg, refs, dossier)
    stockage.ecrire_octets(
        stockage.joindre(dossier, f"{nom_gabarit(cfg)}_controle.jpg"),
        image_controle_jpeg(cfg, refs),
    )
    ctl = cfg["controle"]
    journal.info(
        "%s : %d questions trouvees, manquantes %s, %d blocs, %d champs %s",
        nom_gabarit(cfg),
        len(ctl["questions_trouvees"]),
        ctl["questions_manquantes"] or "aucune",
        ctl["n_blocs"],
        ctl["n_champs"],
        ctl["champs_par_type"],
    )
    journal.info("gabarit ecrit : %s", chemin)
    return 1 if ctl["questions_manquantes"] else 0


def cmd_decouper(a: argparse.Namespace) -> int:
    p = parametres(a.campagne, a.formulaire)
    if a.format or a.blocs:
        from dataclasses import replace

        p = replace(
            p, format=a.format or p.format, blocs=tuple(a.blocs) if a.blocs else p.blocs
        )
    index = decouper_campagne(
        p,
        limite=a.limite,
        refaire=a.refaire,
        paralleles=a.paralleles,
        duree_max_minutes=a.duree_max,
    )
    journal.info(
        "index %s : %s", p.index, json.dumps(resume_index(index), ensure_ascii=False)
    )
    return 0


def _charger_cfg(nom_ou_chemin: str, dossier: str) -> dict:
    if nom_ou_chemin.endswith(".json"):
        return json.loads(stockage.lire_texte(nom_ou_chemin))
    return json.loads(
        stockage.lire_texte(stockage.joindre(dossier, f"{nom_ou_chemin}.json"))
    )


def cmd_comparer(a: argparse.Namespace) -> int:
    dossier = _dossier_gabarits(a.dossier)
    changements = comparer_versions(
        _charger_cfg(a.ancien, dossier), _charger_cfg(a.nouveau, dossier)
    )
    with pd.option_context("display.width", 160, "display.max_colwidth", 80):
        print(
            pd.DataFrame(changements).to_string(index=False)
            if changements
            else "aucun changement"
        )
    return 0


def cmd_catalogue(a: argparse.Namespace) -> int:
    cfg, _ = charger(_dossier_gabarits(a.dossier), a.gabarit)
    df = catalogue(cfg)
    if a.sortie:
        stockage.ecrire_texte(a.sortie, df.to_csv(index=False))
    else:
        print(df.to_string(index=False))
    return 0


def analyseur() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m mesure_qualite.decoupage",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-v", "--verbeux", action="store_true", help="journal detaille")
    sp = ap.add_subparsers(dest="commande", required=True)

    c = sp.add_parser("construire", help="construire le gabarit d'un PDF vierge")
    c.add_argument(
        "--pdf", required=True, help="PDF vierge de l'imprime (local ou s3://)"
    )
    c.add_argument("--formulaire", help="ex. BI (sinon deduit du nom du PDF)")
    c.add_argument(
        "--millesime", type=int, help="ex. 2025 (sinon deduit du nom du PDF)"
    )
    c.add_argument(
        "--sortie", help="dossier des gabarits (defaut : configs/campagnes.yaml)"
    )
    c.add_argument("--dpi", type=int, default=200)
    c.add_argument("--ordre", default="auto", help='ordre des unites : "auto" ou "pdf"')
    c.set_defaults(fonction=cmd_construire)

    d = sp.add_parser(
        "decouper", help="decouper les scans d'un formulaire d'une campagne"
    )
    d.add_argument("--campagne", required=True, help="ex. RG25")
    d.add_argument("--formulaire", required=True, help="ex. BI")
    d.add_argument(
        "--limite", type=int, help="ne traiter que les N premiers scans (essai)"
    )
    d.add_argument("--refaire", action="store_true", help="ignorer l'index existant")
    d.add_argument("--paralleles", type=int, default=8)
    d.add_argument(
        "--duree-max",
        type=float,
        help="minutes ; relancer reprend la ou on s'est arrete",
    )
    d.add_argument("--format", choices=["jpg", "png"])
    d.add_argument("--blocs", nargs="+", help='ex. q23 cadre_nom (defaut : "toutes")')
    d.set_defaults(fonction=cmd_decouper)

    m = sp.add_parser("comparer", help="comparer deux versions d'un formulaire")
    m.add_argument("ancien", help="nom (ex. FLRM_2022) ou chemin d'un JSON")
    m.add_argument("nouveau", help="nom (ex. FLR_2027) ou chemin d'un JSON")
    m.add_argument(
        "--dossier", help="dossier des gabarits (defaut : configs/campagnes.yaml)"
    )
    m.set_defaults(fonction=cmd_comparer)

    k = sp.add_parser("catalogue", help="lister tous les elements d'un gabarit")
    k.add_argument("gabarit", help="ex. BI_2025")
    k.add_argument(
        "--dossier", help="dossier des gabarits (defaut : configs/campagnes.yaml)"
    )
    k.add_argument("--sortie", help="fichier CSV (sinon affichage)")
    k.set_defaults(fonction=cmd_catalogue)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = analyseur().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbeux else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return a.fonction(a)


if __name__ == "__main__":
    sys.exit(main())
