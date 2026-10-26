"""Tests de l'etape 1 (decoupage).

Les tests unitaires n'ont besoin de rien. Les tests d'integration construisent le
gabarit du bulletin individuel 2025 a partir de son PDF vierge : ils sont ignores si le
PDF n'est pas disponible. Indiquer son chemin par la variable d'environnement
MQ_PDF_BI (ex. MQ_PDF_BI="2025-imprime-3-BI avec VAR.pdf" pytest).
Aucune donnee reelle : les scans sont simules a partir de la page vierge.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mesure_qualite import decoupage as dc
from mesure_qualite.decoupage import _outils, stockage
from mesure_qualite.decoupage.campagne import (
    ParametresDecoupage,
    formulaire_et_millesime,
    lire_configuration,
)
from mesure_qualite.decoupage.gabarit import cle_question
from mesure_qualite.decoupage.simulation import erreur_mm, erreur_recalage, faux_scan


# --- tests unitaires -----------------------------------------------------------------
def test_iou():
    assert _outils.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _outils.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # deux carres de 100 qui se recouvrent sur 50 : 50 / 150
    assert _outils.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert np.isnan(_outils.iou(None, (0, 0, 1, 1)))


def test_normaliser_et_slug():
    assert _outils.normaliser("  Quelle est l'adresse ?  ") == "QUELLE EST L ADRESSE"
    assert _outils.normaliser("Élève, étudiant(e)") == "ELEVE ETUDIANT E"
    assert (
        _outils.slug("Cadre à remplir par l'agent recenseur")
        == "cadre_a_remplir_par_l_agent"
    )


def test_lignes_de_separe_deux_colonnes():
    mot = lambda t, x0, x1, top: {
        "text": t,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": top + 8,
    }
    mots = [
        mot("Nom", 10, 30, 100),
        mot(":", 31, 33, 100),
        mot("Surface", 300, 340, 100),
    ]
    lignes = _outils.lignes_de(mots)
    assert [_outils.texte_de(ligne) for ligne in lignes] == ["Nom :", "Surface"]


def test_formulaire_et_millesime():
    assert formulaire_et_millesime("2025-imprime-3-BI avec VAR.pdf") == ("BI", 2025)
    assert formulaire_et_millesime("2027-imprime-009-FLR.pdf") == ("FLR", 2027)
    assert formulaire_et_millesime("questionnaire.pdf") == (None, None)


def _bloc(id_, numero, intitule, variables=()):
    return {
        "id": id_,
        "sorte": "question",
        "question": numero,
        "intitule": intitule,
        "variables_formulaire": list(variables),
    }


def test_comparer_versions_par_cle_et_non_par_numero():
    ancien = {
        "blocs": [
            _bloc("q3", "3", "Ascenseur ?"),
            _bloc("q7", "7", "Organisme d'HLM ?"),
            _bloc("q9", "9", "Sexe", ["SEXE"]),
        ]
    }
    nouveau = {
        "blocs": [
            _bloc("q4", "4", "Ascenseur ?"),
            _bloc("q8", "8", "Bailleur social ?"),
            _bloc("q9", "9", "Votre sexe", ["SEXE"]),
        ]
    }
    ch = {c["changement"] for c in dc.comparer_versions(ancien, nouveau)}
    # renumerotee par le texte ; reformulee = supprimee + ajoutee ; la variable SEXE suffit
    assert ch == {"renumerotee (3 -> 4)", "ajoutee", "supprimee"}
    assert cle_question(nouveau["blocs"][2]) == "var:SEXE"


def test_selectionner_et_blocs_a_decouper():
    cfg = {
        "formulaire": "T",
        "millesime": 2025,
        "marge_defaut": 0.004,
        "blocs": [
            {
                "id": "q1",
                "sorte": "question",
                "question": "1",
                "intitule": "Nom",
                "zone": [0.1, 0.1, 0.4, 0.1],
                "zone_intitule": None,
                "reponse": "ecriture",
                "texte_complet": "Nom",
                "theme": "identite",
            },
            {
                "id": "q2",
                "sorte": "question",
                "question": "2",
                "intitule": "Info",
                "zone": [0.1, 0.3, 0.4, 0.05],
                "zone_intitule": None,
                "reponse": "aucune (information)",
                "texte_complet": "Info",
                "theme": None,
            },
        ],
        "champs": [
            {
                "id": "q1_texte1",
                "bloc": "q1",
                "type": "texte",
                "zone": [0.2, 0.12, 0.2, 0.02],
                "variables_formulaire": ["NOM"],
                "libelle_gauche": "Nom :",
                "libelle_dessous": "",
                "libelle_dessus": "",
                "colonne": "",
                "rangee": "Nom :",
                "rubrique": "",
            }
        ],
    }
    s = dc.selectionner(cfg, variables=["NOM"])
    assert s["elements"] == ["q1_texte1"] and s["zone"] == [0.2, 0.12, 0.2, 0.02]
    assert s["marge"] == 0.0015  # champs seuls : marge serree
    assert dc.selectionner(cfg, question=1, bloc_entier=True)["zone"] == [
        0.1,
        0.1,
        0.4,
        0.1,
    ]
    assert [b["id"] for b in dc.blocs_a_decouper(cfg)] == ["q1"]  # q2 : information
    with pytest.raises(ValueError):
        dc.selectionner(cfg, variables=["ABSENTE"])


def test_stockage_local(tmp_path):
    chemin = stockage.joindre(str(tmp_path), "a", "b.jpg")
    img = np.full((20, 30, 3), 200, np.uint8)
    assert stockage.ecrire_image(chemin, img) > 0
    assert stockage.lire_image(chemin).shape == (20, 30, 3)
    assert stockage.lister(str(tmp_path), (".jpg",)) == [chemin]


def test_configuration_du_depot():
    c = lire_configuration()
    p = dc.parametres("RG25", "BI", c)
    assert p.gabarit == "BI_2025" and p.suffixe == "BI"
    assert p.scans == "s3://projet-mesure-qualite-rp/data/raw/RG25"
    assert p.sortie == "s3://projet-mesure-qualite-rp/data/decoupes/RG25/BI"
    assert p.index.endswith("RG25/BI/index_BI_RG25.csv")
    with pytest.raises(KeyError):
        dc.parametres("RG25", "INCONNU", c)


# --- tests d'integration (PDF vierge du bulletin individuel 2025) --------------------------
PDF_BI = os.environ.get("MQ_PDF_BI")
avec_pdf = pytest.mark.skipif(
    not PDF_BI or not Path(PDF_BI).exists(),
    reason="PDF vierge du BI absent (variable MQ_PDF_BI)",
)


@pytest.fixture(scope="module")
def gabarit_bi():
    cfg, refs = dc.construire(PDF_BI, "BI", 2025)
    return dc.ajouter_themes_par_regles(cfg), refs


@avec_pdf
def test_construction_bi_2025(gabarit_bi):
    cfg, refs = gabarit_bi
    ctl = cfg["controle"]
    assert ctl["questions_trouvees"] == list(range(1, 33))
    assert ctl["questions_manquantes"] == []
    assert ctl["n_blocs"] == 34 and ctl["n_champs"] == 130
    assert len(refs) == 2 and refs[0].shape[1] == cfg["reference"]["largeur"]
    assert len(dc.blocs_a_decouper(cfg)) == 30  # 4 questions d'information
    q23 = next(b for b in cfg["blocs"] if b["id"] == "q23")
    assert {"NUMVOI", "TYPVOI", "NOMVOI"} <= set(q23["variables_formulaire"])


@avec_pdf
@pytest.mark.parametrize("mode", ["couleur", "gris", "noir_blanc"])
def test_recalage_precis_a_moins_d_un_demi_millimetre(gabarit_bi, mode):
    cfg, refs = gabarit_bi
    img, T, hauteurs = faux_scan(
        cfg, refs, echelle=0.75, angle=1.5, mode=mode, graine=3
    )
    _, infos = dc.Recaleur(cfg, refs).toile(img)
    assert all(i["recalee"] for i in infos)
    assert erreur_mm(cfg, erreur_recalage(cfg, infos, T, hauteurs)) < 0.5


@avec_pdf
def test_decouper_campagne_et_reprise(gabarit_bi, tmp_path):
    cfg, refs = gabarit_bi
    gabarits, scans, sortie = (
        tmp_path / "gabarits",
        tmp_path / "scans",
        tmp_path / "decoupes",
    )
    dc.enregistrer(cfg, refs, str(gabarits))
    for k in range(2):
        img, _, _ = faux_scan(cfg, refs, angle=0.5 + k, graine=k)
        stockage.ecrire_image(str(scans / f"6901_00000{k}_BI.jpg"), img)
    stockage.ecrire_image(str(scans / "6901_000009_FL.jpg"), img)  # autre formulaire
    p = ParametresDecoupage(
        campagne="TEST",
        formulaire="BI",
        gabarit="BI_2025",
        scans=str(scans),
        suffixe="BI",
        gabarits=str(gabarits),
        sortie=str(sortie),
    )
    index = dc.decouper_campagne(p, paralleles=2)
    assert len(index) == 2 * 30 and set(index.methode) == {"recalage"}
    assert all(Path(f).exists() for f in index.decoupe)
    assert dc.resume_index(index)["methode_pct"] == {"recalage": 100.0}
    relu = pd.read_csv(p.index)
    assert len(relu) == 60
    # reprise : rien a refaire
    index2 = dc.decouper_campagne(p, paralleles=2)
    assert len(index2) == 60
