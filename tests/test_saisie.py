"""Mise au format Fiqual des lectures (consignes de saisie)."""

from __future__ import annotations

import pandas as pd
import pytest

from mesure_qualite import lecture as lc
from mesure_qualite.lecture import saisie


@pytest.mark.parametrize(
    "valeur, attendu",
    [
        ("5", "05"),
        ("05", "05"),
        ("31", "31"),
        ("32", ""),
        ("ILLISIBLE", "ILLISIBLE"),
        ("", ""),
    ],
)
def test_jour(valeur, attendu):
    assert saisie.jour_ou_mois(valeur, 31) == attendu


def test_mois_en_lettres():
    assert saisie.jour_ou_mois("Mars", 12) == "03"
    assert saisie.jour_ou_mois("13", 12) == ""


@pytest.mark.parametrize(
    "valeur, attendu",
    [("1985", "1985"), ("85", "1985"), ("08", "2008"), ("30", "1930"), ("1890", ""),
     ("2030", ""), ("198", ""), ("ILLISIBLE", "ILLISIBLE")],
)  # fmt: skip
def test_annee(valeur, attendu):
    assert saisie.annee(valeur, 2025) == attendu


def test_date_valide():
    assert saisie.date_valide("29", "02", "2000")
    assert not saisie.date_valide("29", "02", "2001")
    assert not saisie.date_valide("31", "04", "")


@pytest.mark.parametrize(
    "valeur, attendu",
    [("5", "05"), ("69", "69"), ("Rhône", "69"), ("Seine-Saint-Denis", "93"),
     ("2a", "2A"), ("971", "971"), ("69003", "69"), ("97110", "971"), ("96", "")],
)  # fmt: skip
def test_departement(valeur, attendu):
    assert saisie.departement(valeur) == attendu


@pytest.mark.parametrize(
    "valeur, attendu",
    [("St-Étienne", "SAINT ETIENNE"), ("Paris 18", "PARIS 18E ARRONDISSEMENT"),
     ("Paris XVIIIe", "PARIS 18E ARRONDISSEMENT"), ("75018", "PARIS 18E ARRONDISSEMENT"),
     ("Lyon 1er", "LYON 1ER ARRONDISSEMENT"), ("Paris (75)", "PARIS"),
     ("Brest 29", "BREST"), ("Brest 29200", "BREST"), ("Marseille", "MARSEILLE")],
)  # fmt: skip
def test_commune(valeur, attendu):
    assert saisie.commune(valeur) == attendu


def test_pays_et_adresse():
    assert saisie.pays("France", "69") == ""
    assert saisie.pays("France", "971") == "FRANCE"
    assert saisie.pays("France", "") == "FRANCE"
    assert saisie.pays("Irak", "") == "IRAQ"
    assert saisie.numero_voie("10-12") == "10"
    assert saisie.indice("bis") == "B"


def test_au_format_fiqual():
    s = lc.sujet("naissance")
    lus = pd.DataFrame(
        [("1_1", c, v) for c, v in zip(s.noms_champs, ["31", "4", "85", "St Etienne", "42", "France"])],
        columns=["cle", "champ", "lu"],
    )  # fmt: skip
    f = saisie.au_format_fiqual(lus, s, 2025).set_index("champ")
    assert f.loc["ANNEE", "lu"] == "1985" and f.loc["ANNEE", "lu_brut"] == "85"
    assert f.loc["JOUR", "lu"] == "" and f.loc["MOIS", "lu"] == ""  # 31 avril
    assert f.loc["COMMUNE", "lu"] == "SAINT ETIENNE"
    assert f.loc["PAYS", "lu"] == ""


def test_colonne_departement_par_contenu():
    fq = {
        f"1_{i}": ["3", "", "1", "", str(i), "x", "69" if i % 2 else "05", "LYON"]
        for i in range(20)
    }
    assert saisie.colonne_departement_par_contenu(fq, pres_de=7) == 6


def test_nom_prenom_et_sujet_identite():
    assert saisie.nom_ou_prenom("Mme Dupont épouse Martin") == "DUPONT EPOUSE MARTIN"
    assert saisie.nom_ou_prenom("M") == "M"  # une seule lettre : on ne retire rien
    s = lc.sujet("identite")
    assert s.bloc == "cadre_nom" and s.noms_champs == ["NOM", "PRENOM"]
    lus = pd.DataFrame([("1_1", "NOM", "Mr Durand"), ("1_1", "PRENOM", "Jean-Luc")],
                       columns=["cle", "champ", "lu"])  # fmt: skip
    f = saisie.au_format_fiqual(lus, s, 2025).set_index("champ")
    assert f.loc["NOM", "lu"] == "DURAND" and f.loc["PRENOM", "lu"] == "JEAN LUC"
