"""Consigne V2 : departement en entier, metropole et outre-mer pareil ; V1 inchangee."""

from mesure_qualite.lecture.consignes import consigne
from mesure_qualite.lecture.sujets import sujet


def test_v2_ajoute_la_regle_du_departement():
    v1, v2 = consigne(sujet("naissance"), "V1"), consigne(sujet("naissance"), "V2")
    assert "EN ENTIER" in v2 and "EN ENTIER" not in v1
    assert "971" in v2 and "2A" in v2


def test_v2_identique_a_v1_sans_departement():
    assert consigne(sujet("identite"), "V2") == consigne(sujet("identite"), "V1")
