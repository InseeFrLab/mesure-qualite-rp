"""Sujet « sexe » : la case cochee mise au format du Fiqual (1 masculin, 2 feminin)."""

import pandas as pd
import pytest

from mesure_qualite.lecture import saisie, sujets


@pytest.mark.parametrize(
    "valeur, attendu",
    [("1", "1"), ("Masculin", "1"), ("homme", "1"), ("2", "2"), ("Féminin", "2"),
     ("F", "2"), ("1 et 2", ""), ("VIDE", "VIDE"), ("ILLISIBLE", "ILLISIBLE")],
)
def test_sexe(valeur, attendu):
    assert saisie.sexe(valeur) == attendu


def test_sujet_sexe():
    s = sujets.sujet("sexe")
    assert s.noms_champs == ["SEXE"] and s.champ_de_variable("SEXE") == "SEXE"
    lu = pd.DataFrame({"cle": ["k"], "champ": ["SEXE"], "lu": ["Feminin"]})
    assert saisie.au_format_fiqual(lu, s, 2025).lu.tolist() == ["2"]
