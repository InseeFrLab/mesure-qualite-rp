"""Tests de la page HTML des erreurs (donnees simulees)."""

from __future__ import annotations

import io

import pandas as pd
from PIL import Image

from mesure_qualite import lecture as lc


def _jpeg() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (900, 300), "white").save(b, format="JPEG")
    return b.getvalue()


def test_page_des_erreurs_et_comparaison():
    s = lc.sujet("naissance")
    fq = {
        "1_1": ["3", "", "1", "", "1", "05", "03", "1985", "LYON"],
        "1_2": ["3", "", "1", "", "2", "12", "11", "1990", "NICE"],
    }
    dv = pd.DataFrame(
        {"VARIABLE": ["ANAI"], "cle": ["1_2"], "VALEUR_PMQ": ["1990"],
         "VALEUR_PAD": ["1996"], "VALEUR_ATTENDUE": ["1996"], "arbitree": [True]}
    )  # fmt: skip
    cols = {"JOUR": 5, "MOIS": 6, "ANNEE": 7, "COMMUNE": 8}
    v = lc.table_verite(fq, dv, s, cols, list(fq))
    lu = v[["cle", "champ"]].assign(lu=v.verite)
    lu.loc[(lu.cle == "1_1") & (lu.champ == "ANNEE"), "lu"] = "1958"
    t = lc.juger(v, {"m": lu}, s)

    c = lc.comparaison_verite(t, ["m"]).set_index(["champ", "lecteur"])
    assert c.loc[("ANNEE", "m"), "fausses"] == 1
    assert c.loc[("ANNEE", "PMQ"), "fausses"] == 1  # le SeRN donne tort au PMQ
    assert c.loc[("COMMUNE", "m"), "justes_%"] == 100.0

    erreurs = lc.lignes_en_erreur(t, ["m"], s)
    assert set(erreurs.cle) == {"1_1", "1_2"}

    page = lc.galerie_erreurs(t, ["m"], {"1_1": _jpeg, "1_2": _jpeg}, s, titre="essai")
    assert page.startswith("<!doctype html>")
    assert "FAUX, attendu 1985" in page and "arbitrage SeRN" in page
    assert page.count("data:image/jpeg;base64,") == 2
