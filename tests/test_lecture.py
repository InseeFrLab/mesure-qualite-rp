"""Tests de l'etape 2 (lecture). Aucun appel reseau : les services sont simules."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mesure_qualite import lecture as lc
from mesure_qualite.lecture import campagne as lc_campagne
from mesure_qualite.lecture import modeles as lc_modeles


@pytest.fixture(scope="module")
def naissance():
    return lc.sujet("naissance")


# --- sujets et consignes ---------------------------------------------------------------
def test_sujets_de_la_configuration(naissance):
    assert naissance.bloc == "q2" and naissance.formulaire == "BI"
    assert naissance.noms_champs == [
        "JOUR",
        "MOIS",
        "ANNEE",
        "COMMUNE",
        "DEPARTEMENT",
        "PAYS",
    ]
    assert naissance.groupes["DATE"] == ("JOUR", "MOIS", "ANNEE")
    assert naissance.champ_de_variable("jnai") == "JOUR"
    assert naissance.champ_de_variable("DPNAI") == "DEPARTEMENT"
    assert naissance.champ_de_variable("NOM") is None
    adresse = lc.sujet("adresse_travail")
    assert adresse.bloc == "q23" and adresse.champ_de_variable("NOMVOI") == "NOM_VOIE"
    with pytest.raises(KeyError):
        lc.sujet("inconnu")


@pytest.mark.parametrize("version", ["V0", "V1"])
def test_consigne_demande_tous_les_champs_en_json(naissance, version):
    texte = lc.consigne(naissance, version)
    attendu = json.loads(texte[texte.rindex("{") :])
    assert list(attendu) == naissance.noms_champs
    assert "VIDE" in texte and "ILLISIBLE" in texte
    with pytest.raises(ValueError):
        lc.consigne(naissance, "V9")


# --- comparaison -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "valeur, type_, attendu",
    [
        ("05", "entier", "5"),
        (" 1 985 ", "entier", "1985"),
        ("mars", "mois", "3"),
        ("Février", "mois", "2"),
        ("03", "mois", "3"),
        ("2a", "departement", "2A"),
        ("01", "departement", "1"),
        ("St-Étienne", "lieu", "SAINT ETIENNE"),
        ("12 bd Victor-Hugo", "voie", "12 BOULEVARD VICTOR HUGO"),
        ("vide", "lieu", ""),
        (None, "entier", ""),
        ("illisible", "lieu", "ILLISIBLE"),
    ],
)
def test_comparable(valeur, type_, attendu):
    assert lc.comparable(valeur, type_) == attendu


def test_verdicts():
    assert lc.verdict("5", "5") == "juste"
    assert lc.verdict("", "5") == "avouee"
    assert lc.verdict("ILLISIBLE", "5") == "avouee"
    assert lc.verdict("6", "5") == "silencieuse"
    assert lc.verdict("6", "5", signale=True) == "relue"
    assert lc.verdict("6", None) == "ecartee"
    assert lc.verdict("", "") == "juste"  # rien d'ecrit, rien lu
    assert lc.verdict_groupe(["juste", "juste", "juste"]) == "juste"
    assert lc.verdict_groupe(["juste", "avouee", "silencieuse"]) == "silencieuse"
    assert lc.verdict_groupe(["juste", "avouee"]) == "avouee"
    assert lc.verdict_groupe(["juste", "ecartee"]) == "ecartee"


def test_type_erreur():
    assert lc.type_erreur("1958", "1953") == "1 chiffre"
    assert lc.type_erreur("195", "1953") == "chiffre en trop ou en moins"
    assert lc.type_erreur("", "LYON") == "rien lu ou illisible"
    assert lc.type_erreur("LYON", "") == "lu alors que rien n'est ecrit"


# --- reponse d'un modele ---------------------------------------------------------------
def test_extraire_json():
    assert lc.extraire_json('```json\n{"JOUR": "5"}\n```') == {"JOUR": "5"}
    assert lc.extraire_json('Voici : {"A": 1} puis {"JOUR": "7"}') == {"JOUR": "7"}
    assert lc.extraire_json("pas de json") == {}


class _Faux:
    def __init__(self, code, corps):
        self.status_code, self._corps = code, corps
        self.text = json.dumps(corps)

    def json(self):
        return self._corps

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(response=self)


def test_lire_retire_un_parametre_refuse(monkeypatch, naissance):
    monkeypatch.setenv("EDENAI_API_KEY", "cle-de-test")
    envois = []

    def post(url, headers, json, timeout):
        envois.append(json)
        if "temperature" in json:
            return _Faux(400, {"error": "temperature is not supported"})
        contenu = (
            '{"JOUR": "05", "MOIS": "3", "ANNEE": "1985", "COMMUNE": "Lyon", '
            '"DEPARTEMENT": "69", "PAYS": "VIDE"}'
        )
        return _Faux(
            200,
            {
                "choices": [{"message": {"content": contenu}}],
                "usage": {"total_tokens": 900},
                "cost": 0.002,
                "model": "openai/gpt-x-2026",
            },
        )

    monkeypatch.setattr(lc_modeles.requests, "post", post)
    adaptations = {}
    r = lc.lire(
        lc.Lecteur("openai/gpt-x", "eden"),
        b"\xff\xd8jpeg",
        "consigne",
        naissance.noms_champs,
        adaptations=adaptations,
    )
    assert r.ok and r.json_ok and r.champs["JOUR"] == "05" and r.cout == 0.002
    assert adaptations == {"openai/gpt-x": ["temperature"]}
    assert envois[0]["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_lire_sans_cle_explique(monkeypatch):
    monkeypatch.delenv("EDENAI_API_KEY", raising=False)
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="EDENAI_API_KEY"):
        lc.cle_api("eden")


# --- campagne : cache et reprise ---------------------------------------------------------
def test_lire_tout_reprend_sans_repayer(monkeypatch, tmp_path):
    appels = []

    def faux_lire(lecteur, image, consigne, champs, **_):
        appels.append(image)
        return lc.Reponse(
            {c: image.decode() for c in champs},
            True,
            secondes=0.1,
            cout=0.01,
            json_ok=True,
        )

    monkeypatch.setattr(lc_campagne, "lire", faux_lire)
    images = {f"1_{i}": (lambda i=i: str(i).encode()) for i in range(12)}
    lecteur = lc.Lecteur("m", "eden")
    cache = str(tmp_path / "c.csv")
    cpt = lc.Compteurs()
    lus, bilan = lc.lire_tout(
        lecteur, "V1", "x", ["A"], images, cache, cpt, paralleles=3, dire=lambda _: None
    )
    assert len(lus) == 12 and bilan.lus == 12 and len(appels) == 12
    assert cpt.cout == pytest.approx(0.12)
    lus2, bilan2 = lc.lire_tout(
        lecteur, "V1", "x", ["A"], images, cache, cpt, dire=lambda _: None
    )
    assert len(lus2) == 12 and bilan2.lus == 0 and len(appels) == 12
    long = lc.en_long(lus2, ["A"])
    assert set(long.columns) == {"cle", "champ", "lu"} and set(long.champ) == {"A"}


def test_lire_tout_s_arrete_si_tout_echoue(monkeypatch, tmp_path):
    import time

    def echec(*_, **__):
        time.sleep(0.02)
        return lc.Reponse({"A": ""}, False, erreur="404 modele")

    monkeypatch.setattr(lc_campagne, "lire", echec)
    images = {f"1_{i}": (lambda: b"x") for i in range(40)}
    _, bilan = lc.lire_tout(
        lc.Lecteur("faux", "eden"),
        "V1",
        "x",
        ["A"],
        images,
        str(tmp_path / "c.csv"),
        lc.Compteurs(),
        paralleles=1,
        dire=lambda _: None,
    )
    assert bilan.arret.startswith("les 5 premiers appels") and bilan.echecs < 40


# --- verite et mesures -------------------------------------------------------------------
def _fiqual():
    # type|x|UT|x|numero|...|JOUR|MOIS|ANNEE|COMMUNE
    return {
        lc.cle("0101", "0007"): [
            "3",
            "",
            "0101",
            "",
            "0007",
            "X",
            "05",
            "03",
            "1985",
            "LYON",
        ],
        lc.cle("101", "8"): ["3", "", "101", "", "8", "X", "12", "11", "1990", "PARIS"],
        lc.cle("101", "9"): ["3", "", "101", "", "9", "X", "01", "01", "1970", "NICE"],
    }


def _divergences():
    return pd.DataFrame(
        {
            "VARIABLE": ["JNAI", "ANAI", "CNAI"],
            "cle": ["101_7", "101_8", "101_9"],
            "VALEUR_PMQ": ["5", "1990", "NICE"],
            "VALEUR_PAD": ["6", "1996", "NIC"],
            "VALEUR_ATTENDUE": ["5", "1996", ""],
            "arbitree": [True, True, False],
        }
    )


def test_cles():
    assert lc.cle("0101", "0007") == "101_7"
    assert lc.cle_image("/a/b/0407_9761_5300000184_BI.jpg") == "9761_5300000184"
    assert lc.cle_image("6901_5300000403_BI.jpg") == "6901_5300000403"


def test_colonnes_par_votes_et_verite(naissance):
    fq, dv = _fiqual(), _divergences()
    cols, detail = lc.colonnes_par_votes(fq, dv, naissance)
    assert cols == {"JOUR": 6, "ANNEE": 8, "COMMUNE": 9}
    assert set(detail.champ) == {"JOUR", "ANNEE", "COMMUNE"}
    assert lc.colonnes_date_par_contenu(fq, 2025) == {"JOUR": 6, "MOIS": 7, "ANNEE": 8}
    cols["MOIS"] = 7
    v = lc.table_verite(fq, dv, naissance, cols, ["101_7", "101_8", "101_9"])
    ligne = v.set_index(["cle", "champ"])
    assert ligne.loc[("101_7", "JOUR"), "groupe"] == "B"
    assert (
        ligne.loc[("101_8", "ANNEE"), "verite"] == "1996"
    )  # le SeRN donne tort au PMQ
    assert pd.isna(ligne.loc[("101_9", "COMMUNE"), "verite"])  # non arbitree
    assert ligne.loc[("101_8", "MOIS"), "verite"] == "11"


def test_juger_resumer_comparer(naissance):
    fq, dv = _fiqual(), _divergences()
    cols = {"JOUR": 6, "MOIS": 7, "ANNEE": 8, "COMMUNE": 9}
    v = lc.table_verite(fq, dv, naissance, cols, list(fq))
    bon = pd.DataFrame(
        [(k, c, x) for k, c, x in zip(v.cle, v.champ, v.verite.fillna("?"))],
        columns=["cle", "champ", "lu"],
    )
    moyen = bon.copy()
    moyen.loc[moyen.champ == "ANNEE", "lu"] = "2000"
    t = lc.juger(v, {"bon": bon, "moyen": moyen}, naissance)
    date = t[t.champ == "DATE"].set_index("cle")
    assert date.loc["101_7", "verdict bon"] == "juste"
    assert date.loc["101_7", "verdict moyen"] == "silencieuse"
    assert date.loc["101_8", "verdict PMQ"] == "silencieuse"  # erreur prouvee du PMQ
    r = lc.resumer(t, ["bon", "moyen"], ["ANNEE"]).set_index("lecteur")
    assert r.loc["bon", "justes_%"] == 100.0 and r.loc["moyen", "justes"] == 0
    m = lc.comparer(t, "bon", "moyen", ["ANNEE"]).iloc[0]
    assert m["bon seul juste"] == 3 and m["moyen seul juste"] == 0
    bas, haut = lc.barre_pmq(t, "ANNEE")
    assert bas == pytest.approx(1 / 3) and haut == pytest.approx(1 / 3)


def test_wilson_et_mcnemar():
    lo, hi = lc.wilson(42, 43)
    assert 0.87 < lo < 0.88 and hi > 0.99
    a = pd.Series([True] * 13 + [False] * 5)
    b = pd.Series([False] * 13 + [False] * 5)
    assert lc.mcnemar(a, b) == (13, 0, pytest.approx(0.000244, abs=1e-5))
