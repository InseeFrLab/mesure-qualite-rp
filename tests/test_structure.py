"""Verifie que le paquet et ses sous-modules s'importent."""

import pytest

SOUS_MODULES = [
    "io",
    "parsing",
    "regles",
    "codification",
    "perception",
    "assemblage",
    "evaluation",
]


def test_paquet_importable():
    import mesure_qualite  # noqa: F401


@pytest.mark.parametrize("nom", SOUS_MODULES)
def test_sous_module_importable(nom):
    __import__(f"mesure_qualite.{nom}")
