"""Les sujets etudies (date et lieu de naissance, adresse du lieu de travail...).

Un sujet dit quel bloc du gabarit on donne a lire (etape 1) et quels champs on attend
en retour. Tout vient de configs/lecture.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TYPES_CHAMP = {"entier", "mois", "texte", "lieu", "departement", "voie"}


def racine_depot() -> Path:
    """Racine du depot (src/mesure_qualite/lecture/sujets.py -> 3 niveaux plus haut)."""
    return Path(__file__).resolve().parents[3]


def lire_configuration(fichier: str | Path | None = None) -> dict:
    fichier = Path(fichier) if fichier else racine_depot() / "configs" / "lecture.yaml"
    return yaml.safe_load(fichier.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Champ:
    nom: str
    variables: tuple[str, ...]
    type: str
    aide: str

    def reconnait(self, variable: str) -> bool:
        """La variable du fichier des divergences correspond-elle a ce champ ?"""
        v = str(variable).strip().upper()
        return any(re.fullmatch(motif, v) for motif in self.variables)


@dataclass(frozen=True)
class Sujet:
    nom: str
    titre: str
    formulaire: str
    bloc: str
    type_ligne_fiqual: str
    document: str
    cible: str
    champs: tuple[Champ, ...]
    groupes: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def noms_champs(self) -> list[str]:
        return [c.nom for c in self.champs]

    def champ(self, nom: str) -> Champ:
        return next(c for c in self.champs if c.nom == nom)

    def champ_de_variable(self, variable: str) -> str | None:
        """Le champ auquel appartient une variable des divergences, ou None."""
        return next((c.nom for c in self.champs if c.reconnait(variable)), None)


def sujet(nom: str, configuration: dict | None = None) -> Sujet:
    c = configuration or lire_configuration()
    try:
        s = c["sujets"][nom]
    except KeyError as e:
        raise KeyError(
            f"Sujet « {nom} » absent de configs/lecture.yaml "
            f"(disponibles : {sorted(c.get('sujets', {}))})"
        ) from e
    champs = []
    for n, d in s["champs"].items():
        if d["type"] not in TYPES_CHAMP:
            raise ValueError(
                f"{nom}.{n} : type {d['type']!r} inconnu ({sorted(TYPES_CHAMP)})"
            )
        champs.append(
            Champ(n, tuple(v.upper() for v in d["variables"]), d["type"], d["aide"])
        )
    groupes = {g: tuple(m) for g, m in (s.get("groupes") or {}).items()}
    for g, membres in groupes.items():
        inconnus = set(membres) - {c.nom for c in champs}
        if inconnus:
            raise ValueError(f"{nom}, groupe {g} : champs inconnus {sorted(inconnus)}")
    return Sujet(
        nom=nom,
        titre=s["titre"],
        formulaire=s["formulaire"],
        bloc=s["bloc"],
        type_ligne_fiqual=str(s["type_ligne_fiqual"]),
        document=s["document"],
        cible=s["cible"],
        champs=tuple(champs),
        groupes=groupes,
    )
