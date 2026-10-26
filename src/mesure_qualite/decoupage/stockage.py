"""Lecture et ecriture de fichiers, en local ou sur le stockage S3 du SSP Cloud.

Un chemin qui commence par « s3:// » est lu sur S3, tout autre chemin en local. Le
reste du paquet ne connait que ces fonctions : le meme code tourne sur un poste,
dans un service Onyxia ou dans un test.

Sur Onyxia, les identifiants S3 sont dans l'environnement (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_S3_ENDPOINT) : rien a configurer.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

PREFIXE_S3 = "s3://"


def est_s3(chemin: str | Path) -> bool:
    return str(chemin).startswith(PREFIXE_S3)


@lru_cache(maxsize=1)
def _s3():
    """Connexion S3 unique (s3fs, installe avec le projet)."""
    import s3fs

    point = os.environ.get("AWS_S3_ENDPOINT")
    options = {"client_kwargs": {"endpoint_url": f"https://{point}"}} if point else {}
    return s3fs.S3FileSystem(**options)


def reinitialiser_connexion() -> None:
    """A appeler apres un renouvellement des identifiants S3 (jeton expire)."""
    _s3.cache_clear()


def joindre(base: str | Path, *parties: str) -> str:
    """Joint des morceaux de chemin, en local comme sur S3."""
    if est_s3(base):
        return "/".join([str(base).rstrip("/"), *(p.strip("/") for p in parties)])
    return str(Path(base, *parties))


def existe(chemin: str | Path) -> bool:
    return _s3().exists(str(chemin)) if est_s3(chemin) else Path(chemin).exists()


def lire_octets(chemin: str | Path) -> bytes:
    if est_s3(chemin):
        with _s3().open(str(chemin), "rb") as f:
            return f.read()
    return Path(chemin).read_bytes()


def ecrire_octets(chemin: str | Path, octets: bytes) -> None:
    if est_s3(chemin):
        with _s3().open(str(chemin), "wb") as f:
            f.write(octets)
        return
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    Path(chemin).write_bytes(octets)


def lire_texte(chemin: str | Path) -> str:
    return lire_octets(chemin).decode("utf-8")


def ecrire_texte(chemin: str | Path, texte: str) -> None:
    ecrire_octets(chemin, texte.encode("utf-8"))


def lister(dossier: str | Path, suffixes: tuple[str, ...] = ()) -> list[str]:
    """Tous les fichiers sous un dossier (recursif), tries, filtres par extension."""
    if est_s3(dossier):
        racine = str(dossier).removeprefix(PREFIXE_S3)
        fichiers = [PREFIXE_S3 + f for f in _s3().find(racine)]
    else:
        fichiers = [str(p) for p in Path(dossier).rglob("*") if p.is_file()]
    if suffixes:
        fichiers = [
            f
            for f in fichiers
            if f.lower().endswith(tuple(s.lower() for s in suffixes))
        ]
    return sorted(fichiers)


def lire_image(chemin: str | Path) -> np.ndarray:
    """Image RGB (tableau hauteur x largeur x 3, uint8)."""
    return np.array(Image.open(io.BytesIO(lire_octets(chemin))).convert("RGB"))


def encoder_image(a: np.ndarray, format_: str = "jpg", qualite: int = 92) -> bytes:
    b = io.BytesIO()
    if format_.lower() == "png":
        Image.fromarray(a).save(b, format="PNG", optimize=True)
    else:
        Image.fromarray(a).save(b, format="JPEG", quality=qualite)
    return b.getvalue()


def ecrire_image(chemin: str | Path, a: np.ndarray, format_: str = "jpg") -> int:
    """Ecrit l'image et renvoie sa taille en octets."""
    octets = encoder_image(a, format_)
    ecrire_octets(chemin, octets)
    return len(octets)
