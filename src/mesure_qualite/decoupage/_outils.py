"""Outils communs du decoupage : texte, rectangles, polices.

Conventions :
- un rectangle « bb » est un quadruplet (x0, haut, x1, bas), dans le repere de la page
  PDF (points) ou de l'image (pixels) ;
- un objet pdfplumber (mot, rectangle, trait) est un dictionnaire avec les cles
  « x0 », « x1 », « top », « bottom ».
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from PIL import ImageFont

Rect = tuple[float, float, float, float]

GRAS = ("Bold", "Black", "Heavy", "Semibold", "SemiBold", "ExtraBold", "Demi")


# --- texte -------------------------------------------------------------------------
def normaliser(t: object) -> str:
    """Majuscules, sans accents ni ponctuation, espaces simples."""
    t = unicodedata.normalize("NFKD", str(t or "").strip())
    t = "".join(c for c in t if not unicodedata.combining(c)).upper()
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", t)).strip()


def slug(t: object, n: int = 28) -> str:
    """Identifiant court en minuscules, sans accents (ex. « cadre_nom »)."""
    s = re.sub(r"_+", "_", normaliser(t).lower().replace(" ", "_")).strip("_")
    return s[:n].strip("_") or "x"


def est_gras(w: dict) -> bool:
    return any(g in w["fontname"] for g in GRAS)


def est_italique(w: dict) -> bool:
    return "Italic" in w["fontname"] or "Oblique" in w["fontname"]


def famille(fontname: str) -> str:
    """Famille de police sans le prefixe de sous-ensemble (ABCDEF+Arial-Bold -> Arial)."""
    f = fontname.split("+")[-1]
    return re.split(r"[-,]", f)[0].replace("MT", "")


def lignes_de(
    mots: Iterable[dict], tol: float = 2.5, ecart_max: float = 28
) -> list[list[dict]]:
    """Regroupe des mots en lignes (meme hauteur), de haut en bas, chaque ligne de gauche
    a droite. Un grand blanc horizontal (> ecart_max pt) coupe la ligne : deux colonnes
    ne sont pas melangees."""
    lignes: list[list[dict]] = []
    # les grands mots d'abord : ils fixent la hauteur de la ligne
    for w in sorted(mots, key=lambda w: (-(w["bottom"] - w["top"]), w["top"], w["x0"])):
        cy = (w["top"] + w["bottom"]) / 2
        for ligne in lignes:
            ref = ligne[0]
            if ref["top"] - 1 <= cy <= ref["bottom"] + 1 and abs(
                ref["top"] - w["top"]
            ) < max(tol, 0.6 * (ref["bottom"] - ref["top"])):
                ligne.append(w)
                break
        else:
            lignes.append([w])
    lignes = [sorted(ligne, key=lambda w: w["x0"]) for ligne in lignes]
    coupees = []
    for ligne in lignes:
        cour = [ligne[0]]
        for w in ligne[1:]:
            if w["x0"] - cour[-1]["x1"] > ecart_max:
                coupees.append(cour)
                cour = [w]
            else:
                cour.append(w)
        coupees.append(cour)
    return sorted(coupees, key=lambda ligne: (ligne[0]["top"], ligne[0]["x0"]))


def texte_de(mots: Iterable[dict]) -> str:
    """Le texte d'une suite de mots, avec la ponctuation et les traits d'union recolles."""
    t = " ".join(w["text"] for w in mots)
    t = re.sub(r"(\w)- (\w)", r"\1-\2", t)
    t = re.sub(r"\b(\d+) (er|re|e|ème|eme)\b", r"\1\2", t)
    return re.sub(r"\s+", " ", t).replace(" ,", ",").replace(" .", ".").strip()


# --- rectangles ----------------------------------------------------------------------
def centre(o: dict) -> tuple[float, float]:
    return ((o["x0"] + o["x1"]) / 2, (o["top"] + o["bottom"]) / 2)


def dedans(pt: tuple[float, float], r: Sequence[float], tol: float = 0.0) -> bool:
    return r[0] - tol <= pt[0] <= r[2] + tol and r[1] - tol <= pt[1] <= r[3] + tol


def contient(a: Sequence[float], b: Sequence[float], tol: float = 0.5) -> bool:
    """Le rectangle a contient-il le rectangle b ?"""
    return (
        a[0] - tol <= b[0]
        and a[1] - tol <= b[1]
        and b[2] <= a[2] + tol
        and b[3] <= a[3] + tol
    )


def recouvre(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def union(rs: Iterable[Sequence[float]]) -> Rect:
    rs = list(rs)
    return (
        min(r[0] for r in rs),
        min(r[1] for r in rs),
        max(r[2] for r in rs),
        max(r[3] for r in rs),
    )


def surface(r: Sequence[float]) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def bb(o: dict) -> Rect:
    return (float(o["x0"]), float(o["top"]), float(o["x1"]), float(o["bottom"]))


def iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Recouvrement de deux rectangles (x0, y0, x1, y1) : 1 = identiques, 0 = disjoints.

    IoU = surface(a inter b) / surface(a union b). NaN si l'un des deux manque."""
    if a is None or b is None:
        return float("nan")
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = surface(a) + surface(b) - inter
    return inter / u if u > 0 else 0.0


# --- polices (images de controle, scans simules) -----------------------------------
def police(taille: int) -> ImageFont.ImageFont:
    for f in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(f, taille)
        except OSError:
            pass
    return ImageFont.load_default()
