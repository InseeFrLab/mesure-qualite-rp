"""Scans simules, pour mesurer la precision du recalage sans aucune donnee reelle.

On ecrit un texte invente dans les champs de la page vierge, puis on applique a chaque
unite une transformation CONNUE T (rotation, zoom, decalage), du bruit et une
compression JPEG. Le recalage estime M ; l'erreur est l'ecart entre la position vraie
et la position recalee des coins des blocs :
    e_max = max_k || M (T p_k) - p_k ||        (pixels de la reference)
    e_mm  = e_max x 25,4 / dpi
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ._outils import police


def faux_scan(cfg, refs, echelle=0.75, angle=1.0, mode="couleur", graine=0):
    """Un scan SIMULE (aucune donnee reelle) : ecriture inventee dans les champs, chaque unite tournee, mise a
    l'echelle, decalee, bruitee, compressee en JPEG. Renvoie (image, transformations reference -> scan par unite)."""
    import io as _io
    import random

    import cv2

    rng = random.Random(graine)
    W, H = cfg["reference"]["largeur"], cfg["reference"]["hauteur"]
    pages = [Image.fromarray(r.copy()) for r in refs]
    textes = [
        "12 BIS RUE DES LILAS",
        "BAT C ESC 2",
        "DUPONT",
        "MARIE",
        "8 AV DE LA GARE",
        "LYON",
        "1984",
    ]
    for c in cfg["champs"]:
        x, y, w, h = c["zone"]
        ui = c["unite"] - 1
        y0 = (y - cfg["pages"][ui][0]) * H
        d = ImageDraw.Draw(pages[ui])
        if c["type"] in ("texte", "ligne"):
            d.text(
                (x * W + 8, y0 + 2),
                rng.choice(textes),
                fill=(20, 30, 120),
                font=police(max(10, int(h * H * 0.65))),
            )
        elif c["type"] == "case" and rng.random() < 0.33:
            d.line(
                [(x * W, y0), ((x + w) * W, y0 + h * H)], fill=(20, 30, 120), width=3
            )
    scans, T = [], []
    for i, p in enumerate(pages):
        a = np.array(p)
        an = np.radians(angle + 0.3 * i)
        M = np.array(
            [
                [echelle * np.cos(an), -echelle * np.sin(an), 25 + 7 * i],
                [echelle * np.sin(an), echelle * np.cos(an), 18],
            ]
        )
        o = cv2.warpAffine(
            a,
            M,
            (int(a.shape[1] * echelle) + 50, int(a.shape[0] * echelle) + 50),
            borderValue=(250, 250, 250),
        )
        if mode == "gris":
            o = cv2.cvtColor(cv2.cvtColor(o, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        elif mode == "noir_blanc":
            g = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
            o = cv2.cvtColor(
                np.where(g < 150, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
            )
        o = np.clip(
            o + np.random.default_rng(graine + i).normal(0, 6, o.shape), 0, 255
        ).astype(np.uint8)
        scans.append(o)
        T.append(M)
    larg = max(s.shape[1] for s in scans)
    img = np.vstack(
        [
            np.pad(s, ((0, 0), (0, larg - s.shape[1]), (0, 0)), constant_values=250)
            for s in scans
        ]
    )
    b = _io.BytesIO()
    Image.fromarray(img).save(b, "JPEG", quality=60)
    return np.array(Image.open(b).convert("RGB")), T, [s.shape[0] for s in scans]


def erreur_recalage(cfg, infos, T, hauteurs):
    """Ecart maximal (pixels de la reference) entre la position vraie et la position recalee des coins des blocs."""
    W, H = cfg["reference"]["largeur"], cfg["reference"]["hauteur"]
    debut = np.cumsum([0] + list(hauteurs))
    errs = []
    for i, inf in enumerate(infos):
        if not inf["recalee"]:
            return float("inf")
        for b in cfg["blocs"]:
            if b["unite"] != i + 1:
                continue
            x, y, w, h = b["zone"]
            for px, py in [(x, y), (x + w, y + h)]:
                P = np.array([px * W, (py - cfg["pages"][i][0]) * H, 1.0])
                s_ = T[i] @ P
                Q = np.array([s_[0], s_[1] + debut[i] - inf["_debut"], 1.0])
                errs.append(float(np.hypot(*(inf["_M"] @ Q - P[:2]))))
    return max(errs) if errs else float("nan")


def erreur_mm(cfg, erreur_px):
    """Erreur de recalage convertie en millimetres (1 px = 25,4 / dpi mm)."""
    return erreur_px * 25.4 / cfg["reference"]["dpi"]
