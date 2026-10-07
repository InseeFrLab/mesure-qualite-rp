"""Trouver les zones manuscrites (nom, prenom) sur un bulletin : six methodes comparables.

Toutes les zones sont en fractions de la page : (x, y, largeur, hauteur).
Une methode renvoie {"NOM": zone ou None, "PRENOM": zone ou None} ; None = zone non trouvee.

1. gabarit         : positions fixes, toujours au meme endroit
2. encre_bloc      : encre (OpenCV) dans le bloc identite ; les deux lignes d'ecriture
                     les plus marquees, de haut en bas = NOM puis PRENOM
3. encre_gabarit   : le gabarit donne une fenetre elargie, l'encre trouve la ligne dedans
4. modele          : un detecteur (TeleOCR, PaddleOCR) propose des rectangles ; on garde
                     celui qui tombe dans la fenetre du gabarit
5. modele_encre    : 4, puis l'encre ajuste les bords pour ne couper aucune lettre
(PaddleOCR seul et PaddleOCR + encre suivent 4 et 5.)

Pour les methodes 4 et 5, le gabarit ne sert qu'a CHOISIR parmi les rectangles du modele
(lequel est le nom, lequel est le prenom) : c'est dit tel quel dans les resultats.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

CHAMPS = ("NOM", "PRENOM")


# ----------------------------------------------------------------------------- outils

def decouper(page: Image.Image, zone):
    """Decoupe une zone (fractions) ; None si la zone est absente ou vide."""
    if zone is None:
        return None
    L, H = page.size
    x, y, w, h = zone
    boite = (int(max(0, x) * L), int(max(0, y) * H), int(min(1, x + w) * L), int(min(1, y + h) * H))
    if boite[2] - boite[0] < 5 or boite[3] - boite[1] < 5:
        return None
    return page.crop(boite)


def elargir(zone, dx: float, dy: float):
    """Fenetre de recherche autour d'une zone : dx, dy en fractions de page."""
    x, y, w, h = zone
    x0, y0 = max(0.0, x - dx), max(0.0, y - dy)
    x1, y1 = min(1.0, x + w + dx), min(1.0, y + h + dy)
    return (x0, y0, x1 - x0, y1 - y0)


def encre(image: Image.Image) -> np.ndarray:
    """Masque des pixels d'ecriture (seuillage adaptatif, comme la detection OpenCV du 01)."""
    gris = np.asarray(image.convert("L"))
    try:
        import cv2
        return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 10) > 0
    except ImportError:                       # repli sans OpenCV
        return gris < 140


def _bandes(masque: np.ndarray, part_min: float = 0.004, ecart_max: int = 3):
    """Bandes horizontales d'ecriture : groupes de lignes de pixels contenant de l'encre."""
    profil = masque.mean(axis=1)
    lignes = np.where(profil > part_min)[0]
    if not len(lignes):
        return []
    bandes, debut, prec = [], lignes[0], lignes[0]
    for r in lignes[1:]:
        if r - prec > ecart_max:
            bandes.append((debut, prec + 1))
            debut = r
        prec = r
    bandes.append((debut, prec + 1))
    # poids = encre totale de la bande
    return [(a, b, float(masque[a:b].sum())) for a, b in bandes if b - a >= 3]


def _cadrer(masque: np.ndarray, a: int, b: int, marge_px: int = 4):
    """Rectangle (pixels) de l'encre entre les lignes a et b."""
    cols = np.where(masque[a:b].any(axis=0))[0]
    if not len(cols):
        return None
    H, W = masque.shape
    return (max(0, cols[0] - marge_px), max(0, a - marge_px),
            min(W, cols[-1] + 1 + marge_px), min(H, b + marge_px))


def _en_fractions(boite_px, fenetre, page_size):
    L, H = page_size
    fx, fy = fenetre[0] * L, fenetre[1] * H
    x0, y0, x1, y1 = boite_px
    return (float((fx + x0) / L), float((fy + y0) / H), float((x1 - x0) / L), float((y1 - y0) / H))


# ----------------------------------------------------------------------------- methodes

def gabarit(page, zones):
    """1. Positions fixes."""
    return {c: zones.get(c) for c in CHAMPS}


def encre_gabarit(page, zones, dx=0.02, dy=0.012):
    """3. Fenetre elargie autour du gabarit ; la bande d'encre la plus proche de son centre."""
    res = {}
    for c in CHAMPS:
        if c not in zones:
            res[c] = None
            continue
        fen = elargir(zones[c], dx, dy)
        vign = decouper(page, fen)
        if vign is None:
            res[c] = None
            continue
        m = encre(vign)
        bandes = _bandes(m)
        if not bandes:
            res[c] = None
            continue
        centre = m.shape[0] / 2
        a, b, _ = min(bandes, key=lambda t: abs((t[0] + t[1]) / 2 - centre) - 1e-6 * t[2])
        boite = _cadrer(m, a, b)
        res[c] = _en_fractions(boite, fen, page.size) if boite else None
    return res


def encre_bloc(page, zone_bloc, colonne_libelle: float = 0.21):
    """2. Encre dans le bloc identite : les deux bandes les plus chargees, de haut en bas.

    colonne_libelle : part gauche du bloc ignoree (libelles imprimes « Nom : », « Prenom : »).
    """
    vign = decouper(page, zone_bloc)
    if vign is None:
        return {c: None for c in CHAMPS}
    m = encre(vign)
    m[:, : int(m.shape[1] * colonne_libelle)] = False
    bandes = sorted(_bandes(m), key=lambda t: -t[2])[:2]
    bandes = sorted(bandes, key=lambda t: t[0])
    res = {c: None for c in CHAMPS}
    for c, (a, b, _) in zip(CHAMPS, bandes):
        boite = _cadrer(m, a, b)
        res[c] = _en_fractions(boite, zone_bloc, page.size) if boite else None
    return res


def choisir(boites, zones, dx=0.04, dy=0.012):
    """4. Parmi les rectangles d'un detecteur (x1, y1, x2, y2 en fractions), garde pour
    chaque champ celui dont le centre est dans la fenetre du gabarit et le plus proche du sien."""
    res = {}
    for c in CHAMPS:
        if c not in zones or not boites:
            res[c] = None
            continue
        fx, fy, fw, fh = elargir(zones[c], dx, dy)
        zx, zy, zw, zh = zones[c]
        cz = (zx + zw / 2, zy + zh / 2)
        dedans = [(x1, y1, x2, y2) for x1, y1, x2, y2 in boites
                  if fx <= (x1 + x2) / 2 <= fx + fw and fy <= (y1 + y2) / 2 <= fy + fh]
        if not dedans:
            res[c] = None
            continue
        x1, y1, x2, y2 = min(dedans, key=lambda b: math.dist(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), cz))
        res[c] = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
    return res


def raffiner(page, zones_trouvees, dx=0.01, dy=0.004):
    """5. Ajuste une zone trouvee a l'encre qu'elle contient (sans couper de lettre)."""
    res = {}
    for c, z in zones_trouvees.items():
        if z is None:
            res[c] = None
            continue
        fen = elargir(z, dx, dy)
        vign = decouper(page, fen)
        if vign is None:
            res[c] = z
            continue
        m = encre(vign)
        bandes = _bandes(m)
        if not bandes:
            res[c] = z
            continue
        a, b, _ = max(bandes, key=lambda t: t[2])
        boite = _cadrer(m, a, b)
        res[c] = _en_fractions(boite, fen, page.size) if boite else z
    return res


# ----------------------------------------------------------------------------- robustesse

def perturber(page: Image.Image, graine: int, decalage: float = 0.01, angle: float = 1.0):
    """Simule un scan de travers : decalage (fraction de page) et rotation (degres), aleatoires
    mais reproductibles (graine). Fond blanc."""
    r = random.Random(graine)
    dx = r.uniform(-decalage, decalage) * page.size[0]
    dy = r.uniform(-decalage, decalage) * page.size[1]
    a = r.uniform(-angle, angle)
    tournee = page.rotate(a, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return tournee.transform(tournee.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                             resample=Image.BICUBIC, fillcolor=(255, 255, 255))


# ----------------------------------------------------------------------------- geometrie

def iou(a, b) -> float:
    """Recouvrement de deux zones (x, y, l, h) : 0 = disjointes, 1 = identiques."""
    if a is None or b is None:
        return 0.0
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter = max(0.0, min(ax2, bx2) - max(a[0], b[0])) * max(0.0, min(ay2, by2) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def ecriture_coupee(page, vraie, trouvee) -> float:
    """Part de l'encre de la vraie zone qui tombe HORS de la zone trouvee (0 = rien de coupe)."""
    if vraie is None:
        return float("nan")
    if trouvee is None:
        return 1.0
    v = decouper(page, vraie)
    if v is None:
        return float("nan")
    m = encre(v)
    total = m.sum()
    if total == 0:
        return 0.0
    L, H = page.size
    vx, vy = vraie[0] * L, vraie[1] * H
    tx0, ty0 = trouvee[0] * L - vx, trouvee[1] * H - vy
    tx1, ty1 = tx0 + trouvee[2] * L, ty0 + trouvee[3] * H
    yy, xx = np.nonzero(m)
    dedans = (xx >= tx0) & (xx < tx1) & (yy >= ty0) & (yy < ty1)
    return float(1 - dedans.sum() / total)


# ----------------------------------------------------------------------------- PaddleOCR

SCRIPT_PADDLE = r'''
import json, sys
from paddleocr import TextDetection
modele = TextDetection(model_name=sys.argv[3] if len(sys.argv) > 3 else "PP-OCRv5_server_det")
pages = json.load(open(sys.argv[1]))
sortie = {}
for cle, chemin in pages.items():
    res = modele.predict(chemin)
    boites = []
    for r in res:
        d = r.json if hasattr(r, "json") else r
        d = d.get("res", d)
        for poly in d.get("dt_polys", []):
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            boites.append([min(xs), min(ys), max(xs), max(ys)])
    sortie[cle] = boites
json.dump(sortie, open(sys.argv[2], "w"))
'''


def paddle_detecter(pages: dict, travail: Path, python_paddle: Path, modele="PP-OCRv5_server_det"):
    """Detection de texte PaddleOCR, lancee dans SON environnement (comme PyLaia).

    pages : {cle: chemin d'une image}. Renvoie {cle: [(x1, y1, x2, y2) en pixels]}.
    """
    travail = Path(travail)
    travail.mkdir(parents=True, exist_ok=True)
    (travail / "paddle_detecter.py").write_text(SCRIPT_PADDLE, encoding="utf-8")
    (travail / "pages.json").write_text(json.dumps({k: str(v) for k, v in pages.items()}))
    r = subprocess.run([str(python_paddle), str(travail / "paddle_detecter.py"),
                        str(travail / "pages.json"), str(travail / "boites.json"), modele],
                       capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        lignes = [x for x in r.stderr.splitlines() if x.strip()]
        raise RuntimeError("PaddleOCR en echec :\n" + "\n".join(lignes[-6:]))
    return {k: [tuple(b) for b in v] for k, v in json.loads((travail / "boites.json").read_text()).items()}
