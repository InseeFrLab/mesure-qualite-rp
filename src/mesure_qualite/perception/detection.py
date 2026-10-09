"""Trouver les zones manuscrites (nom, prenom) sur un bulletin : methodes comparables.

Toutes les zones sont en fractions de la page : (x, y, largeur, hauteur).
Une methode renvoie {"NOM": zone ou None, "PRENOM": zone ou None} ; None = zone non trouvee.

1. gabarit         : positions fixes, toujours au meme endroit
2. encre_bloc      : encre dans le bloc identite ; les deux lignes d'ecriture les plus marquees,
                     de haut en bas = NOM puis PRENOM (ne depend pas des positions exactes)
3. encre_gabarit   : le gabarit donne une fenetre autour du champ, l'encre trouve la ligne dedans
4. modele          : un detecteur (TeleOCR, PaddleOCR) propose des rectangles ; on reunit ceux
                     qui tombent sur la ligne du champ
5. modele_encre    : 4, puis l'encre ajuste les bords pour ne couper aucune lettre
6. recalage        : la page est alignee sur un bulletin modele (points ORB, homographie),
                     puis les positions fixes sont reportees sur la page : robuste au scan de travers
7. recalage_encre  : 6, puis l'encre ajuste les bords

Pour les methodes 4 a 7, le gabarit ne sert qu'a CHOISIR (lequel est le nom, lequel est le prenom).

Version 2 (corrige les decoupes trop larges de la version 1, CER > 100 %) :
- l'encre est seuillee par Otsu, et les traits imprimes du formulaire (lignes, cadres) sont retires ;
- deux lignes d'ecriture collees ne forment plus une seule bande (elles sont scindees) ;
- les fenetres de recherche sont proportionnelles a la hauteur du champ (elles n'atteignent plus
  le champ voisin) et ne debordent presque pas a gauche (libelles imprimes) ;
- un detecteur qui coupe un nom en plusieurs mots : les mots de la ligne sont reunis.
"""
from __future__ import annotations

import json
import math
import os
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


def elargir(zone, dx: float, dy: float, dx_droite: float | None = None):
    """Fenetre de recherche autour d'une zone (fractions de page). dx a gauche, dx_droite a droite
    (par defaut egal a dx), dy en haut et en bas."""
    x, y, w, h = zone
    dxd = dx if dx_droite is None else dx_droite
    x0, y0 = max(0.0, x - dx), max(0.0, y - dy)
    x1, y1 = min(1.0, x + w + dxd), min(1.0, y + h + dy)
    return (x0, y0, x1 - x0, y1 - y0)


def _otsu(gris: np.ndarray) -> int:
    hist = np.bincount(gris.ravel(), minlength=256).astype(float)
    p = hist / max(hist.sum(), 1)
    w, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    return int(np.argmax((mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-9)))


def retirer_traits(masque: np.ndarray) -> np.ndarray:
    """Retire les traits imprimes du formulaire : lignes horizontales et verticales longues."""
    m = masque.copy()
    H, W = m.shape
    try:
        import cv2
        m8 = m.astype(np.uint8)
        horiz = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, W // 4), 1)))
        vert = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, H // 2))))
        m &= ~(horiz.astype(bool) | vert.astype(bool))
    except (ImportError, AttributeError):            # OpenCV absent ou incomplet : repli numpy seul
        pass
    m[m.mean(axis=1) > 0.5, :] = False              # lignes presque pleines
    m[:, m.mean(axis=0) > 0.6] = False              # colonnes presque pleines
    return m


def encre(image: Image.Image, traits: bool = True) -> np.ndarray:
    """Masque des pixels d'ecriture : seuil d'Otsu (l'encre est bien plus sombre que le papier),
    puis retrait des traits imprimes. Le seuillage adaptatif de la version 1 captait aussi la
    trame et le texte imprime, d'ou des decoupes trop larges."""
    gris = np.asarray(image.convert("L"))
    if gris.size == 0:
        return np.zeros((0, 0), bool)
    t = _otsu(gris)
    m = gris <= t
    if m.mean() > 0.45:                              # image presque vide : Otsu separe du bruit
        m = gris < 140
    return retirer_traits(m) if traits else m


def _bandes(masque: np.ndarray, part_min: float = 0.004, ecart_max: int = 3):
    """Bandes horizontales d'ecriture : groupes de lignes de pixels contenant de l'encre.
    Renvoie [(debut, fin, poids)], poids = encre totale de la bande."""
    if masque.size == 0:
        return []
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
    return [(a, b, float(masque[a:b].sum())) for a, b in bandes if b - a >= 3]


def _scinder(masque: np.ndarray, bandes, hauteur_ligne: float, facteur: float = 1.6):
    """Coupe en deux une bande trop haute (deux lignes d'ecriture collees), au creux du profil."""
    if hauteur_ligne <= 0:
        return bandes
    res = []
    for a, b, poids in bandes:
        if b - a > facteur * hauteur_ligne:
            profil = masque[a:b].mean(axis=1)
            q = (b - a) // 4
            coupe = a + q + int(np.argmin(profil[q:(b - a) - q])) if (b - a) - 2 * q > 0 else (a + b) // 2
            for x, y in [(a, coupe), (coupe, b)]:
                if y - x >= 3:
                    res.append((x, y, float(masque[x:y].sum())))
        else:
            res.append((a, b, poids))
    return res


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


def _recouvrement(a, b, c, d) -> int:
    return max(0, min(b, d) - max(a, c))


def _ligne_dans(page, fen, cible_px, hauteur_px):
    """Dans la fenetre fen (fractions), la bande d'encre qui recouvre le mieux la bande cible
    (lignes cible_px[0]..cible_px[1] de la fenetre), cadree. Renvoie une zone (fractions) ou None."""
    vign = decouper(page, fen)
    if vign is None:
        return None
    m = encre(vign)
    bandes = _scinder(m, _bandes(m), hauteur_px)
    if not bandes:
        return None
    a, b, _ = max(bandes, key=lambda t: (_recouvrement(t[0], t[1], *cible_px), t[2]))
    if _recouvrement(a, b, *cible_px) == 0:
        return None
    boite = _cadrer(m, a, b)
    return _en_fractions(boite, fen, page.size) if boite else None


# ----------------------------------------------------------------------------- methodes

def gabarit(page, zones):
    """1. Positions fixes."""
    return {c: zones.get(c) for c in CHAMPS}


def encre_gabarit(page, zones, marge_g=0.005, marge_d=0.05, marge_v=0.5):
    """3. Fenetre autour du gabarit, la ligne d'encre qui recouvre le mieux la position attendue.

    marge_v : en hauteurs de champ (0.5 = une demi-ligne au-dessus et en dessous : la fenetre
    n'atteint pas le centre du champ voisin) ; marge_g faible : le libelle imprime est a gauche."""
    H = page.size[1]
    res = {}
    for c in CHAMPS:
        if c not in zones:
            res[c] = None
            continue
        x, y, w, h = zones[c]
        fen = elargir(zones[c], marge_g, marge_v * h, dx_droite=marge_d)
        haut = (y - fen[1]) * H
        res[c] = _ligne_dans(page, fen, (int(haut), int(haut + h * H)), h * H)
    return res


def encre_bloc(page, zone_bloc, colonne_libelle: float = 0.21, lignes_attendues: int = 3):
    """2. Encre dans le bloc identite : les deux bandes les plus chargees, de haut en bas.

    colonne_libelle : part gauche du bloc ignoree (libelles imprimes « Nom : », « Prenom : »).
    lignes_attendues : nombre de lignes du bloc, pour estimer la hauteur d'une ligne d'ecriture."""
    vign = decouper(page, zone_bloc)
    if vign is None:
        return {c: None for c in CHAMPS}
    m = encre(vign)
    m[:, : int(m.shape[1] * colonne_libelle)] = False
    h_ligne = m.shape[0] / lignes_attendues
    bandes = [t for t in _scinder(m, _bandes(m), h_ligne)
              if 0.25 * h_ligne <= t[1] - t[0] <= 1.6 * h_ligne]        # hauteur plausible d'une ligne
    bandes = sorted(sorted(bandes, key=lambda t: -t[2])[:2], key=lambda t: t[0])
    res = {c: None for c in CHAMPS}
    for c, (a, b, _) in zip(CHAMPS, bandes):
        boite = _cadrer(m, a, b)
        res[c] = _en_fractions(boite, zone_bloc, page.size) if boite else None
    return res


def choisir(boites, zones, marge_g=0.02, marge_d=0.06, tolerance_v=0.45, rogner_v=0.3):
    """4. Parmi les rectangles d'un detecteur (x1, y1, x2, y2 en fractions), reunit pour chaque
    champ ceux qui tombent sur sa ligne : centre horizontal dans la fenetre, centre vertical a
    moins de tolerance_v hauteurs de champ du centre attendu. Un nom ecrit en plusieurs mots
    (plusieurs boites) est ainsi garde entier. Si la reunion est bien plus haute qu'une ligne
    (detecteur de mise en page qui regroupe nom et prenom), elle est recoupee sur la ligne du champ."""
    res = {}
    for c in CHAMPS:
        if c not in zones or not boites:
            res[c] = None
            continue
        zx, zy, zw, zh = zones[c]
        x_min, x_max = zx - marge_g, zx + zw + marge_d
        cy = zy + zh / 2
        dedans = []
        for x1, y1, x2, y2 in boites:
            bx, by = (x1 + x2) / 2, (y1 + y2) / 2
            haute = (y2 - y1) > 1.5 * zh
            sur_ligne = (y1 <= cy <= y2) if haute else abs(by - cy) <= tolerance_v * zh
            if x_min <= bx <= x_max and sur_ligne:
                dedans.append((x1, y1, x2, y2))
        if not dedans:
            res[c] = None
            continue
        x1, y1 = min(b[0] for b in dedans), min(b[1] for b in dedans)
        x2, y2 = max(b[2] for b in dedans), max(b[3] for b in dedans)
        if y2 - y1 > 1.5 * zh:
            y1, y2 = max(y1, zy - rogner_v * zh), min(y2, zy + zh + rogner_v * zh)
        res[c] = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
    return res


def raffiner(page, zones_trouvees, marge_x=0.01, marge_v=0.3):
    """5. Ajuste une zone trouvee a l'encre qu'elle contient (sans couper de lettre) : la ligne
    d'encre qui recouvre le mieux la zone, dans une fenetre a peine plus grande."""
    H = page.size[1]
    res = {}
    for c, z in zones_trouvees.items():
        if z is None:
            res[c] = None
            continue
        x, y, w, h = z
        fen = elargir(z, marge_x, marge_v * h)
        haut = (y - fen[1]) * H
        trouvee = _ligne_dans(page, fen, (int(haut), int(haut + h * H)), h * H)
        res[c] = trouvee or z
    return res


# ----------------------------------------------------------------------------- recalage

class Recaleur:
    """Aligne une page sur un bulletin modele (points ORB, homographie RANSAC) et reporte les
    positions fixes du modele sur la page. Robuste aux decalages, rotations et changements
    d'echelle du scan. Necessite OpenCV."""

    def __init__(self, page_modele: Image.Image, largeur: int = 1600, n_points: int = 5000):
        import cv2
        if not hasattr(cv2, "ORB_create"):
            raise ImportError("OpenCV incomplet dans cet environnement : reinstaller opencv-python-headless")
        self.cv2, self.largeur = cv2, largeur
        self.orb = cv2.ORB_create(n_points)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.taille = page_modele.size
        self.kp, self.des, self.s = self._points(page_modele)

    def _points(self, page):
        cv2 = self.cv2
        gris = cv2.cvtColor(np.asarray(page.convert("RGB")), cv2.COLOR_RGB2GRAY)
        s = self.largeur / gris.shape[1]
        kp, des = self.orb.detectAndCompute(cv2.resize(gris, None, fx=s, fy=s), None)
        return kp, des, s

    def homographie(self, page):
        """Matrice qui envoie les pixels du MODELE sur ceux de la PAGE, ou None si l'alignement echoue."""
        cv2 = self.cv2
        kp, des, s = self._points(page)
        if des is None or self.des is None:
            return None
        paires = [m for m, n in (x for x in self.bf.knnMatch(self.des, des, k=2) if len(x) == 2)
                  if m.distance < 0.75 * n.distance]
        if len(paires) < 30:
            return None
        src = np.float32([self.kp[m.queryIdx].pt for m in paires]) / self.s
        dst = np.float32([kp[m.trainIdx].pt for m in paires]) / s
        Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if Hm is None or inl.sum() < 20:
            return None
        return Hm

    def zones(self, page, zones_modele):
        """Positions fixes (fractions du modele) reportees sur la page (fractions de la page)."""
        Hm = self.homographie(page)
        if Hm is None:
            return {c: None for c in CHAMPS}
        Lm, Hmod = self.taille
        L, H = page.size
        res = {}
        for c in CHAMPS:
            if c not in zones_modele:
                res[c] = None
                continue
            x, y, w, h = zones_modele[c]
            coins = np.float32([[x * Lm, y * Hmod], [(x + w) * Lm, y * Hmod],
                                [(x + w) * Lm, (y + h) * Hmod], [x * Lm, (y + h) * Hmod]]).reshape(-1, 1, 2)
            p = self.cv2.perspectiveTransform(coins, Hm).reshape(-1, 2)
            x0, y0 = p[:, 0].min() / L, p[:, 1].min() / H
            x1, y1 = p[:, 0].max() / L, p[:, 1].max() / H
            res[c] = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
        return res


def recalage(page, recaleur: Recaleur, zones):
    """6. Positions fixes reportees apres alignement de la page sur le modele."""
    return recaleur.zones(page, zones)


def recalage_encre(page, recaleur: Recaleur, zones):
    """7. Recalage, puis ajustement a l'encre."""
    return raffiner(page, recaleur.zones(page, zones))


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
#
# PaddleOCR tourne dans son propre environnement (comme PyLaia). Deux pieges connus sur ce service :
# - OpenCV « graphique » exige libGL, absent : l'environnement doit utiliser la version headless
#   (voir preparer_env_paddle) ;
# - Paddle 3.x sur CPU echoue avec oneDNN (« ConvertPirAttribute2RuntimeAttribute not support ») :
#   le modele est cree avec enable_mkldnn=False.

SCRIPT_PADDLE = r'''
import json, os, sys
os.environ.setdefault("FLAGS_use_mkldnn", "0")
from paddleocr import TextDetection
nom = sys.argv[3] if len(sys.argv) > 3 else "PP-OCRv5_server_det"
# oneDNN desactive : evite l'erreur 'ConvertPirAttribute2RuntimeAttribute' de Paddle 3.x sur CPU
try:
    modele = TextDetection(model_name=nom, device="cpu", enable_mkldnn=False)
except TypeError:                         # version de paddleocr sans l'option enable_mkldnn
    modele = TextDetection(model_name=nom, device="cpu")
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


def preparer_env_paddle(dossier="/tmp/paddle-env", version_paddle: str | None = None) -> Path:
    """Cree (si besoin) l'environnement PaddleOCR, et REPARE OpenCV a chaque appel s'il le faut.

    - Python fourni par uv (avec ses en-tetes), pas celui du systeme ;
    - OpenCV sans ecran : PaddleX verifie la presence du paquet nomme « opencv-contrib-python » ;
      on l'installe sans dependances, puis on remet la version headless par-dessus (memes
      fichiers cv2, sans libGL). Toute reinstallation de paddleocr remet la version « graphique » :
      la reparation est donc verifiee et refaite a chaque appel (quelques secondes).
    version_paddle : par exemple "3.0.0" pour figer Paddle si enable_mkldnn=False ne suffit pas.
    """
    env = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "UV_LINK_MODE": "copy",
           "UV_PYTHON_INSTALL_DIR": "/tmp/uv-python"}
    py = Path(dossier) / "bin" / "python"

    def sh(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd}\n{r.stderr[-600:]}")

    def ok(test):
        return py.exists() and subprocess.run([str(py), "-c", test], capture_output=True).returncode == 0

    if not ok("import paddleocr"):                      # environnement absent ou casse : on le refait
        paddle = f"paddlepaddle=={version_paddle}" if version_paddle else "paddlepaddle"
        sh(f"rm -rf {dossier}")
        sh(f"uv venv {dossier} --python 3.11 --python-preference only-managed")
        sh(f"uv pip install --python {py} {paddle} paddleocr")
    if not ok("import cv2; cv2.morphologyEx"):         # OpenCV « graphique » (libGL) : reparation
        sh(f"uv pip install --python {py} --no-deps opencv-contrib-python")
        sh(f"uv pip install --python {py} --no-deps --reinstall opencv-contrib-python-headless")
    if not ok("import cv2, paddleocr; cv2.morphologyEx"):
        raise RuntimeError(f"environnement PaddleOCR inutilisable : {dossier}")
    return py


def paddle_detecter(pages: dict, travail: Path, python_paddle: Path, modele="PP-OCRv5_server_det"):
    """Detection de texte PaddleOCR, lancee dans SON environnement (comme PyLaia).

    pages : {cle: chemin d'une image}. Renvoie {cle: [(x1, y1, x2, y2) en pixels]}.
    """
    travail = Path(travail)
    travail.mkdir(parents=True, exist_ok=True)
    preparer_env_paddle(Path(python_paddle).parent.parent)     # verifie / repare OpenCV (quelques s)
    (travail / "paddle_detecter.py").write_text(SCRIPT_PADDLE, encoding="utf-8")
    (travail / "pages.json").write_text(json.dumps({k: str(v) for k, v in pages.items()}))
    r = subprocess.run([str(python_paddle), str(travail / "paddle_detecter.py"),
                        str(travail / "pages.json"), str(travail / "boites.json"), modele],
                       capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "FLAGS_use_mkldnn": "0"})
    if r.returncode != 0:
        lignes = [x for x in r.stderr.splitlines() if x.strip()]
        raise RuntimeError("PaddleOCR en echec :\n" + "\n".join(lignes[-6:]))
    return {k: [tuple(b) for b in v] for k, v in json.loads((travail / "boites.json").read_text()).items()}