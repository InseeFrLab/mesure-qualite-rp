"""Recalage d'un scan sur les images de reference du gabarit, puis decoupe.

Modele : chaque unite du scan se deduit de l'unite vierge par une transformation affine
    q = A p + t     (A : rotation, zoom, etirement ; t : decalage ; 6 parametres).
Estimation, sans aucun modele appris (OpenCV) :
    1. niveaux de gris, reduction a LARGEUR_ORB pixels, egalisation locale (CLAHE) ;
    2. points caracteristiques ORB (FAST + BRIEF oriente), 6 000 par image ;
    3. appariement par distance de Hamming, test du rapport de Lowe (d1 < 0,8 d2) ;
    4. estimation robuste par RANSAC (seuil 3 px, confiance 0,999), puis moindres carres
       sur les paires retenues ;
    5. retour a la pleine resolution : A <- (s / s_ref) A, t <- t / s_ref ;
    6. controles : au moins MIN_POINTS paires retenues, zoom dans ]0,75 ; 1,33[ du zoom
       attendu, rotation d'au plus 10 degres.
Une unite qui echoue n'est pas decoupee au hasard : elle est prise aux proportions et
signalee (infos["recalee"] = False).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import gabarit as _gabarit


class Recaleur:
    """Aligne chaque unite d'un scan sur l'unite vierge de reference, puis decoupe.

    cfg  : configuration du gabarit ;
    refs : images de reference des unites (RGB), dans l'ordre de cfg["unites"].
    """

    LARGEUR_ORB = 1100
    MIN_POINTS = 25

    def __init__(self, cfg: dict, refs: list[np.ndarray]):
        import cv2

        if len(refs) != len(cfg["unites"]):
            raise ValueError(
                f"{len(refs)} image(s) de reference pour {len(cfg['unites'])} unite(s)"
            )
        self.cv2, self.cfg, self.refs = cv2, cfg, refs
        self._ref: dict[int, dict] = {}

    @classmethod
    def depuis_dossier(cls, dossier: str | Path, nom: str) -> Recaleur:
        """Charge le gabarit « nom » (ex. "BI_2025") d'un dossier local ou s3://."""
        cfg, refs = _gabarit.charger(dossier, nom)
        return cls(cfg, refs)

    def _orb(self):
        return self.cv2.ORB_create(
            nfeatures=6000, scaleFactor=1.2, nlevels=8, fastThreshold=10
        )

    def _petit(self, g):
        cv2 = self.cv2
        s = self.LARGEUR_ORB / g.shape[1]
        p = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(p), s

    def _gris(self, a):
        return a if a.ndim == 2 else self.cv2.cvtColor(a, self.cv2.COLOR_RGB2GRAY)

    def ref(self, i):
        if i not in self._ref:
            g = self._gris(np.asarray(self.refs[i]))
            p, s = self._petit(g)
            kp, de = self._orb().detectAndCompute(p, None)
            self._ref[i] = {"forme": g.shape, "s": s, "kp": kp, "de": de}
        return self._ref[i]

    def recaler_unite(self, a, i):
        cv2 = self.cv2
        R = self.ref(i)
        g = self._gris(a)
        p, s = self._petit(g)
        kp, de = self._orb().detectAndCompute(p, None)
        infos = {"unite": i + 1, "points": 0, "retenus": 0}
        if de is None or R["de"] is None or len(kp) < self.MIN_POINTS:
            return None, infos
        paires = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(de, R["de"], k=2)
        bons = [
            m for m in paires if len(m) == 2 and m[0].distance < 0.8 * m[1].distance
        ]
        infos["points"] = len(bons)
        if len(bons) < self.MIN_POINTS:
            return None, infos
        src = np.float32([kp[m[0].queryIdx].pt for m in bons])
        dst = np.float32([R["kp"][m[0].trainIdx].pt for m in bons])
        A, garde = cv2.estimateAffine2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.999,
        )
        if A is None:
            return None, infos
        infos["retenus"] = int(garde.sum())
        M = A.astype(np.float64)
        M[:, :2] *= s / R["s"]
        M[:, 2] /= R["s"]
        attendu = R["forme"][1] / g.shape[1]
        sx, sy = np.hypot(M[0, 0], M[1, 0]), np.hypot(M[0, 1], M[1, 1])
        angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        infos.update(
            {
                "echelle_x": round(float(sx / attendu), 3),
                "echelle_y": round(float(sy / attendu), 3),
                "angle": round(angle, 2),
            }
        )
        if (
            infos["retenus"] < self.MIN_POINTS
            or not (0.75 < sx / attendu < 1.33)
            or not (0.75 < sy / attendu < 1.33)
            or abs(angle) > 10
        ):
            return None, infos
        return M, infos

    def bornes_unites(self, img):
        """(haut, bas) de chaque unite dans le scan empile : coupure dans la bande la plus blanche pres de la
        position attendue (proportions des unites de reference)."""
        g = self._gris(img)
        h = g.shape[0]
        fr = [p_[1] for p_ in self.cfg["pages"][:-1]]
        if not fr:
            return [(0, h)]
        encre = np.convolve((g < 200).mean(axis=1), np.ones(15) / 15, mode="same")
        coupes = []
        for f in fr:
            c = f * h
            lo, hi = int(c - 0.04 * h), int(c + 0.04 * h)
            coupes.append(
                lo
                + int(
                    np.argmin(
                        encre[lo:hi] + 0.02 * np.abs(np.arange(lo, hi) - c) / (0.04 * h)
                    )
                )
            )
        b = [0] + coupes + [h]
        return [(b[i], b[i + 1]) for i in range(len(b) - 1)]

    def toile(self, img):
        """Le scan entier ramene sur la reference (unites empilees, taille de la reference). Renvoie (toile, infos)."""
        cv2 = self.cv2
        img = np.asarray(img)
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        W, H = self.cfg["reference"]["largeur"], self.cfg["reference"]["hauteur"]
        t = np.full((H, W, 3), 255, np.uint8)
        infos = []
        for i, (a, b) in enumerate(self.bornes_unites(img)):
            A, B = (
                int(round(self.cfg["pages"][i][0] * H)),  # noqa: RUF046
                int(round(self.cfg["pages"][i][1] * H)),  # noqa: RUF046
            )
            m = int(0.04 * (b - a))
            a2, b2 = max(0, a - m), min(img.shape[0], b + m)
            M, inf = self.recaler_unite(img[a2:b2], i)
            if M is not None:
                t[A:B] = cv2.warpAffine(
                    img[a2:b2],
                    M,
                    (W, B - A),
                    flags=cv2.INTER_LINEAR,
                    borderValue=(255, 255, 255),
                )
                inf["recalee"] = True
                inf["_M"], inf["_debut"] = M, a2
            else:
                t[A:B] = cv2.resize(img[a:b], (W, B - A))
                inf["recalee"] = False
            infos.append(inf)
        return t, infos

    def decouper(self, toile, zone, marge=None):
        W, H = self.cfg["reference"]["largeur"], self.cfg["reference"]["hauteur"]
        marge = self.cfg.get("marge_defaut", 0.004) if marge is None else marge
        x, y, w, h = zone
        x0, y0 = max(0, x - marge), max(0, y - marge)
        x1, y1 = min(1, x + w + marge), min(1, y + h + marge)
        return toile[int(y0 * H) : int(y1 * H), int(x0 * W) : int(x1 * W)]


def zone_dans_scan(cfg, infos_unite, zone):
    """Rectangle (x0, y0, x1, y1) en pixels du SCAN d'origine d'une zone du gabarit, grace au recalage de son unite.
    None si l'unite n'a pas ete recalee."""
    import cv2

    if not infos_unite.get("recalee"):
        return None
    W, H = cfg["reference"]["largeur"], cfg["reference"]["hauteur"]
    i = infos_unite["unite"] - 1
    x, y, w, h = zone
    y0_ref = (y - cfg["pages"][i][0]) * H
    pts = np.float32(
        [
            [x * W, y0_ref],
            [(x + w) * W, y0_ref],
            [x * W, y0_ref + h * H],
            [(x + w) * W, y0_ref + h * H],
        ]
    )
    Minv = cv2.invertAffineTransform(infos_unite["_M"])
    p = pts @ Minv[:, :2].T + Minv[:, 2]
    p[:, 1] += infos_unite["_debut"]
    return (
        float(p[:, 0].min()),
        float(p[:, 1].min()),
        float(p[:, 0].max()),
        float(p[:, 1].max()),
    )
