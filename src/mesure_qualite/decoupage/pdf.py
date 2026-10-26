"""Lecture du PDF vierge d'un formulaire : objets, rendu en image, unites de scan.

Une « unite » est ce que le scanner produit en une image : une page en hauteur, ou
chaque moitie d'une page en largeur (feuille A3 pliee en deux).
"""

from __future__ import annotations

import io

import numpy as np

from . import stockage


def ouvrir_pdf(source):
    """Ouvre un PDF (chemin local, chemin s3:// ou octets). Renvoie (pdfplumber.PDF, octets)."""
    import pdfplumber

    octets = (
        bytes(source)
        if isinstance(source, (bytes, bytearray))
        else stockage.lire_octets(source)
    )
    return pdfplumber.open(io.BytesIO(octets)), octets


def rendre_pages(octets, dpi):
    """Chaque page du PDF en image RGB (pypdfium2 : pas de dependance systeme)."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(octets)
    out = []
    for i in range(len(doc)):
        pg = doc[i]
        pg.set_cropbox(
            *pg.get_mediabox()
        )  # meme repere que pdfplumber (certaines pages ont une CropBox decalee)
        out.append(np.array(pg.render(scale=dpi / 72).to_pil().convert("RGB")))
    doc.close()
    return out


def unites_du_pdf(pdf, ordre="auto"):
    """Les UNITES telles qu'elles sont scannees. Une page en hauteur = une unite. Une page en largeur (feuille A3
    pliee : deux pages cote a cote) = deux unites, numerotees par le numero imprime en bas (ex. 4 | 1, 2 | 3).
    ordre : "auto" (par numero imprime s'il existe, sinon ordre du PDF), "pdf" (pages entieres, ordre du PDF),
    ou une liste de numeros imprimes (ex. [1, 2, 3, 4])."""
    unites = []
    for pi, p in enumerate(pdf.pages):
        W, H = float(p.width), float(p.height)
        if ordre != "pdf" and W > 1.25 * H:
            moities = [(0.0, W / 2), (W / 2, W)]
        else:
            moities = [(0.0, W)]
        mots = p.extract_words()
        for x0, x1 in moities:
            # numero imprime : un nombre seul, tout en bas, pres d'un bord exterieur de la demi-page
            num = None
            for w in mots:
                if (
                    w["text"].isdigit()
                    and len(w["text"]) <= 2
                    and w["top"] > 0.94 * H
                    and x0 <= w["x0"] < x1
                    and min(w["x0"] - x0, x1 - w["x1"]) < 0.12 * (x1 - x0)
                ):
                    num = int(w["text"])
            unites.append(
                {
                    "page_pdf": pi,
                    "x0": x0,
                    "x1": x1,
                    "largeur_pt": x1 - x0,
                    "hauteur_pt": H,
                    "numero_imprime": num,
                }
            )
    if isinstance(ordre, (list, tuple)):
        par_num = {u["numero_imprime"]: u for u in unites}
        unites = [par_num[n] for n in ordre]
    elif (
        ordre == "auto"
        and len(unites) > len(pdf.pages)
        and all(u["numero_imprime"] for u in unites)
    ):
        unites = sorted(unites, key=lambda u: u["numero_imprime"])
    for i, u in enumerate(unites):
        u["unite"] = i
    return unites
