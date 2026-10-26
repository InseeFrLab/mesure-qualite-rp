"""Construction du masque (gabarit) d'un formulaire a partir de son PDF vierge.

Le PDF de l'imprime est vectoriel : il donne la position exacte de chaque mot, rectangle
et trait, mais pas leur sens. Des regles les regroupent :

- un nombre en gras dans un petit carre fonce ouvre une QUESTION, le plus petit cadre
  qui le porte est son bloc ;
- un bandeau fonce large ouvre une SECTION ;
- un cadre sans numero qui contient des champs est un CADRE (identite, agent...) ;
- un petit carre au centre clair est une CASE, des cellules alignees forment un champ
  de CHIFFRES, un rectangle plus large un champ de TEXTE, un trait fin une LIGNE
  d'ecriture ;
- les noms de variables imprimes (PDF « avec VAR ») sont rattaches aux champs et aux
  blocs qui les portent.

Aucun modele n'intervient ici : le resultat est entierement deterministe.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ._outils import (
    bb,
    centre,
    contient,
    dedans,
    est_gras,
    est_italique,
    famille,
    lignes_de,
    normaliser,
    recouvre,
    slug,
    surface,
    texte_de,
    union,
)
from .pdf import ouvrir_pdf, rendre_pages, unites_du_pdf

VERSION_SCHEMA = 2


def couleur_centre(img, r, k):
    """Couleur mediane du centre (50 %) d'un petit rectangle : l'interieur d'une case, sans son contour."""
    x0, y0, x1, y1 = r
    dx, dy = (x1 - x0) * 0.25, (y1 - y0) * 0.25
    a = img[
        int((y0 + dy) * k) : max(int((y0 + dy) * k) + 1, int((y1 - dy) * k)),
        int((x0 + dx) * k) : max(int((x0 + dx) * k) + 1, int((x1 - dx) * k)),
    ]
    return np.median(a.reshape(-1, 3), axis=0) if a.size else np.array([255, 255, 255])


def couleur_interieure(img, r, k):
    """Couleur mediane de l'interieur d'un rectangle (sur le rendu : independant de l'espace colorimetrique du PDF)."""
    x0, y0, x1, y1 = r
    # quatre petites plages pres des coins (a 12-25 % des bords) : le texte au centre (chiffre blanc d'un carre
    # numerote, libelle d'un cadre) ne fausse pas la couleur du fond
    pts = []
    for fx in (0.12, 0.88):
        for fy in (0.12, 0.88):
            cx, cy = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
            rx, ry = max(0.6, 0.1 * (x1 - x0)), max(0.6, 0.1 * (y1 - y0))
            a = img[
                int((cy - ry) * k) : int((cy + ry) * k) + 1,
                int((cx - rx) * k) : int((cx + rx) * k) + 1,
            ]
            if a.size:
                pts.append(a.reshape(-1, 3))
    if not pts:
        return np.array([255, 255, 255])
    return np.median(np.concatenate(pts), axis=0)


def nature_couleur(c):
    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    sat = int(c.max()) - int(c.min())
    if lum > 236 and sat < 22:
        return "clair"
    if lum < 150:
        return "fonce"
    return "teinte"


def variables_imprimees(mots):
    """Les noms de variables imprimes sur les PDF « avec VAR » : une police reservee a eux, dont presque tous les mots
    sont des identifiants en majuscules (SEXE, NUMVOI, NATE_1...). Renvoie (mots variables, autres mots)."""
    motif = re.compile(r"^\(?[A-Z][A-Z0-9_]{1,}\)?$")
    par_fam = {}
    for w in mots:
        par_fam.setdefault(famille(w["fontname"]), []).append(w)
    total = len(mots) or 1
    vfam = {
        f
        for f, ws in par_fam.items()
        if len(ws) >= 3
        and len(ws) < 0.3 * total
        and sum(bool(motif.match(w["text"])) for w in ws) >= 0.8 * len(ws)
    }
    var = [w for w in mots if famille(w["fontname"]) in vfam and motif.match(w["text"])]
    ids = {id(w) for w in var}
    return var, [w for w in mots if id(w) not in ids]


def extraire_page(p, img, k, x_min=0.0, x_max=None):
    """Tous les objets utiles d'une page (ou d'une demi-page [x_min, x_max]) : cadres, marqueurs (carres numerotes),
    cases, cellules a chiffres, champs texte, lignes d'ecriture, mots, variables."""
    x_max = float(p.width) if x_max is None else x_max
    dans_unite = lambda o: x_min - 1 <= (o["x0"] + o["x1"]) / 2 <= x_max + 1
    mots_tous = [
        w for w in p.extract_words(extra_attrs=["fontname", "size"]) if dans_unite(w)
    ]
    var, mots = variables_imprimees(mots_tous)
    # rectangles (y compris les « courbes » fermees a 4 cotes que certains logiciels produisent)
    rects = []
    for r in list(p.rects) + [c for c in p.curves if len(c.get("pts", [])) in (4, 5)]:
        b0 = bb(r)
        b = (
            max(b0[0], x_min),
            b0[1],
            min(b0[2], x_max),
            b0[3],
        )  # coupe a la pliure (tableau sur deux pages)
        if b[2] - b[0] < max(3.0, 0.3 * (b0[2] - b0[0])) and not dans_unite(r):
            continue
        w_, h_ = b[2] - b[0], b[3] - b[1]
        if w_ < 3 or h_ < 3 or (w_ * h_ > 0.6 * (x_max - x_min) * float(p.height)):
            continue
        if any(max(abs(b[i] - q["bb"][i]) for i in range(4)) < 1.0 for q in rects):
            continue  # meme rectangle (fond + contour) : une seule fois
        rects.append(
            {
                "bb": b,
                "couleur": nature_couleur(couleur_interieure(img, b, k)),
                "centre": nature_couleur(couleur_centre(img, b, k)),
            }
        )
    marqueurs, cases, cellules, textes, cadres = [], [], [], [], []
    for r in rects:
        x0, y0, x1, y1 = r["bb"]
        w_, h_ = x1 - x0, y1 - y0
        carre = abs(w_ - h_) < 0.3 * max(w_, h_)
        dedans_mots = [w for w in mots if dedans(centre(w), r["bb"])]
        court = dedans_mots and len("".join(w["text"] for w in dedans_mots)) <= 3
        sur_fonce = any(
            q["couleur"] == "fonce" and contient(q["bb"], r["bb"], 0.5) and q is not r
            for q in rects
        )
        if (
            court
            and 6 <= max(w_, h_) <= 24
            and min(w_, h_) >= 6
            and (
                (
                    r["couleur"] != "clair"
                    and carre
                    and all(est_gras(w) for w in dedans_mots)
                )
                or (r["couleur"] == "fonce" and carre)
                or (sur_fonce and all(est_gras(w) for w in dedans_mots))
            )
        ):
            marqueurs.append(
                {
                    **r,
                    "texte": "".join(w["text"] for w in dedans_mots),
                    "cote": max(w_, h_),
                }
            )
        elif (
            r["centre"] == "clair" and carre and max(w_, h_) <= 12.5 and not dedans_mots
        ):
            cases.append(r)
        elif (
            r["centre"] == "clair"
            and r["couleur"] != "fonce"
            and 9 <= h_ <= 32
            and w_ <= 1.35 * h_
        ):
            cellules.append(r)
        elif (
            r["centre"] == "clair"
            and r["couleur"] == "clair"
            and 7 <= h_ <= 36
            and w_ > 1.35 * h_
        ):
            textes.append(r)
        elif w_ * h_ >= 1200 and min(w_, h_) >= 12:
            cadres.append(r)
    # marqueurs de question vs petits renvois (« passez en 20 ») : les renvois sont nettement plus petits
    if marqueurs:
        grand = np.percentile([m["cote"] for m in marqueurs], 75)
        for m in marqueurs:
            m["renvoi"] = m["cote"] < 0.82 * grand
    # lignes d'ecriture : traits horizontaux fins, hors bords de rectangles, avec de la place au-dessus pour ecrire
    lignes = []
    tous_champs = cases + cellules + textes
    for l in p.lines:
        if (
            not dans_unite(l)
            or l["bottom"] - l["top"] > 1
            or l["x1"] - l["x0"] < 40
            or l.get("linewidth", 1) > 1.2
        ):
            continue
        y = l["top"]
        if any(
            abs(y - r["bb"][1]) < 1.2 or abs(y - r["bb"][3]) < 1.2
            for r in rects
            if r["bb"][0] - 1 <= l["x0"]
            and l["x1"] <= r["bb"][2] + 1
            and r not in cadres
        ):
            continue
        zone = (l["x0"], y - 15, l["x1"], y + 2)
        if any(recouvre(zone, c["bb"]) for c in tous_champs):
            continue
        encre = sum(
            surface(
                (
                    max(zone[0], w["x0"]),
                    max(zone[1], w["top"]),
                    min(zone[2], w["x1"]),
                    min(zone[3], w["bottom"]),
                )
            )
            for w in mots
            if recouvre(zone, bb(w))
        )
        if encre > 0.3 * surface(zone):
            continue
        lignes.append({"bb": zone, "couleur": "clair", "trait": (l["x0"], y, l["x1"])})
    return {
        "mots": mots,
        "variables": var,
        "cadres": cadres,
        "marqueurs": marqueurs,
        "cases": cases,
        "cellules": cellules,
        "textes": textes,
        "lignes": lignes,
    }


# =====================================================================================================================
# 4. ORGANISATION : blocs (questions, sections, cadres), champs, libelles, variables, codes
# =====================================================================================================================
def _libelles(ch, mots, autres_champs):
    """Les textes imprimes autour d'un champ : a gauche (meme ligne), dessous (legende), dessus (en-tete de colonne)."""
    x0, y0, x1, y1 = ch["bb"]
    cy = (y0 + y1) / 2
    gauche = [
        w
        for w in mots
        if w["x1"] <= x0 + 1
        and x0 - w["x1"] < 160
        and w["top"] - 2 <= cy <= w["bottom"] + 4
        and not any(
            c["bb"][0] >= w["x1"] - 1
            and c["bb"][2] <= x0 + 1
            and c["bb"][1] <= cy <= c["bb"][3]
            for c in autres_champs
            if c is not ch
        )
    ]
    # on garde le groupe de mots le plus proche du champ (ecart < 8 pt entre mots)
    gauche = sorted(gauche, key=lambda w: -w["x1"])
    run = []
    for w in gauche:
        if not run and x0 - w["x1"] > 40:
            break
        if run and run[-1]["x0"] - w["x1"] > 8:
            break
        run.append(w)
    run = list(reversed(run))
    dessous = [
        w
        for w in mots
        if y1 - 1 <= w["top"] <= y1 + 9 and w["x0"] < x1 + 2 and w["x1"] > x0 - 2
    ]
    dessous = (
        [w for w in dessous if abs(w["top"] - min(d["top"] for d in dessous)) < 2.5]
        if dessous
        else []
    )
    dessus_c = [
        w
        for w in mots
        if w["bottom"] <= y0 + 1
        and y0 - w["bottom"] < 45
        and w["x0"] < x1
        and w["x1"] > x0
    ]
    dessus = []
    if dessus_c:
        bas = max(w["bottom"] for w in dessus_c)
        if not any(
            c["bb"][3] <= y0 + 1
            and c["bb"][3] > bas
            and c["bb"][0] < x1
            and c["bb"][2] > x0
            for c in autres_champs
        ):
            dessus = [w for w in dessus_c if abs(w["bottom"] - bas) < 2.5]
    return (
        texte_de(run),
        texte_de(sorted(dessous, key=lambda w: w["x0"])),
        texte_de(sorted(dessus, key=lambda w: w["x0"])),
    )


def organiser(o, largeur_unite):
    """A partir des objets d'une unite : la liste des BLOCS et des CHAMPS, chacun avec ses textes."""
    mots, var = o["mots"], o["variables"]
    champs = (
        [{**c, "type": "case"} for c in o["cases"]]
        + [{**c, "type": "texte"} for c in o["textes"]]
        + [{**c, "type": "ligne"} for c in o["lignes"]]
    )
    # cellules a chiffres voisines sur une meme rangee -> un seul champ « chiffres » de n cases
    groupes = []
    for c in sorted(o["cellules"], key=lambda c: (round(c["bb"][1]), c["bb"][0])):
        g = groupes[-1] if groupes else None
        if (
            g
            and abs(g[-1]["bb"][1] - c["bb"][1]) < 1.5
            and 0 <= c["bb"][0] - g[-1]["bb"][2] < 0.45 * (c["bb"][2] - c["bb"][0])
        ):
            g.append(c)
        else:
            groupes.append([c])
    champs += [
        {
            "bb": union(x["bb"] for x in g),
            "type": "chiffres",
            "cases": len(g),
            "cellules": [x["bb"] for x in g],
        }
        for g in groupes
    ]
    # sous-titres : lignes en gras d au moins 7,5 pt, qui ne sont pas des legendes collees sous un champ
    legende = lambda w: any(
        0 <= w["top"] - c["bb"][3] < 9 and w["x0"] < c["bb"][2] and w["x1"] > c["bb"][0]
        for c in champs
        if c["type"] != "case"
    )
    gras_lignes = [
        l
        for l in lignes_de([w for w in mots if not legende(w)])
        if sum(est_gras(w) and not est_italique(w) and w["size"] >= 7.5 for w in l)
        >= 0.5 * len(l)
    ]
    for ch in champs:
        ch["libelle_gauche"], ch["libelle_dessous"], ch["libelle_dessus"] = _libelles(
            ch, mots, champs
        )
        # rubrique : la ligne en gras la plus proche au-dessus (ex. « Localisation du logement »)
        au_dessus = [
            l
            for l in gras_lignes
            if l[-1]["bottom"] <= ch["bb"][1] + 1
            and ch["bb"][1] - l[-1]["bottom"] < 120
            and l[0]["x0"] < ch["bb"][2]
            and l[-1]["x1"] > ch["bb"][0] - 200
            and len(texte_de(l)) > 3
        ]
        ch["rubrique"] = (
            texte_de(max(au_dessus, key=lambda l: l[-1]["bottom"])) if au_dessus else ""
        )
        vs = [w for w in var if dedans(centre(w), ch["bb"], 3)]
        ch["variables"] = [t.strip("()") for w in vs for t in w["text"].split()]
        ch["_vars"] = vs
        if ch["type"] == "case":
            _x0, y0, x1, y1 = ch["bb"]
            cy = (y0 + y1) / 2
            d = [
                w
                for w in mots
                if re.fullmatch(r"\d{1,2}", w["text"])
                and 0 <= w["x0"] - x1 < 12
                and abs((w["top"] + w["bottom"]) / 2 - cy) < 6
            ]
            ch["code"] = d[0]["text"] if d else None
    # suite d'un champ : une ligne / un champ sans libelle juste sous un champ qui en a un (adresse sur 2 lignes)
    for ch in sorted(champs, key=lambda c: c["bb"][1]):
        if ch["type"] in ("texte", "ligne") and not ch["libelle_gauche"]:
            au_dessus = [
                c
                for c in champs
                if c is not ch
                and c["type"] in ("texte", "ligne")
                and c["libelle_gauche"]
                and -4 <= ch["bb"][1] - c["bb"][3] < 10
                and c["bb"][0] < ch["bb"][2]
                and ch["bb"][0] < c["bb"][2]
            ]
            if au_dessus:
                ch["libelle_gauche"] = au_dessus[0]["libelle_gauche"]
                ch["suite"] = True
    # ---- bandeaux : barres foncees larges et basses (titres de sections, ex. « A  Habitants permanents... »)
    cadres = o["cadres"]
    bandeaux = [
        c
        for c in cadres
        if c["couleur"] == "fonce"
        and c["bb"][3] - c["bb"][1] < 30
        and c["bb"][2] - c["bb"][0] > 0.5 * largeur_unite
    ]

    def fin_section(bar):
        """Une section va de son bandeau jusqu'au bandeau suivant (meme colonne), sinon jusqu'au dernier element."""
        suiv = [
            b_["bb"][1]
            for b_ in bandeaux
            if b_["bb"][1] > bar[3] + 2
            and b_["bb"][0] < bar[2]
            and b_["bb"][2] > bar[0]
        ]
        # le cadre qui entoure le bandeau (s'il existe) borne la section : legendes du bas comprises
        autour = [
            c["bb"][3]
            for c in cadres
            if contient(c["bb"], bar, 1) and c["bb"] != bar and c["bb"][3] > bar[3] + 10
        ]
        if suiv:
            return min([min(suiv) - 1.5] + autour)
        if autour:
            return min(autour)
        bas = [
            c["bb"][3]
            for c in champs + cadres
            if c["bb"][1] >= bar[3] - 1 and c["bb"][0] < bar[2] and c["bb"][2] > bar[0]
        ]
        return max(bas) + 10 if bas else bar[3]

    blocs = []
    bars_pris = []
    # ---- tableaux : l'en-tete de colonne descend, le libelle de rangee se propage vers la droite
    for ch in sorted(champs, key=lambda c: c["bb"][1]):
        ch["colonne"] = ch["libelle_dessus"]
        if not ch["colonne"]:
            dessus = [
                c
                for c in champs
                if c is not ch
                and c["bb"][3] <= ch["bb"][1] + 1
                and ch["bb"][1] - c["bb"][3] < 25
                and abs((c["bb"][0] + c["bb"][2]) / 2 - (ch["bb"][0] + ch["bb"][2]) / 2)
                < 0.5 * (ch["bb"][2] - ch["bb"][0]) + 3
                and c.get("colonne")
            ]
            if dessus:
                ch["colonne"] = max(dessus, key=lambda c: c["bb"][3])["colonne"]
    for ch in sorted(champs, key=lambda c: c["bb"][0]):
        ch["rangee"] = ch["libelle_gauche"]
        if not ch["rangee"]:
            gauche = [
                c
                for c in champs
                if c is not ch
                and c["bb"][2] <= ch["bb"][0] + 1
                and ch["bb"][0] - c["bb"][2] < 60
                and abs((c["bb"][1] + c["bb"][3]) / 2 - (ch["bb"][1] + ch["bb"][3]) / 2)
                < 6
                and c.get("rangee")
            ]
            if gauche:
                ch["rangee"] = max(gauche, key=lambda c: c["bb"][2])["rangee"]
    # ---- blocs : chaque marqueur de question -> le plus petit cadre qui le porte (coin haut gauche)
    for m in [m for m in o["marqueurs"] if not m["renvoi"]]:
        mx0, my0, mx1, my1 = m["bb"]
        s = m["cote"]
        cand = [
            c
            for c in cadres
            if c["bb"][0] - 1.5 * s <= mx0 <= c["bb"][0] + 2 * s
            and c["bb"][1] - s <= my0 <= c["bb"][1] + 2 * s
            and c["bb"][2] > mx1
            and c["bb"][3] > my1
        ]
        cand.sort(key=lambda c: surface(c["bb"]))
        if not cand:
            continue
        cadre = cand[0]["bb"]
        if any(cand[0] is b_ for b_ in bandeaux):  # bandeau de section
            cadre = (cadre[0], cadre[1], cadre[2], fin_section(cadre))
            bars_pris.append(cand[0])
            sorte = "section"
        else:
            sorte = "question"
        blocs.append(
            {
                "bb": union([cadre, m["bb"]]),
                "cadre": cadre,
                "marqueur": m,
                "numero": m["texte"],
                "sorte": sorte,
            }
        )
    # ---- bandeaux sans marqueur : sections nommees par leur texte (ou suite d'une section coupee par la pliure)
    for bar in bandeaux:
        if any(bar is b_ for b_ in bars_pris):
            continue
        cadre = (bar["bb"][0], bar["bb"][1], bar["bb"][2], fin_section(bar["bb"]))
        if cadre[3] - bar["bb"][3] < 10:
            continue  # simple barre de titre, sans rien dessous
        blocs.append(
            {
                "bb": cadre,
                "cadre": cadre,
                "marqueur": None,
                "numero": None,
                "sorte": "section",
                "_titre_bar": texte_de(
                    [
                        w
                        for l in lignes_de(
                            [w for w in mots if dedans(centre(w), bar["bb"])]
                        )
                        for w in l
                    ]
                ),
            }
        )
    # ---- cadres sans numero qui contiennent des champs (identite, cadre de l'agent, localisation du logement...)
    pris = [b["cadre"] for b in blocs]
    for c in sorted(cadres, key=lambda c: surface(c["bb"])):
        if any(contient(c["bb"], p_) or contient(p_, c["bb"]) for p_ in pris):
            continue
        dedans_ch = [ch for ch in champs if contient(c["bb"], ch["bb"], 1)]
        if not dedans_ch or surface(c["bb"]) > 0.45 * largeur_unite * 1000:
            continue
        # titre : la ligne de texte juste au-dessus du cadre (ex. « Cadre a remplir par l'agent recenseur »)
        dessus = [
            l
            for l in lignes_de(
                [
                    w
                    for w in mots
                    if w["bottom"] <= c["bb"][1] + 1
                    and c["bb"][1] - w["bottom"] < 22
                    and w["x0"] < c["bb"][2]
                    and w["x1"] > c["bb"][0]
                ]
            )
            if sum(est_gras(w) for w in l) >= 0.5 * len(l)
        ]  # un titre en gras (pas « Exemple : »)
        blocs.append(
            {
                "bb": c["bb"],
                "cadre": c["bb"],
                "marqueur": None,
                "numero": None,
                "sorte": "cadre",
                "_titre_bar": texte_de(dessus[-1]) if dessus else "",
            }
        )
        pris.append(c["bb"])
    # ---- rattacher chaque champ a son bloc (le plus petit qui le contient), textes du bloc
    for ch in champs:
        cont = [b for b in blocs if contient(b["bb"], ch["bb"], 1.5)]
        ch["_bloc"] = min(cont, key=lambda b: surface(b["bb"])) if cont else None
    # une « case » isolee, hors de tout bloc et sans code, est un motif decoratif (logo) : ignoree
    champs = [
        ch
        for ch in champs
        if ch["_bloc"] or not (ch["type"] == "case" and not ch.get("code"))
    ]
    for b in blocs:
        b_mots = [
            w
            for w in mots
            if dedans(centre(w), b["bb"], 0.5)
            and not (b["marqueur"] and dedans(centre(w), b["marqueur"]["bb"]))
            and not any(
                dedans(centre(w), x["bb"])
                for x in blocs
                if x is not b and contient(b["bb"], x["bb"])
            )
        ]
        L = lignes_de(b_mots)
        titre, consignes = [], []
        for l in L:
            # debut gras de la ligne (« Quel est votre sexe ? » suivi de « Masculin 1 Feminin 2 » non gras)
            debut = []
            for w in l:
                if est_gras(w) and not est_italique(w):
                    debut.append(w)
                else:
                    break
            gras_ = len(debut) >= max(1, 0.5 * len(l)) or (
                len(debut) >= 2 and debut[-1]["text"].endswith(("?", ":"))
            )
            if (
                gras_
                and not consignes
                and (not titre or l[0]["top"] - titre[-1][-1]["bottom"] < 8)
            ):
                titre.append(debut)
                if len(debut) < len(l):
                    break
                continue
            if titre and sum(est_italique(w) for w in l) >= 0.6 * len(l):
                consignes.append(l)
                continue
            if titre:
                break
        t = texte_de([w for l in titre for w in l])
        coupe = max(t.rfind("?"), t.rfind(" :"))
        if coupe > 0.4 * len(
            t
        ):  # « Quel est votre sexe ? Masculin Feminin » -> la question
            t = t[: coupe + (1 if t[coupe] == "?" else 2)]
        b["intitule"] = t if (t or not b.get("_titre_bar")) else b["_titre_bar"]
        if b["sorte"] == "section" and not b["numero"] and b.get("_titre_bar"):
            b["intitule"] = b["_titre_bar"]
        b["consignes"] = texte_de([w for l in consignes for w in l])
        b["zone_intitule"] = union(bb(w) for l in titre for w in l) if titre else None
        b["texte_complet"] = "\n".join(texte_de(l) for l in L)
        b["variables"] = []
        for w in sorted(
            [w for w in var if dedans(centre(w), b["bb"], 3)],
            key=lambda w: (round(w["top"]), w["x0"]),
        ):
            b["variables"] += [t_.strip("()") for t_ in w["text"].split()]
        # case a cocher sans variable propre : la variable de la question (SEXE, MATR...) si elle est la seule « libre »
        libres = [
            t_.strip("()")
            for w in var
            if dedans(centre(w), b["bb"], 3)
            and not any(w in ch["_vars"] for ch in champs)
            for t_ in w["text"].split()
        ]
        for ch in champs:
            if (
                ch["_bloc"] is b
                and ch["type"] == "case"
                and not ch["variables"]
                and len(libres) == 1
            ):
                ch["variables"] = libres
    return blocs, champs


# =====================================================================================================================
# 5. CONSTRUCTION DE LA CONFIGURATION (tout le formulaire)
# =====================================================================================================================
def construire(
    pdf_source, formulaire, millesime, dpi=200, ordre="auto", source_nom=None
):
    """Construit la configuration complete. Renvoie (config, images de reference des unites)."""
    pdf, octets = ouvrir_pdf(pdf_source)
    rendus = rendre_pages(octets, dpi)
    k = dpi / 72
    unites = unites_du_pdf(pdf, ordre)
    larg_px = max(round(u["largeur_pt"] * k) for u in unites)
    haut_px = [round(u["hauteur_pt"] * k) for u in unites]
    W, H = larg_px, sum(haut_px)
    debut = np.cumsum([0] + haut_px)
    blocs_cfg, champs_cfg, refs, reperes = [], [], [], []
    vus = {}

    def nouvel_id(base):
        vus[base] = vus.get(base, 0) + 1
        return base if vus[base] == 1 else f"{base}_{vus[base]}"

    def z(r, ui, marge=0.0):
        """Rectangle en points (repere de la page PDF) -> [x, y, w, h] en fractions de l'image EMPILEE."""
        u = unites[ui]
        x0, y0, x1, y1 = (
            (r[0] - u["x0"] - marge) * k,
            (r[1] - marge) * k,
            (r[2] - u["x0"] + marge) * k,
            (r[3] + marge) * k,
        )
        x0, x1 = max(0, x0), min(W, x1)
        return [
            round(x0 / W, 5),
            round((y0 + debut[ui]) / H, 5),
            round((x1 - x0) / W, 5),
            round((y1 - y0) / H, 5),
        ]

    parts = []
    for ui, u in enumerate(unites):
        p = pdf.pages[u["page_pdf"]]
        img = rendus[u["page_pdf"]]
        o = extraire_page(p, img, k, u["x0"], u["x1"])
        blocs, champs = organiser(o, u["largeur_pt"])
        parts.append((o, blocs, champs))
    # sections coupees par la pliure (tableau sur deux pages) : la moitie sans lettre reprend la lettre de l'autre
    lettre_par_y = {
        (unites[ui]["page_pdf"], round(b["cadre"][1])): b["numero"]
        for ui, (_, bl, _) in enumerate(parts)
        for b in bl
        if b["sorte"] == "section" and b["numero"]
    }
    for ui, u in enumerate(unites):
        p = pdf.pages[u["page_pdf"]]
        img = rendus[u["page_pdf"]]
        o, blocs, champs = parts[ui]
        # image de reference de l'unite : rendu, noms de variables effaces (ils ne sont pas sur les vrais formulaires)
        a = img[:, int(round(u["x0"] * k)) : int(round(u["x1"] * k))].copy()  # noqa: RUF046
        for w in o["variables"]:
            x0, y0 = int((w["x0"] - u["x0"]) * k) - 2, int(w["top"] * k) - 2
            x1, y1 = int((w["x1"] - u["x0"]) * k) + 2, int(w["bottom"] * k) + 2
            bord = np.concatenate(
                [
                    a[max(0, y0 - 3) : max(0, y0), max(0, x0) : x1].reshape(-1, 3),
                    a[y1 : y1 + 3, max(0, x0) : x1].reshape(-1, 3),
                ]
            )
            a[max(0, y0) : y1, max(0, x0) : x1] = (
                np.median(bord, axis=0) if len(bord) else 255
            )
        if a.shape[1] < W:
            a = np.pad(a, ((0, 0), (0, W - a.shape[1]), (0, 0)), constant_values=255)
        refs.append(a[: haut_px[ui], :W])
        # reperes : traits epais courts aux coins (information ; le recalage n'en a pas besoin)
        rp = []
        for l in p.lines:
            if (
                l.get("linewidth", 0) >= 1.5
                and max(l["x1"] - l["x0"], l["bottom"] - l["top"]) < 16
                and u["x0"] - 1 <= l["x0"] <= u["x1"] + 1
            ):
                rp.append(
                    [
                        round((l["x0"] - u["x0"]) / u["largeur_pt"], 4),
                        round(l["top"] / u["hauteur_pt"], 4),
                    ]
                )
        reperes.append(rp)
        # ---- blocs (ordre de lecture : colonne puis hauteur)
        col = lambda r, u=u: 0 if (r[0] - u["x0"]) < 0.45 * u["largeur_pt"] else 1
        for b in sorted(
            blocs, key=lambda b: (b["sorte"] != "cadre", col(b["bb"]), b["bb"][1])
        ):
            if b["numero"]:
                base = ("q" if b["numero"].isdigit() else "section_") + b["numero"]
            elif b["sorte"] == "section":
                l_ = next(
                    (
                        v
                        for (pp, y), v in lettre_par_y.items()
                        if pp == u["page_pdf"] and abs(y - b["cadre"][1]) < 3
                    ),
                    None,
                )
                base = (
                    f"section_{l_}_suite"
                    if l_
                    else "section_" + slug(b["intitule"] or f"u{ui + 1}")
                )
                if l_:
                    b["suite_de"] = f"section_{l_}"
            else:
                ch1 = sorted(
                    [c for c in champs if c["_bloc"] is b],
                    key=lambda c: (c["bb"][1], c["bb"][0]),
                )
                nom = (
                    b["intitule"]
                    or b.get("_titre_bar")
                    or next(
                        (c["libelle_gauche"] for c in ch1 if c["libelle_gauche"]), ""
                    )
                    or next(
                        (
                            c["libelle_dessus"] or c["libelle_dessous"]
                            for c in ch1
                            if c["libelle_dessus"] or c["libelle_dessous"]
                        ),
                        "",
                    )
                )
                base = "cadre_" + slug(nom or f"u{ui + 1}")
            b["id"] = nouvel_id(base)
            blocs_cfg.append(
                {
                    "id": b["id"],
                    "sorte": b["sorte"],
                    "question": b["numero"],
                    "unite": ui + 1,
                    "numero_page_imprime": u["numero_imprime"],
                    "intitule": b["intitule"],
                    "consignes": b["consignes"],
                    "texte_complet": b["texte_complet"],
                    "zone": z(b["bb"], ui),
                    "zone_intitule": z(b["zone_intitule"], ui)
                    if b["zone_intitule"]
                    else None,
                    "marge": 0.004,
                    "variables_formulaire": b["variables"],
                    "parent": None,
                    "champs": [],
                    "theme": None,
                    "intitule_court": None,
                    "statut": "a_verifier",
                    **({"suite_de": b["suite_de"]} if b.get("suite_de") else {}),
                }
            )
        for b in blocs:  # imbrication (un cadre dans une section...)
            parents = [x for x in blocs if x is not b and contient(x["bb"], b["bb"], 1)]
            if parents:
                pb = min(parents, key=lambda x: surface(x["bb"]))
                next(c for c in blocs_cfg if c["id"] == b["id"])["parent"] = pb["id"]
        # ---- champs (ordre de lecture dans leur bloc)
        compte = {}
        for ch in sorted(
            champs,
            key=lambda c: (
                c["_bloc"]["id"] if c["_bloc"] else "~",
                round(c["bb"][1] / 4),
                c["bb"][0],
            ),
        ):
            bid = ch["_bloc"]["id"] if ch["_bloc"] else f"hors_bloc_u{ui + 1}"
            cle = (bid, ch["type"])
            compte[cle] = compte.get(cle, 0) + 1
            cid = f"{bid}_{ch['type']}{compte[cle]}"
            e = {
                "id": cid,
                "bloc": bid,
                "type": ch["type"],
                "unite": ui + 1,
                "zone": z(ch["bb"], ui, 1.0),
                "libelle_gauche": ch["libelle_gauche"],
                "libelle_dessous": ch["libelle_dessous"],
                "libelle_dessus": ch["libelle_dessus"],
                "rubrique": ch["rubrique"],
                "colonne": ch["colonne"],
                "rangee": ch["rangee"],
                "variables_formulaire": ch["variables"],
            }
            if ch.get("suite"):
                e["suite"] = True
            if ch["type"] == "case":
                e["code"] = ch.get("code")
            if ch["type"] == "chiffres":
                e["cases"] = ch["cases"]
                e["cellules"] = [z(c_, ui) for c_ in ch["cellules"]]
            champs_cfg.append(e)
            if ch["_bloc"]:
                next(c for c in blocs_cfg if c["id"] == bid)["champs"].append(cid)
    for b in blocs_cfg:
        b["reponse"] = nature_reponse([c for c in champs_cfg if c["bloc"] == b["id"]])
    nums = [
        int(b["question"])
        for b in blocs_cfg
        if b["question"] and str(b["question"]).isdigit()
    ]
    manquants = sorted(set(range(1, max(nums) + 1)) - set(nums)) if nums else []
    cfg = {
        "schema": VERSION_SCHEMA,
        "formulaire": formulaire,
        "millesime": millesime,
        "source": source_nom
        or (
            Path(str(pdf_source)).name
            if not isinstance(pdf_source, (bytes, bytearray))
            else "pdf"
        ),
        "reference": {
            "largeur": int(W),
            "hauteur": int(H),
            "dpi": dpi,
            "source": "pdf_vierge",
        },
        "unites": [
            {
                "unite": u["unite"] + 1,
                "page_pdf": u["page_pdf"] + 1,
                "numero_imprime": u["numero_imprime"],
                "fichier": None,
            }
            for u in unites
        ],
        "pages": [
            [round(debut[i] / H, 6), round(debut[i + 1] / H, 6)]
            for i in range(len(unites))
        ],
        "images_reference": [],
        "reperes": reperes,
        "recalage": "orb_affine",
        "marge_defaut": 0.004,
        "controle": {
            "questions_trouvees": sorted(set(nums)),
            "questions_manquantes": manquants,
            "n_blocs": len(blocs_cfg),
            "n_champs": len(champs_cfg),
            "champs_par_type": {
                t: sum(c["type"] == t for c in champs_cfg)
                for t in ("case", "chiffres", "texte", "ligne")
            },
        },
        "blocs": blocs_cfg,
        "champs": champs_cfg,
    }
    return cfg, refs


def nature_reponse(chs):
    t = {c["type"] for c in chs}
    if not t:
        return "aucune (information)"
    if t == {"case"}:
        return "cases a cocher"
    if "case" in t:
        return "ecriture + cases"
    return "ecriture" if t & {"texte", "ligne"} else "chiffres"


THEMES = [
    "identite",
    "adresse",
    "commune",
    "date",
    "naissance",
    "nationalite",
    "etudes",
    "situation_professionnelle",
    "profession",
    "etablissement",
    "activite",
    "transport",
    "logement",
    "habitants",
    "agent_recenseur",
    "sante",
    "couple",
    "information",
    "autre",
]
MOTS_THEMES = [
    (
        "adresse",
        ["ADRESSE", "LOCALISATION", "NOM DE LA VOIE", "NUMERO", "RESIDENCE HAMEAU"],
    ),
    ("agent_recenseur", ["AGENT RECENSEUR"]),
    ("naissance", ["NAISSANCE", "NE LE", "NE E LE"]),
    ("nationalite", ["NATIONALITE"]),
    ("etudes", ["ENSEIGNEMENT", "DIPLOME", "ETUDES"]),
    ("profession", ["PROFESSION"]),
    ("etablissement", ["ETABLISSEMENT"]),
    ("transport", ["TRANSPORT"]),
    (
        "situation_professionnelle",
        [
            "SITUATION PRINCIPALE",
            "TRAVAILL",
            "EMPLOI",
            "CONTRAT",
            "SALARI",
            "TELETRAVAIL",
        ],
    ),
    ("habitants", ["HABITANTS", "PERSONNES", "ENFANTS"]),
    ("identite", ["NOM", "PRENOM", "SEXE"]),
    ("logement", ["LOGEMENT", "PIECES", "SURFACE", "BATIMENT", "CHAUFFAGE"]),
    ("sante", ["SANTE", "HANDICAP", "LIMITE"]),
    ("couple", ["COUPLE", "MARIE", "PACSE"]),
]


def theme_par_regles(bloc):
    """Theme sans modele (repli) : mots-cles dans l'intitule puis dans tout le texte du bloc."""
    if (bloc.get("reponse") or "").startswith("aucune"):
        return "information"
    for txt in (bloc.get("intitule") or "", bloc.get("texte_complet") or ""):
        t = normaliser(txt).replace("S ADRESSE", "")
        for theme, mots in MOTS_THEMES:
            if any(m in t for m in mots):
                return theme
    return "information" if bloc.get("reponse", "").startswith("aucune") else "autre"
