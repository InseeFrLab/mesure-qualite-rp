"""Le gabarit (masque) d'un formulaire : enregistrement, chargement, selection, controle.

Un gabarit est un dictionnaire JSON (schema 2) accompagne d'une image de reference par
unite de scan (PNG). Toutes les zones sont en fractions de l'image de reference, unites
empilees : (x, y, largeur, hauteur), chaque valeur entre 0 et 1.

Fichiers d'un gabarit « BI_2025 » dans un dossier (local ou s3://) :
    BI_2025.json, BI_2025_u1.png, BI_2025_u2.png
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import stockage
from ._outils import normaliser, police
from .construction import theme_par_regles


def nom_gabarit(cfg: dict) -> str:
    return f"{cfg['formulaire']}_{cfg['millesime']}"


def enregistrer(cfg: dict, refs: list[np.ndarray], dossier: str | Path) -> str:
    """Ecrit le gabarit et ses images de reference dans un dossier (local ou s3://).
    Renvoie le chemin du JSON."""
    nom = nom_gabarit(cfg)
    cfg["images_reference"] = []
    for i, a in enumerate(refs):
        f = f"{nom}_u{i + 1}.png"
        stockage.ecrire_octets(
            stockage.joindre(dossier, f), stockage.encoder_image(a, "png")
        )
        cfg["images_reference"].append(f)
        cfg["unites"][i]["fichier"] = f
    chemin = stockage.joindre(dossier, f"{nom}.json")
    stockage.ecrire_texte(chemin, json.dumps(cfg, ensure_ascii=False, indent=1))
    return chemin


def charger(dossier: str | Path, nom: str) -> tuple[dict, list[np.ndarray]]:
    """Lit un gabarit et ses images de reference. Renvoie (configuration, images)."""
    cfg = json.loads(stockage.lire_texte(stockage.joindre(dossier, f"{nom}.json")))
    refs = [
        stockage.lire_image(stockage.joindre(dossier, f))
        for f in cfg["images_reference"]
    ]
    return cfg, refs


def ajouter_themes_par_regles(cfg: dict) -> dict:
    """Donne un theme a chaque bloc qui n'en a pas, par mots-cles (sans modele)."""
    for b in cfg["blocs"]:
        if not b.get("theme"):
            b["theme"] = theme_par_regles(b)
    return cfg


def image_controle(
    cfg: dict, refs: list[np.ndarray], montrer_champs: bool = True
) -> Image.Image:
    """Toutes les unites empilees ; chaque bloc encadre (rouge = question, bleu = section,
    violet = cadre) avec son identifiant ; chaque champ encadre par type."""
    W, H = cfg["reference"]["largeur"], cfg["reference"]["hauteur"]
    can = Image.fromarray(np.vstack(refs))
    d = ImageDraw.Draw(can)
    F, f2 = police(max(18, W // 50)), police(max(11, W // 110))
    if montrer_champs:
        for c in cfg["champs"]:
            x, y, w, h = c["zone"]
            col = {
                "case": (0, 150, 0),
                "chiffres": (0, 120, 220),
                "texte": (0, 90, 220),
                "ligne": (230, 120, 0),
            }[c["type"]]
            d.rectangle([x * W, y * H, (x + w) * W, (y + h) * H], outline=col, width=2)
    for b in cfg["blocs"]:
        x, y, w, h = b["zone"]
        col = {
            "question": (220, 0, 0),
            "section": (0, 60, 200),
            "cadre": (150, 0, 170),
        }[b["sorte"]]
        d.rectangle([x * W, y * H, (x + w) * W, (y + h) * H], outline=col, width=4)
        lab = f"Q{b['question']}" if b["sorte"] == "question" else b["id"]
        if b["sorte"] == "cadre":
            d.text((x * W + 6, (y + h) * H - 4), lab, fill=col, font=f2, anchor="ld")
            continue
        tw = d.textlength(lab, font=F)
        d.rectangle(
            [(x + w) * W - tw - 14, y * H + 3, (x + w) * W - 3, y * H + F.size + 10],
            fill=col,
        )
        d.text(((x + w) * W - tw - 8, y * H + 5), lab, fill="white", font=F)
    return can


def catalogue(cfg):
    """Tableau de tous les elements (un par ligne) : pour l'explorer, le filtrer, l'exporter."""
    import pandas as pd

    lignes = []
    for b in cfg["blocs"]:
        lignes.append(
            {
                "id": b["id"],
                "niveau": "bloc",
                "type": b["sorte"],
                "question": b["question"],
                "unite": b["unite"],
                "intitule / libelle": b.get("intitule_court") or b["intitule"],
                "theme": b.get("theme"),
                "reponse / code": b["reponse"],
                "variables": " ".join(b["variables_formulaire"]),
                "parent": b.get("parent"),
                "statut": b.get("statut"),
                "zone": b["zone"],
            }
        )
    for c in cfg["champs"]:
        lib = " / ".join(
            x
            for x in [c["libelle_gauche"], c["libelle_dessous"], c["libelle_dessus"]]
            if x
        )
        lignes.append(
            {
                "id": c["id"],
                "niveau": "champ",
                "type": c["type"],
                "question": None,
                "unite": c["unite"],
                "intitule / libelle": lib,
                "theme": None,
                "reponse / code": c.get("code")
                or (f"{c['cases']} cases" if c.get("cases") else ""),
                "variables": " ".join(c["variables_formulaire"]),
                "parent": c["bloc"],
                "statut": None,
                "zone": c["zone"],
            }
        )
    return pd.DataFrame(lignes)


def selectionner(
    cfg,
    id=None,
    question=None,
    variables=None,
    libelle=None,
    rubrique=None,
    colonne=None,
    rangee=None,
    theme=None,
    titre=None,
    types=None,
    avec_intitule=False,
    bloc_entier=False,
):
    """Choisit des elements et renvoie {"zone": union des zones, "elements": [ids], "blocs": [ids]}.
    Criteres (combinables) :
      id        : un ou plusieurs identifiants (bloc ou champ), ex. "q23" ou ["q23_texte1", "q23_texte2"]
      question  : numero de question (ex. 23) -> tout le bloc de la question
      variables : noms de variables imprimes (PDF « avec VAR »), ex. ["NUMVOI", "CPLADR"] -> les champs qui les portent
      libelle   : debut du libelle imprime a cote du champ (ou une liste), ex. "Adresse" -> « Adresse : » et sa suite
      rubrique  : sous-titre en gras au-dessus des champs, ex. "Localisation du logement"
      colonne, rangee : dans un tableau, l'en-tete de colonne (ex. "Nom") et le libelle de rangee (ex. 1 ou [1, 2])
      titre     : un mot de l'intitule du bloc, ex. "Localisation du logement"
      theme     : theme donne par le modele (ex. "adresse")
      types     : ne garder que ces types de champs (ex. ["texte", "ligne"])
    avec_intitule : ajoute l'intitule imprime de la question (et ce qui est entre lui et les champs : les consignes)
    bloc_entier   : prend le bloc entier des elements trouves."""
    blocs = {b["id"]: b for b in cfg["blocs"]}
    champs = {c["id"]: c for c in cfg["champs"]}
    pris = []
    if id is not None:
        for i in [id] if isinstance(id, str) else id:
            if i not in blocs and i not in champs:
                raise KeyError(
                    f"« {i} » n'existe pas dans le gabarit {cfg['formulaire']} {cfg['millesime']}"
                )
            pris.append(i)
    if question is not None:
        qs = {
            str(q)
            for q in ([question] if isinstance(question, (str, int)) else question)
        }
        pris += [b["id"] for b in cfg["blocs"] if str(b["question"]) in qs]
    cand = list(champs.values())
    filtre = False
    if variables is not None:
        vs = {
            v.upper()
            for v in ([variables] if isinstance(variables, str) else variables)
        }
        cand = [c for c in cand if vs & {v.upper() for v in c["variables_formulaire"]}]
        filtre = True
    if libelle is not None:
        ns = [
            normaliser(x) for x in ([libelle] if isinstance(libelle, str) else libelle)
        ]
        cand = [
            c
            for c in cand
            if any(
                normaliser(c.get(k) or "").startswith(n)
                for n in ns
                for k in ("libelle_gauche", "libelle_dessous", "libelle_dessus")
            )
        ]
        filtre = True
    if colonne is not None:
        n = normaliser(colonne)
        cand = [c for c in cand if normaliser(c.get("colonne") or "").startswith(n)]
        filtre = True
    if rangee is not None:
        rs = {
            normaliser(str(x))
            for x in ([rangee] if isinstance(rangee, (str, int)) else rangee)
        }
        cand = [c for c in cand if normaliser(c.get("rangee") or "") in rs]
        filtre = True
    if rubrique is not None:
        n = normaliser(rubrique)
        cand = [c for c in cand if n in normaliser(c.get("rubrique") or "")]
        filtre = True
    if titre is not None:
        n = normaliser(titre)
        ok = {
            b["id"]
            for b in cfg["blocs"]
            if n in normaliser(b["intitule"])
            or n in normaliser(b.get("texte_complet", ""))
        }
        cand = [c for c in cand if c["bloc"] in ok] if filtre or theme else cand
        if not filtre and not theme:
            pris += sorted(ok, key=lambda i: surface_zone(blocs[i]["zone"]))[:1]
    if theme is not None:
        ok = {b["id"] for b in cfg["blocs"] if b.get("theme") == theme}
        cand = [c for c in cand if c["bloc"] in ok]
        filtre = True
    if types is not None:
        cand = [c for c in cand if c["type"] in types]
        filtre = True
    if filtre:
        pris += [c["id"] for c in cand]
    pris = list(dict.fromkeys(pris))
    if not pris:
        raise ValueError(
            "Aucun element ne correspond a cette selection : verifier dans le catalogue (python -m mesure_qualite.decoupage catalogue <gabarit>)."
        )
    blocs_concernes = list(
        dict.fromkeys(i if i in blocs else champs[i]["bloc"] for i in pris)
    )
    zones = [blocs[i]["zone"] if i in blocs else champs[i]["zone"] for i in pris]
    if bloc_entier:
        zones += [blocs[b]["zone"] for b in blocs_concernes if b in blocs]
    if avec_intitule:
        for b in blocs_concernes:
            if b in blocs:
                zones.append(blocs[b]["zone_intitule"] or blocs[b]["zone"])
                if blocs[b][
                    "zone_intitule"
                ]:  # du haut du bloc (numero compris) a l'intitule
                    x, y, _w, _h = blocs[b]["zone"]
                    zones.append(
                        [
                            x,
                            y,
                            0.001,
                            blocs[b]["zone_intitule"][1]
                            - y
                            + blocs[b]["zone_intitule"][3],
                        ]
                    )
    xs0 = min(z_[0] for z_ in zones)
    ys0 = min(z_[1] for z_ in zones)
    xs1 = max(z_[0] + z_[2] for z_ in zones)
    ys1 = max(z_[1] + z_[3] for z_ in zones)
    if avec_intitule:  # largeur du bloc : l'intitule et les champs alignes
        for b in blocs_concernes:
            if b in blocs:
                xs0 = min(xs0, blocs[b]["zone"][0])
                xs1 = max(xs1, blocs[b]["zone"][0] + blocs[b]["zone"][2])
    que_des_champs = all(i in champs for i in pris) and not (
        avec_intitule or bloc_entier
    )
    return {
        "zone": [
            round(float(xs0), 5),
            round(float(ys0), 5),
            round(float(xs1 - xs0), 5),
            round(float(ys1 - ys0), 5),
        ],
        "elements": pris,
        "blocs": blocs_concernes,
        "marge": 0.0015 if que_des_champs else cfg.get("marge_defaut", 0.004),
    }  # champs seuls : marge serree


def surface_zone(z_):
    return z_[2] * z_[3]


def blocs_a_decouper(cfg, quoi="toutes"):
    """Les blocs a decouper : "toutes" = chaque question / cadre / section qui attend une reponse ; ou une liste d'ids
    ou de numeros (ex. ["q23", "cadre_nom"] ou [23, 27])."""
    if quoi == "toutes":
        return [b for b in cfg["blocs"] if not b["reponse"].startswith("aucune")]
    voulus = {str(q) for q in quoi}
    return [
        b for b in cfg["blocs"] if b["id"] in voulus or str(b["question"]) in voulus
    ]


def image_controle_jpeg(
    cfg: dict, refs: list[np.ndarray], largeur: int = 1200
) -> bytes:
    """Image de controle reduite, en JPEG (legere, a joindre a une revue ou a MLflow)."""
    im = image_controle(cfg, refs)
    im = im.resize((largeur, int(im.height * largeur / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=85)
    return b.getvalue()


def cle_question(b: dict) -> str:
    """Identite d'une question d'une version a l'autre, independante de son numero :
    ses variables imprimees si le PDF en porte, sinon le debut de son intitule."""
    if b["variables_formulaire"]:
        return "var:" + "+".join(sorted(b["variables_formulaire"]))
    return "txt:" + normaliser(b["intitule"])[:60]


def comparer_versions(ancien: dict, nouveau: dict) -> list[dict]:
    """Questions ajoutees, supprimees et renumerotees entre deux versions d'un formulaire.

    Le rapprochement se fait par cle_question, jamais par numero. Une question
    reformulee sans variables imprimees apparait comme supprimee puis ajoutee."""
    q_nouv = {cle_question(b): b for b in nouveau["blocs"] if b["sorte"] == "question"}
    q_anc = {cle_question(b): b for b in ancien["blocs"] if b["sorte"] == "question"}
    ajoutees = [
        {
            "changement": "ajoutee",
            "question": b["question"],
            "intitule": b["intitule"][:80],
        }
        for k, b in q_nouv.items()
        if k not in q_anc
    ]
    supprimees = [
        {
            "changement": "supprimee",
            "question": b["question"],
            "intitule": b["intitule"][:80],
        }
        for k, b in q_anc.items()
        if k not in q_nouv
    ]
    renumerotees = [
        {
            "changement": f"renumerotee ({q_anc[k]['question']} -> {b['question']})",
            "question": b["question"],
            "intitule": b["intitule"][:80],
        }
        for k, b in q_nouv.items()
        if k in q_anc and q_anc[k]["question"] != b["question"]
    ]
    return ajoutees + supprimees + renumerotees
