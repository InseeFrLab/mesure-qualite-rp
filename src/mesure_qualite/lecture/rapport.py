"""Page HTML des erreurs : chaque bulletin ou un lecteur (ou le PMQ) se trompe, avec
l'image de la decoupe a cote, la verite et ce que chaque lecteur a lu.

Une seule page, autonome (images integrees), lisible dans MLflow ou dans un navigateur.
Elle contient des donnees reelles : elle reste dans le stockage du projet et dans MLflow,
jamais sur Git.
"""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Callable

import pandas as pd
from PIL import Image

from .comparaison import normaliser
from .sujets import Sujet

COULEURS = {
    "juste": ("#e6f4ea", "#1e6b34"),
    "avouee": ("#fff4e0", "#8a5300"),
    "relue": ("#fff4e0", "#8a5300"),
    "silencieuse": ("#fde8e8", "#a11b1b"),
    "ecartee": ("#eeeeee", "#555555"),
}
LIBELLES = {
    "juste": "juste",
    "avouee": "rien lu / illisible",
    "relue": "fausse, signalee",
    "silencieuse": "FAUSSE",
    "ecartee": "verite inconnue",
}

_STYLE = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;color:#1b1b1b;background:#fafafa}
h1{font-size:22px;margin:0 0 4px}
.sous{color:#555;margin:0 0 18px}
table{border-collapse:collapse;font-size:13px}
th,td{border:1px solid #d6d6d6;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f0f0f0}
.resume td,.resume th{text-align:right}.resume td:first-child,.resume th:first-child{text-align:left}
.carte{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:14px 0;
  display:flex;gap:16px;flex-wrap:wrap}
.carte img{max-width:560px;width:100%;border:1px solid #ccc}
.entete{font-weight:600;margin-bottom:6px}
.legende span{display:inline-block;padding:2px 8px;margin-right:6px;border-radius:4px}
code{font-family:ui-monospace,Consolas,monospace}
h2{font-size:17px;margin:22px 0 8px}
.note{font-size:11px;margin-top:2px}
.faux{color:#a11b1b;font-weight:600}
.comparaison td{min-width:120px}
"""


def _image_base64(octets: bytes, largeur: int) -> str:
    im = Image.open(io.BytesIO(octets)).convert("RGB")
    if im.width > largeur:
        im = im.resize((largeur, max(1, int(im.height * largeur / im.width))))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=70)
    return base64.b64encode(b.getvalue()).decode()


def _texte(valeur: object) -> str:
    if valeur is None or (isinstance(valeur, float) and pd.isna(valeur)):
        return ""
    return str(valeur)


def _cellule(contenu: str, verdict: str = "", note: str = "") -> str:
    fond, couleur = COULEURS.get(verdict, ("#ffffff", "#1b1b1b"))
    sous = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return f'<td style="background:{fond};color:{couleur}">{contenu}{sous}</td>'


def _valeur(v: str) -> str:
    return f"<code>{html.escape(v)}</code>" if v else "<i>(vide)</i>"


def _lecture(lu: object, verdict: object, verite: object, brut: object = None) -> str:
    """Une lecture comparee a la verite : la valeur, le verdict ecrit en clair, et ce que
    le modele a ecrit avant la mise au format du Fiqual quand c'est different."""
    lu, verdict, vrai, brut = _texte(lu), _texte(verdict), _texte(verite), _texte(brut)
    note = {
        "juste": "juste",
        "avouee": "rien lu / illisible",
        "relue": f"faux, signale (attendu {vrai or 'vide'})",
        "silencieuse": f"FAUX, attendu {vrai or 'vide'}",
        "ecartee": "verite inconnue",
    }.get(verdict, "")
    if brut and normaliser(brut) != lu:
        note = (
            f"{note} · ecrit : {normaliser(brut)}"
            if note
            else f"ecrit : {normaliser(brut)}"
        )
    return _cellule(_valeur(lu), verdict, note)


def _source_verite(groupe: str) -> str:
    return {
        "A": "PMQ = PAD",
        "B": "arbitrage SeRN",
        "C": "divergence non arbitree",
    }.get(groupe, "")


def lignes_en_erreur(
    t: pd.DataFrame, lecteurs: list[str], sujet: Sujet
) -> pd.DataFrame:
    """Les lignes (bulletin x champ) ou au moins un lecteur, ou le PMQ, se trompe ou ne lit
    rien. Les lignes a verite inconnue (divergence non arbitree) ne comptent pas."""
    s = t[t.champ.isin(sujet.noms_champs)].copy()
    colonnes = ["verdict PMQ", *[f"verdict {n}" for n in lecteurs]]
    s["n_erreurs"] = s[colonnes].isin(["avouee", "relue", "silencieuse"]).sum(axis=1)
    return s[s.n_erreurs > 0]


def comparaison_verite(t: pd.DataFrame, lecteurs: list[str]) -> pd.DataFrame:
    """Par champ (et groupe de champs) et par lecteur : justes, rien lu, fausses, contre
    la verite arbitree (lignes a verite inconnue ecartees)."""
    lignes = []
    for champ in dict.fromkeys(t.champ):
        s = t[t.champ == champ]
        for nom in ["PMQ", *lecteurs]:
            v = s[f"verdict {nom}"].dropna()
            v = v[v != "ecartee"]
            n = len(v)

            def pc(x: int, n: int = n) -> float:
                return round(x / n * 100, 1) if n else float("nan")

            justes = int((v == "juste").sum())
            rien = int(v.isin(["avouee"]).sum())
            fausses = int(v.isin(["silencieuse", "relue"]).sum())
            lignes.append(
                {
                    "champ": champ,
                    "lecteur": nom,
                    "lignes": n,
                    "justes": justes,
                    "justes_%": pc(justes),
                    "rien lu": rien,
                    "rien_lu_%": pc(rien),
                    "fausses": fausses,
                    "fausses_%": pc(fausses),
                }
            )
    return pd.DataFrame(lignes)


def _tableau_comparaison(c: pd.DataFrame) -> str:
    champs = list(dict.fromkeys(c.champ))
    tete = "<tr><th>lecteur</th>" + "".join(
        f"<th>{html.escape(ch)}</th>" for ch in champs
    )
    corps = []
    for nom in dict.fromkeys(c.lecteur):
        cases = []
        for ch in champs:
            r = c[(c.champ == ch) & (c.lecteur == nom)].iloc[0]
            cases.append(
                f"<td><b>{r['justes_%']}&nbsp;%</b> juste<br>"
                f"<span class='faux'>{r['fausses']} faux</span> · {r['rien lu']} rien lu"
                f" · sur {r['lignes']}</td>"
            )
        corps.append(f"<tr><th>{html.escape(nom)}</th>{''.join(cases)}</tr>")
    return f'<table class="comparaison">{tete}</tr>{"".join(corps)}</table>'


def galerie_erreurs(
    t: pd.DataFrame,
    lecteurs: list[str],
    images: dict[str, Callable[[], bytes]],
    sujet: Sujet,
    titre: str = "",
    max_bulletins: int = 400,
    largeur: int = 560,
) -> str:
    """La page HTML : la comparaison avec la verite (en haut), puis chaque bulletin en
    erreur avec sa decoupe. t : table de mesures.juger ; lecteurs : noms des lecteurs."""
    erreurs = lignes_en_erreur(t, lecteurs, sujet)
    base = t[t.champ.isin(sujet.noms_champs)]
    par_bulletin = erreurs.groupby("cle").n_erreurs.sum().sort_values(ascending=False)
    retenus = list(par_bulletin.index[:max_bulletins])
    comparaison = _tableau_comparaison(comparaison_verite(t, lecteurs))
    legende = "".join(
        f'<span style="background:{f};color:{c}">{LIBELLES[k]}</span>'
        for k, (f, c) in COULEURS.items()
        if k != "relue"
    )

    cartes = []
    for k in retenus:
        lignes = base[base.cle == k]
        tete = (
            "<tr><th>champ</th><th>verite</th><th>PMQ</th><th>PAD</th>"
            + "".join(f"<th>{html.escape(n)}</th>" for n in lecteurs)
            + "</tr>"
        )
        corps = []
        for r in lignes.to_dict("records"):
            groupe = _texte(r["groupe"])
            vrai = _texte(r["verite"])
            cellule_verite = _cellule(
                _valeur(vrai) if groupe != "C" else "<i>inconnue</i>",
                "ecartee" if groupe == "C" else "",
                _source_verite(groupe),
            )
            corps.append(
                f"<tr><th>{html.escape(r['champ'])}</th>{cellule_verite}"
                + _lecture(r["PMQ"], r["verdict PMQ"], vrai)
                + _cellule(_valeur(_texte(r["PAD"])))
                + "".join(
                    _lecture(r[f"lu {n}"], r[f"verdict {n}"], vrai, r.get(f"brut {n}"))
                    for n in lecteurs
                )
                + "</tr>"
            )
        try:
            img = (
                '<img alt="decoupe" src="data:image/jpeg;base64,'
                f'{_image_base64(images[k](), largeur)}">'
            )
        except Exception as e:  # noqa: BLE001 (une image illisible n'empeche pas la page)
            img = f"<p>image indisponible ({html.escape(type(e).__name__)})</p>"
        cartes.append(
            f'<div class="carte" id="b{html.escape(k)}"><div>{img}</div><div>'
            f'<div class="entete">bulletin {html.escape(k)} · '
            f"{int(par_bulletin[k])} lecture(s) differente(s) de la verite</div>"
            f"<table>{tete}{''.join(corps)}</table></div></div>"
        )
    plus = (
        f"<p>{len(par_bulletin) - len(retenus)} autres bulletins en erreur non affiches "
        f"(max_bulletins = {max_bulletins}).</p>"
        if len(par_bulletin) > len(retenus)
        else ""
    )
    return (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f"<title>Erreurs de lecture</title><style>{_STYLE}</style></head><body>"
        f"<h1>Comparaison avec la verite : {html.escape(sujet.titre)}</h1>"
        f'<p class="sous">{html.escape(titre)} · verite = valeur du PMQ quand PMQ et PAD '
        "sont d'accord, arbitrage du SeRN sinon ; lectures mises au format du Fiqual "
        "(consignes de saisie) puis comparees ; « ecrit : » = la lecture avant mise au format</p>"
        f"<h2>Par champ et par lecteur</h2>{comparaison}"
        f"<h2>Les bulletins en erreur ({len(par_bulletin)} sur {base.cle.nunique()})</h2>"
        f"<p class='legende'>{legende}</p>{''.join(cartes)}{plus}</body></html>"
    )
