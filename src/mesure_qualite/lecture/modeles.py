"""Faire lire une image par un modele, interne (SSP Cloud) ou du marche (Eden AI).

Les deux services parlent le meme format (« chat completions ») : une seule fonction,
`lire`, quel que soit le modele. Elle renvoie toujours une `Reponse`, jamais d'exception
pour un appel rate (l'erreur est dans `Reponse.erreur`) : une lecture ratee ne fait pas
tomber une campagne de milliers d'images.

Les cles ne sont jamais ecrites : variable d'environnement, sinon Vault (Onyxia).
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field

import requests

from .sujets import lire_configuration

# Parametres que certains modeles « qui reflechissent » refusent : retires a la volee
_PARAMETRES_REFUSABLES = ("temperature", "max_tokens")


@dataclass(frozen=True)
class Lecteur:
    """Un modele chez un fournisseur (« interne » ou « eden »)."""

    modele: str
    fournisseur: str

    @property
    def nom(self) -> str:
        court = self.modele.split("/")[-1]
        return f"{court} ({self.fournisseur})"

    @property
    def externe(self) -> bool:
        return self.fournisseur != "interne"


@dataclass
class Reponse:
    champs: dict[str, str]
    ok: bool
    secondes: float = 0.0
    jetons: int | None = None
    cout: float | None = None
    version: str | None = None
    json_ok: bool = False
    erreur: str = ""
    brut: str = field(default="", repr=False)


# --- configuration et cles --------------------------------------------------------------
def fournisseur(nom: str, configuration: dict | None = None) -> dict:
    c = configuration or lire_configuration()
    try:
        return c["fournisseurs"][nom]
    except KeyError as e:
        raise KeyError(f"fournisseur {nom!r} absent de configs/lecture.yaml") from e


def lire_vault(chemin: str) -> dict:
    """Secret Vault (KV v2) par l'API web. {} si Vault est absent ou refuse."""
    adresse, jeton = os.environ.get("VAULT_ADDR"), os.environ.get("VAULT_TOKEN")
    if not adresse or not jeton or "/" not in chemin:
        return {}
    montage, reste = chemin.split("/", 1)
    try:
        r = requests.get(
            f"{adresse.rstrip('/')}/v1/{montage}/data/{reste}",
            headers={"X-Vault-Token": jeton},
            timeout=30,
        )
        return r.json()["data"]["data"] if r.status_code == 200 else {}
    except (requests.RequestException, KeyError, ValueError):
        return {}


def cle_api(nom_fournisseur: str, configuration: dict | None = None) -> str:
    """La cle du fournisseur : variable d'environnement, sinon Vault (et on la garde en
    memoire dans l'environnement). Erreur claire si on ne la trouve nulle part."""
    f = fournisseur(nom_fournisseur, configuration)
    variable = f["cle"]
    if not os.environ.get(variable):
        utilisateur = os.environ.get("VAULT_TOP_DIR") or os.environ.get("USER", "")
        valeur = lire_vault(f.get("vault", "").format(utilisateur=utilisateur)).get(
            variable
        )
        if valeur:
            os.environ[variable] = valeur
    cle = os.environ.get(variable, "")
    if not cle:
        raise RuntimeError(
            f"Cle {variable} absente ({f['nom']}). La mettre dans Vault "
            f"({f.get('vault', '')}) ou dans l'environnement avant de lancer."
        )
    if cle.startswith("hvs."):
        raise RuntimeError(
            f"{variable} contient un jeton Vault (hvs.…), pas la cle du service."
        )
    return cle


def lister_internes(configuration: dict | None = None) -> list[str]:
    """Les modeles du service interne, sans ceux qui ne lisent pas d'images
    (embeddings, audio...). L'essai ecartera ensuite ceux qui ne savent pas lire."""
    c = configuration or lire_configuration()
    f = fournisseur("interne", c)
    r = requests.get(
        f["url_modeles"],
        headers={"Authorization": f"Bearer {cle_api('interne', c)}"},
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    liste = j.get("data", j) if isinstance(j, dict) else j
    noms = sorted({m["id"] if isinstance(m, dict) else str(m) for m in liste})
    exclus = [e.lower() for e in c.get("modeles_internes_exclus", [])]
    return [n for n in noms if not any(e in n.lower() for e in exclus)]


# --- un appel ----------------------------------------------------------------------------
def extraire_json(texte: str) -> dict:
    """Le dernier objet JSON du texte (les modeles ajoutent parfois des phrases autour,
    ou des balises ```json). {} si aucun."""
    texte = re.sub(r"```(?:json)?", "", texte or "")
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", texte, re.DOTALL))):
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    return {}


def _image_en_url(image: bytes) -> str:
    mime = "image/png" if image[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image).decode()}"


def lire(
    lecteur: Lecteur,
    image: bytes,
    consigne: str,
    champs: list[str],
    max_jetons: int = 2000,
    delai: float = 180,
    tentatives: int = 2,
    adaptations: dict[str, list[str]] | None = None,
    configuration: dict | None = None,
) -> Reponse:
    """Une image (octets JPEG ou PNG), une consigne, temperature 0.

    adaptations : dict partage modele -> parametres refuses (retires aux appels suivants,
    et note de methode pour MLflow)."""
    f = fournisseur(lecteur.fournisseur, configuration)
    adaptations = {} if adaptations is None else adaptations
    entetes = {
        "Authorization": f"Bearer {cle_api(lecteur.fournisseur, configuration)}",
        "Content-Type": "application/json",
        **(f.get("entetes") or {}),
    }
    corps = {
        "model": lecteur.modele,
        "stream": False,
        "temperature": 0,
        "max_tokens": max_jetons,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_en_url(image)}},
                    {"type": "text", "text": consigne},
                ],
            }
        ],
    }
    for p in adaptations.get(lecteur.modele, []):
        corps.pop(p, None)
        if p == "max_tokens":
            corps["max_completion_tokens"] = max_jetons

    erreur, essai = "", 0
    while essai < tentatives:
        t0 = time.time()
        try:
            r = requests.post(f["url"], headers=entetes, json=corps, timeout=delai)
            if r.status_code == 400:
                texte_err = (r.text or "").lower()
                refuse = next(
                    (
                        p
                        for p in _PARAMETRES_REFUSABLES
                        if p in texte_err and p in corps
                    ),
                    None,
                )
                if refuse:
                    corps.pop(refuse)
                    if refuse == "max_tokens":
                        corps["max_completion_tokens"] = max_jetons
                    adaptations.setdefault(lecteur.modele, []).append(refuse)
                    continue
            r.raise_for_status()
            rep = r.json()
            texte = rep["choices"][0]["message"].get("content") or ""
            d = extraire_json(texte)
            usage = rep.get("usage") or {}
            return Reponse(
                champs={c: str(d.get(c, "") or "") for c in champs},
                ok=True,
                secondes=time.time() - t0,
                jetons=usage.get("total_tokens"),
                cout=rep.get("cost", usage.get("cost")),
                version=rep.get("model", lecteur.modele),
                json_ok=bool(d) and all(c in d for c in champs),
                brut=texte,
            )
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            reponse = getattr(e, "response", None)
            code = getattr(reponse, "status_code", "")
            erreur = f"{type(e).__name__} {code} {(getattr(reponse, 'text', '') or '')[:200]}"
            essai += 1
            if essai < tentatives:
                time.sleep(20 * essai if code == 429 else 3 * essai)
    return Reponse(champs={c: "" for c in champs}, ok=False, erreur=erreur.strip())
