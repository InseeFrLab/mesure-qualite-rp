"""Enregistrer un meme essai dans DEUX serveurs MLflow : le personnel et le partage.

Le MLflow partage est decrit par un secret Vault (onyxia-kv/projet-mesure-qualite-rp/MLFLOW_SECRET),
comme dans le script du referent technique. Le MLflow personnel est celui que le service
Onyxia configure deja (variables d'environnement presentes au demarrage).

Les deux configurations sont gardees separement ; pour chaque cible, on remet ses variables
d'environnement, on pointe MLflow dessus, puis on rejoue la meme fonction d'enregistrement.

Usage dans un notebook :

    from mesure_qualite.evaluation import double_mlflow as dm

    def enregistrer():
        with mlflow.start_run(run_name="mon-essai") as run:
            mlflow.set_tags({...})
            mlflow.log_metric("exact", 0.84)
            mlflow.log_artifact("resultat.csv")
        return run.info.run_id

    dm.partout(enregistrer)            # -> {"perso": "ab12...", "partage": "cd34..."}
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess

import mlflow

CHEMIN_SECRET = "onyxia-kv/projet-mesure-qualite-rp/MLFLOW_SECRET"
EXPERIENCE = "fiqual-banc-essai"

# Les variables qui decrivent un serveur MLflow et le stockage de ses fichiers
VARIABLES = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
             "MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_INSECURE_TLS", "MLFLOW_S3_ENDPOINT_URL",
             "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION"]

# Photo de la configuration personnelle, prise a l'import (avant tout chargement du secret)
_PERSO = {k: os.environ[k] for k in VARIABLES if k in os.environ}


def _lire_vault_cli(chemin):
    r = subprocess.run(["vault", "kv", "get", "-format=json", chemin],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"vault kv get : {r.stderr.strip()[:200]}")
    return json.loads(r.stdout)["data"]["data"]


def _lire_vault_http(chemin):
    """Meme lecture par l'API web de Vault (services sans la commande `vault`).

    Onyxia fournit VAULT_ADDR et VAULT_TOKEN dans l'environnement. Pour un moteur KV v2,
    'onyxia-kv/a/b' se lit a l'adresse /v1/onyxia-kv/data/a/b.
    """
    import requests
    adresse, jeton = os.environ.get("VAULT_ADDR"), os.environ.get("VAULT_TOKEN")
    if not adresse or not jeton:
        raise RuntimeError("ni commande `vault`, ni VAULT_ADDR / VAULT_TOKEN dans l'environnement")
    moteur, reste = chemin.split("/", 1)
    url = f"{adresse.rstrip('/')}/v1/{moteur}/data/{reste}"
    r = requests.get(url, headers={"X-Vault-Token": jeton}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Vault HTTP {r.status_code} sur {url} : {r.text[:200]}")
    return r.json()["data"]["data"]


def _secret_partage(chemin=CHEMIN_SECRET) -> dict:
    """Lit le secret du MLflow partage dans Vault, SANS toucher aux variables d'environnement.

    Essaie la commande `vault` si elle existe, sinon l'API web de Vault.
    """
    if shutil.which("vault"):
        try:
            secret = _lire_vault_cli(chemin)
        except Exception:
            secret = _lire_vault_http(chemin)
    else:
        secret = _lire_vault_http(chemin)
    secret = {k: str(v) for k, v in secret.items()}
    if "MLFLOW_TRACKING_URI" not in secret:
        raise RuntimeError("Le secret ne contient pas MLFLOW_TRACKING_URI. Cles presentes : "
                           f"{sorted(secret)}")
    return secret


def cibles(perso=True, partage=True) -> dict:
    """Les serveurs ou enregistrer : {"perso": {...}, "partage": {...}}."""
    res = {}
    if perso:
        if "MLFLOW_TRACKING_URI" not in _PERSO:
            print("MLflow personnel : MLFLOW_TRACKING_URI absent au demarrage, cible ignoree.")
        else:
            res["perso"] = dict(_PERSO)
    if partage:
        try:
            res["partage"] = {**_PERSO, **_secret_partage()}
        except Exception as e:
            print(f"MLflow partage ignore : {e}")
    return res


@contextlib.contextmanager
def _environnement(valeurs: dict):
    """Remplace temporairement les variables MLflow / S3, puis remet celles d'avant."""
    avant = {k: os.environ.get(k) for k in VARIABLES}
    try:
        for k in VARIABLES:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in valeurs.items() if k in VARIABLES})
        mlflow.set_tracking_uri(valeurs["MLFLOW_TRACKING_URI"])
        yield
    finally:
        for k, v in avant.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if avant.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(avant["MLFLOW_TRACKING_URI"])


def partout(fonction, experience=EXPERIENCE, perso=True, partage=True) -> dict:
    """Rejoue `fonction` (qui fait l'enregistrement MLflow) sur chaque serveur.

    Une cible en echec n'empeche pas l'autre ; renvoie {cible: valeur renvoyee ou erreur}.
    """
    resultats = {}
    for nom, valeurs in cibles(perso, partage).items():
        with _environnement(valeurs):
            try:
                mlflow.set_experiment(experience)
                resultats[nom] = fonction()
                print(f"MLflow {nom:8} : enregistre ({valeurs['MLFLOW_TRACKING_URI']})")
            except Exception as e:
                resultats[nom] = f"ECHEC {type(e).__name__}: {str(e)[:200]}"
                print(f"MLflow {nom:8} : ECHEC {type(e).__name__}: {str(e)[:200]}")
    return resultats
