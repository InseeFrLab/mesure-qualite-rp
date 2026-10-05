"""Suivi des essais Fiqual dans MLflow.

Trois regles, appliquees par ce module pour qu'on n'ait pas a s'en souvenir :

1. Une seule experience : `fiqual-banc-essai`.
2. Chaque essai porte les memes etiquettes : `etude`, `notebook`, `statut`,
   `git_commit`. On retrouve donc tout par etude, et non par nom de notebook.
3. Pour chaque etude, UN SEUL essai a le statut `retenu`. C'est celui que la
   presentation affiche. Les autres sont `exploratoire` ou `remplace`.

Utilisation dans un notebook :

    from mesure_qualite.evaluation import suivi
    with suivi.essai("cv-trocr", notebook="05", graine=GRAINE) as run:
        ...                                  # calculs
        mlflow.log_metrics({...})
        mlflow.log_artifact("resume_cv.csv")

    suivi.tableau()                          # ou en est-on ?
    suivi.retenir(run.info.run_id)           # ce chiffre devient le chiffre officiel
"""
import contextlib
import platform
import subprocess
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

EXPERIENCE = "fiqual-banc-essai"

# Une etude = une question a laquelle on repond. Ajouter ici toute nouvelle etude.
ETUDES = {
    "cv-trocr": "TrOCR ajuste, validation croisee par bulletin",
    "complementarite": "croisement gemma4 sur bloc / TrOCR ajuste",
    "bascule-illisible": "TrOCR quand gemma4 repond ILLISIBLE",
    "rg26": "validation sur la campagne RG26",
    "pylaia": "PyLaia, brut et ajuste",
    "dico-noms": "liste des noms Insee comme signal ou correction",
    "derive": "indicateurs de derive sans verite terrain",
}


def _git(*args):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return ""


@contextlib.contextmanager
def essai(etude, notebook, nom=None, graine=None, **etiquettes):
    """Ouvre un essai MLflow avec les etiquettes du projet.

    Appele a l'interieur d'un autre essai, il cree un essai enfant
    (par exemple un bloc de validation croisee) : rien a faire de plus.
    """
    if etude not in ETUDES:
        raise ValueError(f"etude inconnue : {etude!r}. Choisir parmi {sorted(ETUDES)} "
                         f"ou l'ajouter dans ETUDES.")
    mlflow.set_experiment(EXPERIENCE)
    enfant = mlflow.active_run() is not None
    with mlflow.start_run(run_name=nom or f"{etude}-nb{notebook}", nested=enfant) as run:
        modifie = _git("status", "--porcelain")
        mlflow.set_tags({
            "etude": etude,
            "notebook": str(notebook),
            "statut": "exploratoire",
            "git_commit": _git("rev-parse", "--short", "HEAD") or "inconnu",
            # Si le code n'etait pas commite, le commit ne suffit pas a rejouer l'essai
            "git_code_modifie": "oui" if modifie else "non",
            "python": platform.python_version(),
            **{k: str(v) for k, v in etiquettes.items()},
        })
        if graine is not None:
            mlflow.log_param("graine", graine)
        yield run


def retenir(run_id):
    """Fait de cet essai le chiffre officiel de son etude.

    L'ancien essai retenu de la meme etude passe en `remplace` : il reste
    consultable, mais ne sera plus affiche.
    """
    client = MlflowClient()
    etude = client.get_run(run_id).data.tags.get("etude")
    if not etude:
        raise ValueError("cet essai n'a pas d'etiquette 'etude' : il n'a pas ete "
                         "lance avec suivi.essai()")
    anciens = mlflow.search_runs(
        experiment_names=[EXPERIENCE],
        filter_string=f"tags.etude = '{etude}' and tags.statut = 'retenu'")
    for rid in anciens.get("run_id", []):
        if rid != run_id:
            client.set_tag(rid, "statut", "remplace")
    client.set_tag(run_id, "statut", "retenu")
    print(f"{etude} : essai {run_id[:8]} retenu"
          + (f" ({len(anciens)} ancien(s) passe(s) en 'remplace')" if len(anciens) else ""))


def abandonner(run_id, raison):
    """Marque un essai rate ou sans suite, avec la raison (on la retrouvera)."""
    client = MlflowClient()
    client.set_tag(run_id, "statut", "abandonne")
    client.set_tag(run_id, "raison", raison)


def tableau(etude=None, avec_enfants=False, n=40):
    """Vue d'ensemble : un essai par ligne, le plus recent en haut."""
    filtre = "tags.etude != ''"
    if etude:
        filtre = f"tags.etude = '{etude}'"
    df = mlflow.search_runs(experiment_names=[EXPERIENCE], filter_string=filtre,
                            order_by=["attributes.start_time DESC"], max_results=500)
    if df.empty:
        print("aucun essai etiquete")
        return df
    if not avec_enfants and "tags.mlflow.parentRunId" in df:
        df = df[df["tags.mlflow.parentRunId"].isna()]
    vue = pd.DataFrame({
        "date": df["start_time"].dt.strftime("%d/%m %H:%M"),
        "etude": df["tags.etude"],
        "statut": df["tags.statut"],
        "nb": df.get("tags.notebook"),
        "nom": df.get("tags.mlflow.runName"),
        "commit": df.get("tags.git_commit"),
        "run": df["run_id"].str[:8],
    })
    # les metriques principales, si presentes
    for m in ["exact", "apres_exact", "systeme_exact", "bloc_exact", "oracle"]:
        col = f"metrics.{m}"
        if col in df:
            vue[m] = df[col].round(3)
    return vue.head(n).reset_index(drop=True)


def resultats(etude, dossier=None):
    """Lit l'essai RETENU d'une etude : metriques, fichiers et origine.

    Renvoie un dict : metrics, fichiers (chemins locaux), source (texte a citer).
    """
    df = mlflow.search_runs(
        experiment_names=[EXPERIENCE],
        filter_string=f"tags.etude = '{etude}' and tags.statut = 'retenu'")
    if df.empty:
        raise LookupError(f"aucun essai retenu pour l'etude {etude!r} : "
                          f"lancer suivi.retenir(run_id) apres verification")
    if len(df) > 1:
        raise LookupError(f"{len(df)} essais retenus pour {etude!r} : il en faut un seul")
    ligne = df.iloc[0]
    run_id = ligne["run_id"]
    dossier = Path(dossier or tempfile.mkdtemp(prefix=f"mlflow-{etude}-"))
    chemin = Path(mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=str(dossier)))
    fichiers = {p.name: p for p in chemin.rglob("*") if p.is_file()}
    metrics = {c.removeprefix("metrics."): ligne[c]
               for c in df.columns if c.startswith("metrics.") and pd.notna(ligne[c])}
    source = (f"essai {run_id[:8]} du {ligne['start_time']:%d/%m/%Y}, "
              f"commit {ligne.get('tags.git_commit', '?')}")
    return {"metrics": metrics, "fichiers": fichiers, "source": source, "run_id": run_id}
