"""Banc d'essai des modeles de lecture manuscrite (HTR) sur les lignes du Fiqual.

Un meme protocole pour tous les modeles : memes lignes, memes blocs, memes mesures.

- PyLaia (modeles Teklia) : lance dans son propre environnement (~/work/laia-env),
  par fichiers de configuration YAML.
- TrOCR (Hugging Face) : charge dans le noyau du projet.

Les lignes sont preparees une fois par le notebook 05 (cellule d'export) dans
~/work/banc-htr/donnees : images/<id>.png et lignes.csv (id, cle, champ, texte, bloc).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LAIA_BIN = Path.home() / "work" / "laia-env" / "bin"


# --------------------------------------------------------------------------- donnees

def normaliser(texte) -> str:
    """Majuscules, sans accents, espaces simplifies : la forme comparee a la reference."""
    t = unicodedata.normalize("NFKD", str(texte)).encode("ascii", "ignore").decode().upper()
    t = re.sub(r"[^A-Z0-9' -]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def charger_lignes(dossier) -> pd.DataFrame:
    """Les lignes exportees par le notebook 05, avec le chemin de chaque image."""
    dossier = Path(dossier)
    df = pd.read_csv(dossier / "lignes.csv", dtype=str).fillna("")
    df["bloc"] = df.bloc.astype(int)
    df["image"] = [str(dossier / "images" / f"{i}.png") for i in df.id]
    manquantes = [p for p in df.image if not Path(p).exists()]
    if manquantes:
        raise FileNotFoundError(f"{len(manquantes)} images manquantes, par exemple {manquantes[0]}")
    return df


def distance(a: str, b: str) -> int:
    """Distance d'edition : nombre de caracteres a changer pour passer de a a b."""
    prec = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cour = [i]
        for j, cb in enumerate(b, 1):
            cour.append(min(prec[j] + 1, cour[j - 1] + 1, prec[j - 1] + (ca != cb)))
        prec = cour
    return prec[-1]


def mesurer(lignes: pd.DataFrame, lectures: dict) -> pd.DataFrame:
    """Une ligne par image : attendu, lu, exact, distance."""
    rows = []
    for r in lignes.itertuples():
        att, lu = normaliser(r.texte), normaliser(lectures.get(r.id, ""))
        rows.append({"id": r.id, "cle": r.cle, "champ": r.champ, "bloc": r.bloc,
                     "attendu": att, "lu": lu, "exact": int(lu == att),
                     "dist": distance(lu, att), "len": max(len(att), 1)})
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    demi = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - demi), min(1.0, centre + demi)


def resumer(mesures: pd.DataFrame) -> dict:
    """Exact (avec intervalle), CER et part de lectures vides, par champ et au total."""
    res = {}
    for champ in ["NOM", "PRENOM", "TOUS"]:
        g = mesures if champ == "TOUS" else mesures[mesures.champ == champ]
        if not len(g):
            continue
        lo, hi = wilson(int(g.exact.sum()), len(g))
        res[champ] = {"N": len(g), "exact": g.exact.mean(), "ic_bas": lo, "ic_haut": hi,
                      "cer": g.dist.sum() / g["len"].sum(), "vides": (g.lu == "").mean()}
    return res


# --------------------------------------------------------------------------- PyLaia

def pylaia_telecharger(depot: str, destination) -> Path:
    """Telecharge un modele PyLaia de Hugging Face (une seule fois)."""
    destination = Path(destination)
    if not (destination / "weights.ckpt").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=depot, local_dir=str(destination))
    manquants = [f for f in ["model", "weights.ckpt", "syms.txt"] if not (destination / f).exists()]
    if manquants:
        raise FileNotFoundError(f"{depot} : fichiers absents {manquants} — pas un modele PyLaia "
                                f"standard, a examiner a la main dans {destination}")
    return destination


def _aide(exe: Path) -> str:
    r = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=600)
    return r.stdout + r.stderr


class PyLaia:
    """Un modele PyLaia, utilise dans un dossier de travail qui lui est propre.

    Le dossier d'origine du modele n'est jamais modifie : architecture et poids sont copies.
    """

    def __init__(self, dossier_modele, travail, hauteur: int = 128, batch: int = 8,
                 bin_dir: Path = LAIA_BIN):
        self.modele = Path(dossier_modele)
        self.travail = Path(travail)
        self.hauteur, self.batch = hauteur, batch
        self.train_exe = Path(bin_dir) / "pylaia-htr-train-ctc"
        self.decode_exe = Path(bin_dir) / "pylaia-htr-decode-ctc"
        for exe in [self.train_exe, self.decode_exe]:
            if not exe.exists():
                raise FileNotFoundError(f"PyLaia introuvable : {exe}")
        self.travail.mkdir(parents=True, exist_ok=True)
        self.images = self.travail / f"images_h{hauteur}"
        self.syms = self.modele / "syms.txt"
        self.vocab = {l.split()[0] for l in self.syms.read_text(encoding="utf-8").splitlines()
                      if l.strip()}
        # poids isoles : PyLaia peut prendre un autre .ckpt place a cote de celui demande
        self.poids = self.travail / "poids" / "weights.ckpt"
        self.poids.parent.mkdir(exist_ok=True)
        if (not self.poids.exists()
                or self.poids.stat().st_size != (self.modele / "weights.ckpt").stat().st_size):
            shutil.copy(self.modele / "weights.ckpt", self.poids)
        self.archi = self.travail / "brut"
        self.archi.mkdir(exist_ok=True)
        shutil.copy(self.modele / "model", self.archi / "model")
        self.aide_train, self.aide_decode = _aide(self.train_exe), _aide(self.decode_exe)
        gpu = subprocess.run([str(Path(bin_dir) / "python"), "-c",
                              "import torch; print(torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=600)
        self.gpu = gpu.stdout.strip().endswith("True")

    # -- preparation

    def preparer_images(self, lignes: pd.DataFrame):
        self.images.mkdir(exist_ok=True)
        for r in lignes.itertuples():
            f = self.images / f"{r.id}.png"
            if not f.exists():
                im = Image.open(r.image).convert("L")
                w, h = im.size
                im.resize((max(1, round(w * self.hauteur / max(h, 1))), self.hauteur),
                          Image.LANCZOS).save(f)

    def etiquette(self, texte: str) -> list:
        car = []
        for c in normaliser(texte):
            if c == " ":
                if "<space>" in self.vocab:
                    car.append("<space>")
            elif c in self.vocab:
                car.append(c)
        return car

    def _table(self, lignes, chemin):
        with open(chemin, "w", encoding="utf-8") as f:
            for r in lignes.itertuples():
                car = self.etiquette(r.texte)
                if car:
                    f.write(f"{r.id} {' '.join(car)}\n")

    # -- lancement

    def _lancer(self, exe, conf, nom, delai):
        import yaml
        conf_f = self.travail / f"{nom}.yaml"
        conf_f.write_text(yaml.safe_dump(conf, allow_unicode=True), encoding="utf-8")
        t0 = time.time()
        r = subprocess.run([str(exe), "--config", str(conf_f)], capture_output=True, text=True,
                           timeout=delai, cwd=self.travail)
        journal = self.travail / f"{nom}.log"
        journal.write_text(r.stdout + "\n----- stderr -----\n" + r.stderr, encoding="utf-8")
        return r, time.time() - t0, journal

    @staticmethod
    def _erreur(r, journal):
        lignes = [x for x in (r.stderr or "").splitlines() if "error" in x.lower()]
        return (f"code retour {r.returncode} ; journal : {journal}\n"
                + "\n".join(lignes[-5:] or (r.stderr or r.stdout or "")[-800:].splitlines()))

    @staticmethod
    def _verifier_poids(r, attendu, journal):
        """PyLaia ecrit 'Using checkpoint "..."' ; en mode pretrain il utilise <nom>_reset.ckpt,
        une copie des memes poids avec les compteurs d'entrainement remis a zero."""
        m = re.search(r'Using checkpoint "([^"]+)"', (r.stdout or "") + (r.stderr or ""))
        attendu = Path(attendu).resolve()
        acceptes = {attendu, attendu.with_name(attendu.stem + "_reset" + attendu.suffix)}
        if m and Path(m.group(1)).resolve() not in acceptes:
            raise RuntimeError(f"PyLaia a utilise {m.group(1)} au lieu de {attendu}\n"
                               f"journal : {journal}")

    @staticmethod
    def _recoller(texte):
        texte = texte.replace("<space>", "\x00")
        morceaux = texte.split()
        if morceaux and all(len(m) == 1 for m in morceaux):
            texte = "".join(morceaux)
        return texte.replace("\x00", " ").strip()

    # -- lecture

    def lire(self, lignes: pd.DataFrame, nom: str = "brut", archi=None, poids=None) -> dict:
        """Lit les lignes ; par defaut avec le modele d'origine. Renvoie {id: lecture}."""
        self.preparer_images(lignes)
        archi, poids = Path(archi or self.archi), Path(poids or self.poids)
        liste = self.travail / f"liste_{nom}.txt"
        liste.write_text("".join(f"{i}\n" for i in lignes.id), encoding="utf-8")
        conf = {"syms": str(self.syms), "img_list": str(liste), "img_dirs": [str(self.images)],
                "common": {"train_path": str(archi), "experiment_dirname": "experiment",
                           "checkpoint": str(poids.resolve())},
                "decode": {"use_symbols": True, "convert_spaces": True, "join_string": ""}}
        if "--data.batch_size" in self.aide_decode:
            conf["data"] = {"batch_size": self.batch}
        r, _, journal = self._lancer(self.decode_exe, conf, f"lecture_{nom}", 3600)
        self._verifier_poids(r, poids, journal)
        ids = set(lignes.id)
        lus = {}
        for ligne in r.stdout.splitlines():
            p = ligne.strip().split(None, 1)
            if p and Path(p[0]).stem in ids:
                lus[Path(p[0]).stem] = self._recoller(p[1] if len(p) > 1 else "")
        if r.returncode != 0 or not lus:
            raise RuntimeError(f"lecture '{nom}' en echec\n" + self._erreur(r, journal))
        return lus

    # -- ajustement

    def ajuster(self, train: pd.DataFrame, val: pd.DataFrame, nom: str, epoques: int = 100,
                pas: float = 5e-4, patience: int = 100):
        """Ajuste le modele en partant de ses poids. Renvoie (dossier, poids, duree, courbe)."""
        self.preparer_images(pd.concat([train, val]))
        rep = self.travail / nom
        if rep.exists():
            shutil.rmtree(rep)
        rep.mkdir(parents=True)
        shutil.copy(self.modele / "model", rep / "model")
        self._table(train, rep / "train.txt")
        self._table(val, rep / "val.txt")
        conf = {"syms": str(self.syms), "img_dirs": [str(self.images)],
                "tr_txt_table": str(rep / "train.txt"), "va_txt_table": str(rep / "val.txt"),
                "common": {"train_path": str(rep), "experiment_dirname": "experiment",
                           "checkpoint": str(self.poids.resolve())},
                "train": {"pretrain": True}, "trainer": {"max_epochs": epoques}}
        options = [("--common.monitor", ("common", "monitor", "va_cer")),
                   ("--train.augment_training", ("train", "augment_training", True)),
                   ("--train.early_stopping_patience", ("train", "early_stopping_patience", patience)),
                   ("--data.batch_size", ("data", "batch_size", self.batch)),
                   ("--trainer.gpus", ("trainer", "gpus", 1 if self.gpu else 0)),
                   ("--optimizer.learning_rate", ("optimizer", "learning_rate", pas))]
        for option, (section, cle, valeur) in options:
            if option in self.aide_train:
                conf.setdefault(section, {})[cle] = valeur
        r, duree, journal = self._lancer(self.train_exe, conf, f"ajustement_{nom}", 12 * 3600)
        self._verifier_poids(r, self.poids, journal)
        nouveaux = list(rep.rglob("*.ckpt"))
        if r.returncode != 0 or not nouveaux:
            raise RuntimeError(f"ajustement '{nom}' en echec\n" + self._erreur(r, journal))
        meilleurs = [p for p in nouveaux if "lowest_va_cer" in p.name]
        poids = max(meilleurs or nouveaux, key=lambda p: p.stat().st_mtime)
        valeurs = re.findall(r"va_cer[=:]\s*([0-9.]+)", journal.read_text(errors="ignore"))
        courbe = [round(float(v), 3) for v in valeurs][-epoques:]
        return rep, poids, duree, courbe


# --------------------------------------------------------------------------- TrOCR

def _trocr_tokenizer(source: str, processeur: str | None):
    """Le decoupeur de texte du modele, en essayant les classes une par une.

    Le chargement generique (AutoTokenizer, ou via TrOCRProcessor) echoue sous transformers 5
    avec les depots TrOCR : on essaie d'abord les classes explicites, qui lisent directement
    les fichiers du depot. La classe retenue est affichee.
    """
    import transformers as tf
    essais = []
    for depot in [source] + ([processeur] if processeur else []):
        for nom in ["RobertaTokenizerFast", "XLMRobertaTokenizerFast", "AutoTokenizer"]:
            classe = getattr(tf, nom, None)
            if classe is None:
                continue
            try:
                tok = classe.from_pretrained(depot)
            except Exception as e:
                essais.append(f"{nom}({depot}) : {type(e).__name__} : {str(e)[:120]}")
                continue
            note = "" if depot == source else f" — NOTE : celui de {depot}"
            print(f"decoupeur {type(tok).__name__}{note} ...", end=" ", flush=True)
            return tok
    raise RuntimeError("aucun decoupeur ne se charge :\n  " + "\n  ".join(essais))


def _trocr_images(source: str, processeur: str | None):
    """Le preparateur d'images (redimensionnement, normalisation) du modele."""
    import transformers as tf
    essais = []
    for depot in ([processeur] if processeur else []) + [source]:
        for nom in ["ViTImageProcessor", "AutoImageProcessor", "TrOCRProcessor"]:
            classe = getattr(tf, nom, None)
            if classe is None:
                continue
            try:
                p = classe.from_pretrained(depot)
                return getattr(p, "image_processor", p)
            except Exception as e:
                essais.append(f"{nom}({depot}) : {type(e).__name__}")
    raise RuntimeError("aucun preparateur d'images ne se charge :\n  " + "\n  ".join(essais))


def trocr_charger(source: str, processeur: str | None = None):
    """Charge un modele TrOCR : (preparateur d'images, decoupeur, modele, appareil)."""
    import torch
    from transformers import VisionEncoderDecoderModel

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    proc = _trocr_images(source, processeur)
    tok = _trocr_tokenizer(source, processeur)
    modele = VisionEncoderDecoderModel.from_pretrained(source).to(appareil)
    if getattr(modele.config, "decoder_start_token_id", None) is None:
        modele.config.decoder_start_token_id = tok.cls_token_id or tok.bos_token_id
    if getattr(modele.config, "pad_token_id", None) is None:
        modele.config.pad_token_id = tok.pad_token_id
    return proc, tok, modele, appareil


def _trocr_generer(proc, tok, modele, appareil, chemins, lot=8, max_jetons=32) -> list:
    """Lit une liste d'images ; renvoie les textes dans le meme ordre."""
    import torch
    modele.eval()
    textes = []
    for i in range(0, len(chemins), lot):
        images = [Image.open(p).convert("RGB") for p in chemins[i:i + lot]]
        pixels = proc(images=images, return_tensors="pt").pixel_values.to(appareil)
        with torch.no_grad():
            sortie = modele.generate(pixels, max_new_tokens=max_jetons)
        textes += [t.strip() for t in tok.batch_decode(sortie, skip_special_tokens=True)]
    return textes


def _liberer(appareil):
    import gc
    import torch
    gc.collect()
    if appareil == "cuda":
        torch.cuda.empty_cache()


def trocr_lire(source: str, lignes: pd.DataFrame, processeur: str | None = None,
               lot: int = 8, max_jetons: int = 32) -> dict:
    """Lit les lignes avec un modele TrOCR de Hugging Face, tel quel. Renvoie {id: lecture}.

    processeur : depot du preprocesseur d'images, si le modele n'en fournit pas
    (cas de agomberto/trocr-large-handwritten-fr, qui utilise celui de Microsoft).
    """
    proc, tok, modele, appareil = trocr_charger(source, processeur)
    textes = _trocr_generer(proc, tok, modele, appareil, list(lignes.image), lot, max_jetons)
    del modele
    _liberer(appareil)
    return dict(zip(lignes.id, textes))


def trocr_ajuster(source: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                  processeur: str | None = None, epoques: int = 12, pas: float = 1e-5,
                  lot: int = 4, max_jetons: int = 32, graine: int = 42):
    """Ajuste un TrOCR en partant de ses poids, garde la meilleure epoque (CER de validation),
    puis lit les lignes de test.

    Renvoie (lectures du test {id: texte}, courbe du CER de validation, duree en s,
    meilleure epoque). Boucle d'entrainement PyTorch simple, sans Trainer : moins de
    dependance aux changements de version de transformers.
    """
    import torch

    t0 = time.time()
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    proc, tok, modele, appareil = trocr_charger(source, processeur)

    images = [Image.open(p).convert("RGB") for p in train.image]
    textes = [normaliser(t) for t in train.texte]
    attendus_val = [normaliser(t) for t in val.texte]
    pas_total = epoques * int(np.ceil(len(images) / lot))

    opt = torch.optim.AdamW(modele.parameters(), lr=pas)
    planif = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: max(0.0, 1 - k / pas_total))
    amp = appareil == "cuda"
    echelle = torch.amp.GradScaler("cuda", enabled=amp)

    courbe, meilleur, meilleur_etat, meilleure_epoque = [], float("inf"), None, -1
    for ep in range(epoques):
        modele.train()
        ordre = rng.permutation(len(images))
        pertes = []
        for i in range(0, len(ordre), lot):
            idx = ordre[i:i + lot]
            pixels = proc(images=[images[k] for k in idx],
                          return_tensors="pt").pixel_values.to(appareil)
            enc = tok([textes[k] for k in idx], padding="max_length", max_length=max_jetons,
                      truncation=True, return_tensors="pt")
            etiquettes = enc.input_ids.clone()
            etiquettes[etiquettes == tok.pad_token_id] = -100
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                perte = modele(pixel_values=pixels, labels=etiquettes.to(appareil)).loss
            opt.zero_grad(set_to_none=True)
            echelle.scale(perte).backward()
            echelle.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            echelle.step(opt)
            echelle.update()
            planif.step()
            pertes.append(float(perte.detach()))

        lus = _trocr_generer(proc, tok, modele, appareil, list(val.image), 8, max_jetons)
        dist = sum(distance(normaliser(l), a) for l, a in zip(lus, attendus_val))
        cer = dist / max(1, sum(max(len(a), 1) for a in attendus_val))
        courbe.append(round(cer, 4))
        mieux = cer < meilleur
        if mieux:
            meilleur, meilleure_epoque = cer, ep
            meilleur_etat = {k: v.detach().to("cpu", copy=True)
                             for k, v in modele.state_dict().items()}
        print(f"    epoque {ep + 1:2d}/{epoques} | perte {np.mean(pertes):.3f} | "
              f"CER validation {cer:.1%}{'  <- meilleure' if mieux else ''}", flush=True)

    modele.load_state_dict(meilleur_etat)
    lus_test = _trocr_generer(proc, tok, modele, appareil, list(test.image), 8, max_jetons)
    del modele, opt, meilleur_etat
    _liberer(appareil)
    return dict(zip(test.id, lus_test)), courbe, time.time() - t0, meilleure_epoque + 1


# --------------------------------------------------------------------------- TeleOCR

TELEOCR_DEPOTS = ["XingChen-AGI/TeleOCR", "StarDoc-AI/TeleOCR"]
CONSIGNE_TEXTE = "Please output the text content from the image."
CONSIGNE_MISE_EN_PAGE = "Analyze the image layout."


def teleocr_charger(depots=TELEOCR_DEPOTS):
    """Charge TeleOCR (modele vision-langage d'analyse de documents, ~1,2 G parametres).

    ATTENTION : trust_remote_code=True execute du code fourni par le depot du modele.
    A faire valider avant tout usage sur des donnees reelles.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    erreurs = []
    for depot in depots:
        try:
            proc = AutoProcessor.from_pretrained(depot, trust_remote_code=True, use_fast=True)
            try:
                modele = AutoModel.from_pretrained(depot, trust_remote_code=True, dtype=torch.bfloat16)
            except TypeError:
                modele = AutoModel.from_pretrained(depot, trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16)
            appareil = "cuda" if torch.cuda.is_available() else "cpu"
            return proc, modele.to(appareil).eval(), depot
        except Exception as e:
            erreurs.append(f"{depot} : {type(e).__name__}: {str(e)[:300]}")
    raise RuntimeError("TeleOCR ne se charge pas :\n  " + "\n  ".join(erreurs))


def teleocr_lire(proc, modele, image, consigne=CONSIGNE_TEXTE, max_jetons=512, taille=None) -> str:
    """Une image + une consigne -> le texte produit par TeleOCR (methode de sa fiche officielle)."""
    import torch
    im = image.convert("RGB")
    if taille:
        im = im.resize((taille, taille), Image.BICUBIC)
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": consigne}]}]
    invite = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    entrees = proc(text=[invite], images=[im], padding=True,
                   return_tensors="pt").to(device=modele.device, dtype=modele.dtype)
    with torch.no_grad():
        sortie = modele.generate(**entrees, use_cache=True, max_new_tokens=max_jetons, do_sample=False)
    ids = sortie.cpu().tolist()[0][len(entrees.input_ids[0]):]
    return proc.batch_decode([ids], skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)[0].strip()


def _majuscules(t: str) -> str:
    return unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().upper()


def extraire_champs(texte: str) -> dict:
    """Retrouve le nom et le prenom dans un texte lu sur un bloc ou une page,
    grace aux libelles imprimes (« Nom : », « Prenom(s) : »)."""
    plat = "\n".join(_majuscules(l) for l in str(texte).splitlines())
    res = {}
    for champ, libelle in [("PRENOM", r"\bPRENOMS?\b"), ("NOM", r"\bNOM\b")]:
        m = (re.search(libelle + r"[^:\n]{0,25}:\s*([^\n]*)", plat)
             or re.search(libelle + r"\s+([^\n]*)", plat))
        valeur = m.group(1) if m else ""
        valeur = re.split(r"\bPRENOMS?\b|\bNOM\b|\bSEXE\b|\bNE\b|\bDATE\b", valeur)[0]
        res[champ] = normaliser(valeur)
    return res


def boites(texte: str, taille: int = 1036) -> list:
    """Les rectangles (x1, y1, x2, y2) trouves dans une sortie de mise en page, en fractions
    de l'image. Hypothese : coordonnees en pixels de l'image redimensionnee (taille x taille),
    ou deja en fractions si toutes <= 1. A verifier sur les images de controle."""
    nombre = r"(-?\d+(?:\.\d+)?)"
    trouves = re.findall(r"\[\s*" + r"\s*,\s*".join([nombre] * 4) + r"\s*\]", texte)
    trouves += re.findall(r"\(\s*" + nombre + r"\s*,\s*" + nombre + r"\s*\)\s*,\s*\(\s*"
                          + nombre + r"\s*,\s*" + nombre + r"\s*\)", texte)
    b = [tuple(float(v) for v in t) for t in trouves]
    b = [x for x in b if x[2] > x[0] and x[3] > x[1]]
    if not b:
        return []
    echelle = 1.0 if max(max(x) for x in b) <= 1.0 else float(taille)
    return [tuple(v / echelle for v in x) for x in b]


def couverture(zone, liste_boites) -> float:
    """Part de la zone de reference (x, y, l, h en fractions) couverte par la meilleure boite."""
    zx1, zy1, zx2, zy2 = zone[0], zone[1], zone[0] + zone[2], zone[1] + zone[3]
    aire = max(1e-9, (zx2 - zx1) * (zy2 - zy1))
    meilleure = 0.0
    for x1, y1, x2, y2 in liste_boites:
        inter = max(0.0, min(x2, zx2) - max(x1, zx1)) * max(0.0, min(y2, zy2) - max(y1, zy1))
        meilleure = max(meilleure, inter / aire)
    return meilleure
