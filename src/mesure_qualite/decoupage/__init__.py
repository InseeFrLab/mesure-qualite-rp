"""Etape 1 du pipeline : decouper un questionnaire scanne en blocs.

    PDF vierge --construire--> gabarit (masque) --+
                                                  +--> recalage --> decoupes + index
    scans de la campagne -------------------------+

1. construire : le gabarit est calcule a partir du PDF vierge de l'imprime (positions
   exactes des cadres, numeros, cases, champs, noms de variables). Rien n'est saisi a la
   main : une nouvelle version du questionnaire = relancer la construction.
2. recaler : chaque scan est aligne sur la page vierge (points ORB, RANSAC,
   transformation affine), sans modele appris.
3. decouper : chaque bloc du gabarit est decoupe sur le scan recale ; un index CSV
   relie chaque decoupe a son scan, son bloc et la qualite du recalage.

En ligne de commande (voir « python -m mesure_qualite.decoupage --help ») :

    python -m mesure_qualite.decoupage construire --pdf "<pdf vierge>"
    python -m mesure_qualite.decoupage decouper --campagne RG25 --formulaire BI

Dans un notebook :

    from mesure_qualite import decoupage as dc
    r = dc.Recaleur.depuis_dossier(dossier_gabarits, "BI_2025")
    crop = dc.decouper_selection(image, r, variables=["NUMVOI", "NOMVOI"])
"""

from ._outils import iou
from .campagne import (
    Decoupe,
    ParametresDecoupage,
    decouper_campagne,
    decouper_scan,
    decouper_selection,
    parametres,
    resume_index,
)
from .construction import construire
from .gabarit import (
    ajouter_themes_par_regles,
    blocs_a_decouper,
    catalogue,
    charger,
    comparer_versions,
    enregistrer,
    image_controle,
    selectionner,
)
from .recalage import Recaleur, zone_dans_scan

__all__ = [
    "Decoupe",
    "ParametresDecoupage",
    "Recaleur",
    "ajouter_themes_par_regles",
    "blocs_a_decouper",
    "catalogue",
    "charger",
    "comparer_versions",
    "construire",
    "decouper_campagne",
    "decouper_scan",
    "decouper_selection",
    "enregistrer",
    "image_controle",
    "iou",
    "parametres",
    "resume_index",
    "selectionner",
    "zone_dans_scan",
]
