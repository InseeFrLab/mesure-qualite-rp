#!/usr/bin/env bash
# Installe les extensions Quarto des présentations dans docs/presentations/_extensions :
#   - la charte Insee (branche dev_extension de la PR #11 de quarto-insee-extension),
#     épinglée sur un commit pour que le rendu ne change pas sans qu'on le décide ;
#   - Font Awesome (icônes), déjà embarquée par la charte mais non chargée par Quarto
#     quand elle est imbriquée.
# Les extensions pèsent 44 Mo et ne sont pas versionnées.
set -euo pipefail

COMMIT="6e0cee524b5465874b08b399ae0db053d01c9a30"   # dev_extension au 02/10/2026
CIBLE="${1:-$(cd "$(dirname "$0")/.." && pwd)/docs/presentations}"

cd "$CIBLE"
rm -rf _extensions
quarto add "https://github.com/nicotlm/quarto-insee-extension/archive/${COMMIT}.zip" --no-prompt

# Les chemins internes de l'extension attendent _extensions/inseefrlab/insee-clair.
# Installée depuis une archive, elle arrive sans dossier de propriétaire.
mkdir -p _extensions/inseefrlab
mv _extensions/insee-clair _extensions/inseefrlab/insee-clair

mkdir -p _extensions/quarto-ext
cp -r _extensions/inseefrlab/insee-clair/_extensions/fontawesome _extensions/quarto-ext/fontawesome

echo "Extensions installées dans $CIBLE/_extensions :"
ls _extensions _extensions/*
