#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex >/dev/null
test -s main.pdf
if command -v pdfinfo >/dev/null 2>&1; then
  pages="$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')"
  test -n "$pages"
  if [ "$pages" -gt 8 ]; then
    echo "Paper exceeds the eight-page budget: $pages pages" >&2
    exit 1
  fi
  echo "Built paper/main.pdf ($pages pages)"
else
  echo 'Built paper/main.pdf; pdfinfo unavailable for page-budget check'
fi
