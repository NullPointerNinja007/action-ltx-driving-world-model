# CS231N Final Report Overleaf Project

Upload this folder or `action_ltx_final_overleaf_project.zip` to Overleaf.

Main files:

- `main.tex`: CVPR-style report source.
- `references.bib`: bibliography.
- `figures/`: report plots used by `main.tex`.
- `cvpr.sty`, `cvpr_eso.sty`, `eso-pic.sty`, `ieee.bst`: CVPR 2017 template files.

Compile target:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Edit placeholders in the author block before submission.
