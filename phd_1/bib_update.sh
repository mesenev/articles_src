rm 0_main.bbl
rm 0_main.aux
rm 0_main.log
rm 0_main.blg
pdflatex 0_main.tex
bibtex 0_main
pdflatex 0_main.tex
pdflatex 0_main.tex
