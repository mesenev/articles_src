docker run --rm -it -v \
$(pwd):/source -w \
/source aergus/latex latexmk -bibtex -aux-directory=AuxDirectory --output-directory=PDF -pdf article.tex
