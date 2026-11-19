MAIN := cpn-semantics
LATEXMK := latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error

.PHONY: all watch clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(wildcard *.tex) $(wildcard *.bib)
	$(LATEXMK) $(MAIN).tex

watch:
	$(LATEXMK) -pvc $(MAIN).tex

clean:
	latexmk -bibtex -c $(MAIN).tex

distclean:
	latexmk -bibtex -C $(MAIN).tex
