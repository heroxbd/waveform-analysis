.PHONY: arxiv

submission.tar.gz:
	pdflatex arxiv.tex
	bibtex arxiv.aux
	tar -zcvf $@ *.tex *.bbl figures
