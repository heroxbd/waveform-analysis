.PHONY: arxiv

submission.tar.gz:
	pdflatex arxiv.tex
	bibtex arxiv.aux
	tar -zcvf $@ --exclude=Note.tex *.tex *.bbl figures

jinst.tar.gz:
	xelatex Note.tex
	bibtex Note.aux
	tar -zcvf $@ --exclude=arxiv.tex *.tex *.bbl figures
