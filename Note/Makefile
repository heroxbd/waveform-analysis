.PHONY: arxiv

arxiv: submission.tar.gz
jinst: jinst.tar.gz

submission.tar.gz:
	pdflatex arxiv.tex
	bibtex arxiv.aux
	tar -hzcvf $@ --exclude=Note.tex *.tex *.bbl figures

jinst.tar.gz:
	pdflatex Note.tex
	bibtex Note.aux
	tar -hzcvf $@ --exclude=arxiv.tex *.tex *.bbl figures
