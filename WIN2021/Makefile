.PHONY: main clean FORCE

main: poster.pdf poster.png

poster.pdf: FORCE
	latexmk -pdflatex='lualatex -interaction nonstopmode' -pdf poster.tex

poster.png: poster.pdf
	convert -density 200 poster.pdf poster.png

clean:
	latexmk -pdf -C
