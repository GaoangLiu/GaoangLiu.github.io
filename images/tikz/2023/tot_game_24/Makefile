
all: run

run:
	@pdflatex *.tex
	pdf=$(shell ls *.pdf|shuf -n1); \
	

convert:
	converter="convert -density 1000 -quality 100"; \
	pdf=$(shell ls *.pdf|shuf -n1); \
	$$converter $$pdf $${pdf%.pdf}.png

clean:
	@rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.pdf *.png *.gz

.PHONY: all run clean