#PANDOC = pandoc
PANDOC = /home/marshall/src/pandoc-3.2.1/bin/pandoc
SRC = gpu_report.rst
OUT = report.pdf

FLAGS = \
  -V "mainfont:DejaVu Serif" \
  -V "monofont:DejaVu Sans Mono" \
  --pdf-engine=xelatex

all: $(OUT)

$(OUT): $(SRC)
	$(PANDOC) $(FLAGS) -f rst -t pdf -o $@ $<

# TODO: Support rst2pdf output nicely?
#report.pdf: gpu_report.rst
#	rst2pdf $^ -o $@

clean:
	rm -f $(OUT)
