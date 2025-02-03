PANDOC = pandoc
OUT_REPORT = gpu_report.pdf
OUT_CHECKLIST = gpu_port_checklist.pdf

FLAGS = \
  -V "mainfont:DejaVu Serif" \
  -V "monofont:DejaVu Sans Mono" \
  --pdf-engine=xelatex

all: $(OUT_REPORT) $(OUT_CHECKLIST)

$(OUT_REPORT): gpu_report.rst
	$(PANDOC) $(FLAGS) -f rst -t pdf -o $@ $<

$(OUT_CHECKLIST): gpu_port_checklist.md
	$(PANDOC) $(FLAGS) -f markdown -t pdf -o $@ $<

# TODO: Support rst2pdf output nicely?
#report.pdf: gpu_report.rst
#	rst2pdf $^ -o $@

clean:
	rm -f $(OUT_REPORT) $(OUT_CHECKLIST)
