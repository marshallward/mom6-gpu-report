all: report.pdf

# TODO: Replace with pandoc
report.pdf: gpu_report.rst
	rst2pdf $^ -o $@
