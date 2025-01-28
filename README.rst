================================
Porting MOM6 to GPUs with OpenMP
================================

This repository tracks our progress in porting MOM6 to GPUs using OpenMP
directives.  It is currently very sparse, but hopefully it will grow into a
more mature document.

Notes are currently in ``gpu_output.rst``.

The notes can be compiled into a more readable PDF using the Makefile.  To
build, just ``make``.  The exact requirements are not yet obvious to me, but
you will at least need Pandoc and a mature LaTeX installation.  If that fails,
try Docutils::

   rst2pdf gpu_output.rst -o report.pdf
