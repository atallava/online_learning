#!/bin/bash

rm report.log report.dvi report.blg report.aux report.out
pdflatex -shell-escape report.tex

