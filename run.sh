#!/bin/bash

echo "Run handin2 "

# Script that pipes outputs to a file

python3 q1a.py > q1a.txt
python3 q1b.py > q1b.txt
wget https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt
python3 q1cde.py > q1cde.txt
python3 q2.py > q2.txt
python3 q3.py > q3.txt
python3 q4ab.py > q4ab.txt
python3 q5d.py > q5d.txt
python3 q5e.py > q5e.txt

# Download dataset
wget strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt -O grbs.txt
python3 q6.py > q6.txt

echo "Generating the pdf"
pdflatex report.tex

