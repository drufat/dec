SHELL := /bin/bash

SOURCESTEX   := $(shell find doc -name '*.tex')
SOURCESLYX   := $(shell find doc -name '*.lyx')
SOURCESRST   := $(SOURCESTEX:.tex=.rst) $(SOURCESLYX:.lyx=.rst)

SOURCESIPYNB := $(shell find examples -name '*.ipynb')

RESULTS := $(SOURCESRST)                 \
           $(SOURCESIPYNB:.ipynb=.rst)   \
#           $(SOURCESIPYNB:.ipynb=.html)  \


all: $(SOURCESTEX) $(SOURCESLYX) $(SOURCESIPYNB) $(RESULTS)
	sphinx-build -b html doc build/html


%.rst: %.tex
	./make_rst.py $< 

%.rst: %.lyx
	./make_rst.py $< 

%.html: %.ipynb
	ipython nbconvert --to html --stdout $< > $@

%.rst: %.ipynb
	./nbconvert.sh rst $<
	
clean:
	rm -f $(RESULTS) 
	rm -rf examples/*_files
