SOURCESTEX   := $(shell find doc -name '*.tex')
SOURCESLYX   := $(shell find doc -name '*.lyx')
SOURCESRST   := $(SOURCESTEX:.tex=.rst) $(SOURCESLYX:.lyx=.rst)

SOURCESIPYNB := $(shell find notebooks -name '*.ipynb')

RESULTS := $(SOURCESRST) $(SOURCESIPYNB:.ipynb=.html)

all: $(SOURCESTEX) $(SOURCESLYX) $(SOURCESIPYNB) $(RESULTS)


%.rst: %.tex
	./make_rst.py $< 

%.rst: %.lyx
	./make_rst.py $< 

%.html: %.ipynb
	ipython nbconvert --to html --stdout $< > $@

clean:
	rm -f $(RESULTS)
