SOURCESTEX := $(shell find doc -name '*.tex')
SOURCESLYX := $(shell find doc -name '*.lyx')
SOURCESRST := $(SOURCESTEX:.tex=.rst) $(SOURCESLYX:.lyx=.rst)

RESULTS := $(SOURCESRST)

all: $(SOURCESTEX) $(SOURCESLYX) $(RESULTS)

%.rst: %.tex
	./make_rst.py $< 

%.rst: %.lyx
	./make_rst.py $< 

clean:
	rm -f $(RESULTS)
