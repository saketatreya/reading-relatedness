PYTHON ?= python3
export OPENBLAS_NUM_THREADS := 1
FASTME := experiments/.cache/fastme/bin/fastme

.PHONY: all setup test analyze paper reproduce verify clean

all: paper

$(FASTME):
	$(MAKE) setup

setup:
	$(PYTHON) -m experiments.setup_fastme

test: $(FASTME)
	$(PYTHON) -m unittest discover -s tests -v

analyze: $(FASTME)
	$(PYTHON) -m experiments.analyze

paper:
	$(PYTHON) paper/neurreps/build.py

reproduce: analyze
	$(MAKE) paper
	$(MAKE) verify

verify:
	$(PYTHON) -m experiments.verify

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in (Path('paper/neurreps/build'), Path('paper/neurreps/__pycache__'), Path('experiments/__pycache__'), Path('tests/__pycache__'))]"
