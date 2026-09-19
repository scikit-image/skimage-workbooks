# Build and publish the scikit-image technical workbooks.
#
# Every target runs Python through the `$(PYTHON)` on PATH, never an absolute
# interpreter path. In this directory `.python-version` makes pyenv resolve
# that to the virtualenv it names, so the notebooks execute in that
# environment: their frontmatter asks for `kernelspec: name: python3`, which
# means "whatever python3 kernel the executing Jupyter offers", so the
# environment follows the caller and the caller is fixed here.
#
# MYST is the mystmd CLI (an npm package), resolved from PATH like PYTHON.

SHELL := bash

PYTHON ?= python
MYST ?= myst
CC ?= gcc
PIP_INSTALL_CMD ?= $(PYTHON) -m pip install
BUILD_DIR = _build/html
FIXTURES_DIR = notebooks/bresenham_nd_fixtures
ZINGL_BIN = $(FIXTURES_DIR)/zingl_line3d

.PHONY: help html book clean rm-ipynb bresenham-fixtures fixtures check-fixtures kernel

help:
	@echo "make html      build the site, warnings as errors"
	@echo "make clean     remove _build and the paired .ipynb files"
	@echo "make bresenham-fixtures   regenerate bresenham_nd_fixtures/*.json"
	@echo "make fixtures        regenerate notebook fixtures from library/ papers"
	@echo "make check-fixtures  verify committed fixtures against the papers"
	@echo "make environment.yml   regenerate the conda environment file"

# Registers the "python3" kernelspec the notebooks ask for, pointing at
# $(PYTHON); installing ipykernel does not register it on its own.
kernel:
	$(PYTHON) -m ipykernel install --user --name python3

# Kept in sync with build_requirements.txt by a pre-commit hook.
environment.yml: build_requirements.txt
	@$(PYTHON) make_environment_yml.py $< -o $@

# Regenerate notebook fixtures extracted from the reference papers under the
# gitignored `library/` symlink (currently the Ng et al. 2014 Fig. 2 panels
# that on_hessian_filter.md compares against). Needs poppler's `pdfimages` and
# a local library/; the committed fixtures are what the build reads, so this is
# a developer target. `make fixtures SET=hhf_fig2` builds just one set.
# See each destination's README.md, and /LICENSE, for the copyright terms.
fixtures:
	$(PYTHON) library/generate_fixtures.py $(SET)

# Verify the committed fixtures still match a fresh extraction from the
# papers. Writes nothing; exits non-zero on any pixel difference.
check-fixtures:
	$(PYTHON) library/generate_fixtures.py --check $(SET)

# The 3-D cross-check in bresenham_nd_cython.md shells out to this binary.
$(ZINGL_BIN): $(FIXTURES_DIR)/zingl_line3d.c
	$(CC) -O2 -o $@ $<

html: kernel $(ZINGL_BIN)
	# Check for ipynb files in source (should all be paired .md).
	if compgen -G "notebooks/*.ipynb" 2> /dev/null; then \
	  (echo "ipynb files" && exit 1); fi
	$(MYST) build --html --strict --execute

github-pages:
	@BASE_URL=/skimage-workbooks $(MAKE) html

# `book` is an alias for `html`, kept because the notebooks refer to it.
book: html

# GitHub Pages publishing is handled by .github/workflows/gh-pages.yml
# (actions/upload-pages-artifact + actions/deploy-pages), not this Makefile.

clean: rm-ipynb
	rm -rf _build
	rm -f $(ZINGL_BIN)

rm-ipynb:
	rm -rf notebooks/*.ipynb
