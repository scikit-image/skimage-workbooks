# Build and publish the scikit-image technical workbooks.
#
# Every target runs Python through the `$(PYTHON)` on PATH, never an absolute
# interpreter path, so the notebooks execute in whatever environment is
# active. Their frontmatter asks for `kernelspec: name: python3`, which means
# "whatever python3 kernel the executing Jupyter offers"; `make kernel`
# registers that name against $(PYTHON).
#
# MYST is the mystmd CLI, resolved from PATH like PYTHON. C sources build with
# make's default `cc`.

SHELL := bash

PYTHON ?= python
MYST ?= myst
GH ?= gh
FIXTURES_DIR = notebooks/bresenham_nd_fixtures
ZINGL_BIN = $(FIXTURES_DIR)/zingl_line3d

# Shared Zotero export; private repo, so this target needs a logged-in $(GH).
SKIMAGE_REFS_REPO ?= matthew-brett/skimage-refs
SKIMAGE_REFS_REF ?= main
LIBRARY_BIB = library.bib

.PHONY: help html preview book clean rm-ipynb kernel github-pages update-library-bib

help:
	@echo "make html            build the site, warnings as errors"
	@echo "make preview         live preview on localhost:3000, executing notebooks"
	@echo "make book            alias for html"
	@echo "make github-pages    build the site for publishing under /skimage-workbooks"
	@echo "make kernel          register the python3 kernelspec against \$$(PYTHON)"
	@echo "make clean           remove _build and the paired .ipynb files"
	@echo "make environment.yml regenerate the conda environment file"
	@echo "make update-library-bib  refresh $(LIBRARY_BIB) from $(SKIMAGE_REFS_REPO)@$(SKIMAGE_REFS_REF)"

# Registers the "python3" kernelspec the notebooks ask for, pointing at
# $(PYTHON); installing ipykernel does not register it on its own.
kernel:
	$(PYTHON) -m ipykernel install --user --name python3

# Kept in sync with build_requirements.txt by a pre-commit hook.
environment.yml: build_requirements.txt
	@$(PYTHON) make_environment_yml.py $< -o $@

# Pull the Zotero-synced bibliography from skimage-refs on GitHub. Overwrites
# $(LIBRARY_BIB) in place; misc.bib stays local. Needs `gh auth login` for the
# private repo.
update-library-bib:
	$(GH) api "repos/$(SKIMAGE_REFS_REPO)/contents/$(LIBRARY_BIB)?ref=$(SKIMAGE_REFS_REF)" \
		-H "Accept: application/vnd.github.raw" \
		> $(LIBRARY_BIB).tmp
	@mv $(LIBRARY_BIB).tmp $(LIBRARY_BIB)
	@echo "updated $(LIBRARY_BIB) from $(SKIMAGE_REFS_REPO)@$(SKIMAGE_REFS_REF)"

# The Bresenham N-D fixtures need a dedicated env and build; regenerate them
# by hand, per $(FIXTURES_DIR)/README.md.

# The 3-D cross-check in bresenham_nd_cython.md shells out to this binary.
$(ZINGL_BIN): $(FIXTURES_DIR)/zingl_line3d.c
	$(CC) -O2 -o $@ $<

# Notebooks are stored as paired .md; a stray .ipynb means the pairing broke.
html: kernel $(ZINGL_BIN)
	@if compgen -G "notebooks/*.ipynb" 2> /dev/null; then \
	  echo "unpaired ipynb files:" && ls notebooks/*.ipynb && exit 1; fi
	$(MYST) build --html --strict --execute

# Live preview, rebuilding a page when it changes. Same prerequisites as
# `html`: the python3 kernelspec the notebooks ask for, and the binary
# bresenham_nd_cython.md shells out to. `myst start` reads PORT from the
# environment, so `PORT=8000 make preview` moves it off 3000. Drop --execute
# for a faster pass over prose and layout only.
preview: kernel $(ZINGL_BIN)
	$(MYST) start --execute

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
