SHELL := /bin/bash
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = skeleton
SOURCEDIR     = src
BUILDDIR      = build
DOCSOURCE     = ../src
SPHINXSRC     = src/api

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

create-doc:
	sphinx-apidoc -H API -f -M -e -d 1 --tocfile api -o "$(SPHINXSRC)" "$(DOCSOURCE)"

.PHONY: help Makefile create-doc

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
