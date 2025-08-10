# Include core make support
-include .make/base.mk

-include .make/python.mk

-include .make/oci.mk

MAKE_GIT_HOOKS_DIR := .githooks/

PROJECT_NAME = ska-sdp-instrumental-calibration

BASE_YQ_INSTALL_DIR := ~/.local/bin

docs-pre-build: docs/src
ifeq ("$(DOCS_TARGET_ARGS)", "clean")
	@echo "Cleaning api files..."
	rm -rf docs/src/api
else
	@echo "Generating api files..."
	make -C docs/ create-doc
endif
