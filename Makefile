# Include core make support
-include .make/base.mk

-include .make/python.mk

-include .make/oci.mk

MAKE_GIT_HOOKS_DIR := .githooks/

BASE_YQ_INSTALL_DIR := ~/.local/bin