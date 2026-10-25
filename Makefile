PYTHON ?= python3
UNAME_S := $(shell uname -s)
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')" 2>/dev/null || echo .so)
EXT_GLOB := tcp_sessions_ext*$(EXT_SUFFIX)

ifeq ($(UNAME_S),Darwin)
  EXTRA_CLEAN_GLOBS := tcp_sessions_ext*.so tcp_sessions_ext*.dylib
else
  EXTRA_CLEAN_GLOBS := tcp_sessions_ext*.so tcp_sessions_ext*.pyd
endif

.PHONY: help build build-ext verify-ext rebuild clean test check-env print-config

help:
	@echo "Targets: build build-ext verify-ext rebuild clean test check-env print-config"

check-env:
	@$(PYTHON) -c "import nanobind" >/dev/null
	@echo "nanobind import OK"

print-config:
	@echo "OS=$(UNAME_S)"
	@echo "PYTHON=$(PYTHON)"
	@echo "EXT_SUFFIX=$(EXT_SUFFIX)"

build: check-env
	$(PYTHON) setup.py build_ext --inplace

build-ext: build

verify-ext:
	$(PYTHON) -c "import tcp_sessions_ext; print('tcp_sessions_ext import OK')"

rebuild: clean build

clean:
	rm -rf build
	rm -f $(EXT_GLOB) $(EXTRA_CLEAN_GLOBS)

test:
	$(PYTHON) -m pytest -q
