VENV_ACTIVATE = source ../myenv/bin/activate

.PHONY: all
all:
	@echo "Usage: make <script_name> (e.g., make fine_tune)"

.PHONY: run
%:
	@$(VENV_ACTIVATE) && python scripts/$@.py
