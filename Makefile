.PHONY: *

PYTHON_EXEC := python3.10
DROPBOX_DATASET := .dropbox_dataset

CLEARML_PROJECT_NAME := ml_refactoring_lecture
CLEARML_DATASET_NAME := ml_refactoring_lecture_dataset


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install --with notebooks
	poetry run pre-commit install
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python


jupyterlab_start:
	# These lines ensure that CTRL+B can be used to jump to definitions in
	# code of installed modules.
	# Explained here: https://github.com/jupyter-lsp/jupyterlab-lsp/blob/39ee7d93f98d22e866bf65a80f1050d67d7cb504/README.md?plain=1#L175
	ln -s / .lsp_symlink || true  # Create if does not exist.
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	jupyter lab --ContentsManager.allow_hidden=True


migrate_dataset:
	# Migrate dataset to ClearML datasets.
	rm -R $(DROPBOX_DATASET) || true
	mkdir $(DROPBOX_DATASET)
	wget "https://www.dropbox.com/scl/fi/nrn0y41dsfwqsrssav2eo/Classification_data.zip?rlkey=oieytodt749yzyippc6384tge&dl=0" -O $(DROPBOX_DATASET)/dataset.zip
	unzip -q $(DROPBOX_DATASET)/dataset.zip -d $(DROPBOX_DATASET)
	rm $(DROPBOX_DATASET)/dataset.zip
	find $(DROPBOX_DATASET) -type f -name '.DS_Store' -delete
	clearml-data create --project $(CLEARML_PROJECT_NAME) --name $(CLEARML_DATASET_NAME)
	clearml-data add --files $(DROPBOX_DATASET)
	clearml-data close --verbose
	rm -R $(DROPBOX_DATASET)


run_training:
	poetry run $(PYTHON_EXEC) -m src.train


local_test:
	poetry run $(PYTHON_EXEC) -m pytest tests
