# Lecture: Refactoring Jupyter notebook to Python project

<a href="https://www.pytorchlightning.ai/index.html"><img alt="PytorchLightning" src="https://img.shields.io/badge/PytorchLightning-7930e3?logo=lightning&style=flat"></a>
<a href="https://clear.ml/docs/latest/"><img alt="Config: Hydra" src="https://img.shields.io/badge/MLOps-Clear%7CML-%2309173c"></a>

# TODO

`<Videos to be added>`

# Getting started

1. Follow [instructions](https://github.com/python-poetry/install.python-poetry.org)
   to install Poetry:
   ```bash
   # Unix/MacOs installation
   curl -sSL https://install.python-poetry.org | python3 -
   ```
1. Check that poetry was installed successfully:
   ```bash
   poetry --version
   ```
1. Setup workspace:
   ```bash
   make setup_ws
   ```
1. Setup ClearML:
   ```bash
   clearml-init
   ```
1. Migrate dataset to your ClearML workspace:
   ```bash
   make migrate_dataset
   ```
1. (Optional) Configure and run Jupyter lab:
   ```bash
   make jupyterlab_start
   ```

# Train

```bash
make run_training
```

# Useful resources

- [PytorchLightning + Hydra](https://github.com/ashleve/lightning-hydra-template) - a very nice repo template
- [ClearML intro](https://www.youtube.com/playlist?list=PLMdIlCuMqSTnoC45ME5_JnsJX0zWqDdlO) - showcase of main ClearML features
