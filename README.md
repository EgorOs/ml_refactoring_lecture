# Lecture: Refactoring Jupyter notebook to Python project

<div style="text-align: left; display: flex; grid-template-columns: 1fr 1fr;">

<div style="min-width: 132px">

<img src=media/pimp_my_jupyter.png height="187px" style="margin: 16px 0px 0px 0px">

</div>

<div>

```html

   The purpose of this live-coding series is to take a
   simple model trained in Jupyter notebook and refactor
   it into a nice Python project. We demonstrate how
   good coding practices can be applied in
   Deep Learning domain. These series are
   non-exhaustive and serve as an entrypoint for those who
   are curious about `MLOps` and `PytorchLightning`.

```

</div>
</div>
<a href="https://www.pytorchlightning.ai/index.html"><img alt="PytorchLightning" src="https://img.shields.io/badge/PytorchLightning-7930e3?logo=lightning&style=flat"></a>
<a href="https://clear.ml/docs/latest/"><img alt="Config: Hydra" src="https://img.shields.io/badge/MLOps-Clear%7CML-%2309173c"></a>

# Lecture recordings

- [Live-coding 1 \[RU\]](https://www.youtube.com/watch?v=zVIyAJucDBc)
- [Live-coding 2 \[RU\]](https://www.youtube.com/watch?v=-Rre9LSHVMQ)
- [Live-coding 3 \[RU\]](https://www.youtube.com/watch?v=ZXmk2ylwVHQ)

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
