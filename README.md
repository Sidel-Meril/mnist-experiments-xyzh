# MNIST experiments

Nine `mnist-*-xyzh` checkouts in one tree and one environment. Python 3.13 and [uv](https://docs.astral.sh/uv/).

```text
uv sync
```

That installs the shared stack every source repo already pinned: PyTorch, torchvision, Optuna, pandera, DVC, and the notebook tools. `uv run pytest` runs every package test. `uv run ruff check src tests` lints them.

## Layout

| Path | What it is |
| --- | --- |
| `src/input_contracts/` | Shared `data_contract.py` and `model_contract.py` |
| `src/<experiment>/` | Stages, model, and that experiment's YAML |
| `data/<experiment>/` | Pipeline tables and pixel arrays. Produced by DVC, not copied here |
| `models/<experiment>/` | Checkpoints copied from the source checkout |
| `reports/<experiment>/` | Metrics, plots, and archives copied from the source checkout |
| `notebooks/<experiment>/` | The notebooks from that checkout |
| `scripts/<experiment>/` | Pull, comparison, and growth scripts |
| `dvc.yaml` | Every stage, prefixed with the experiment name |

Each experiment keeps its own `data_contract.yaml` and `model_contract.yaml` next to its stages. The Python that reads those files is shared:

```python
from input_contracts.data_contract import DataContract
from input_contracts.model_contract import ModelingContract
```

Call `load` with that package's YAML path (`DATA_CONTRACT_PATH` and `MODELING_CONTRACT_PATH` in its `config.py`).

Pixel arrays stay out of this tree. Several source checkouts had `pixels.npy` between 200 MB and 400 MB. `dvc repro` writes them under `data/<experiment>/`.

## Reproduce a pipeline

From this directory. DVC runs a stage after the files it reads exist, so naming the last stage rebuilds that experiment.

```text
uv run dvc repro <stage>
```

| Experiment | Command | What the stages do |
| --- | --- | --- |
| `src/mnist_ae_compression` | `uv run dvc repro mnist_ae_compression_eval` | MNIST autoencoder. Optuna compares latent sizes by reconstruction MSE, then trains and writes reconstruction grids. |
| `src/mnist_betta_vae` | `uv run dvc repro mnist_betta_vae_eval mnist_betta_vae_explore` | β-VAE on MNIST digits. `eval` scores the tuned checkpoint. `explore` fits β in {0.1, 1, 4, 10} and writes the sweep the notebook reads. |
| `src/mnist_f_vae` | `uv run dvc repro mnist_f_vae_eval` | VAE whose posterior is a RealNVP flow in the latent space. |
| `src/mnist_gan_comparison` | `uv run dvc repro mnist_gan_comparison_eval` | Vanilla GAN. This stage trains the dataset currently in `data/mnist_gan_comparison/`. The comparison notebook needs both archives; build them with the script below. |
| `src/mnist_gan_extensions` | `uv run dvc repro mnist_gan_extensions_eval` | Class-conditional GAN. The DVC stage is one run. The six-archive comparison is the script below. |
| `src/mnist_pg_gan` | `uv run dvc repro mnist_pg_gan_eval` | Progressive GAN on CIFAR-10. The DVC train stage is the short pipeline (`--epochs 1` in `dvc.yaml`). The growth notebook reads the longer archive from the script below. |
| `src/mnist_realnvp` | `uv run dvc repro mnist_realnvp_eval mnist_realnvp_eval_fashion` | Multi-scale RealNVP. `eval` is MNIST digits. `eval_fashion` trains and scores Fashion-MNIST from the same split. |
| `src/mnist_vae_comparison` | `uv run dvc repro mnist_vae_comparison_eval` | VAE on one dataset. The comparison notebook reads `reference/`, not this live `data/` directory. |

`make repro-<experiment>` runs the same command. `make install` is `uv sync`.

GAN and VAE comparisons that swap datasets overwrite one `data/<experiment>/` tree. The scripts keep each run:

```text
uv run python scripts/mnist_gan_comparison/train_comparison.py --train-per-class 400 --epochs 40
uv run python scripts/mnist_gan_extensions/model_selection.py --train-per-class 250 --epochs 40
uv run python scripts/mnist_pg_gan/train_progan.py --train-per-class 200 --test-per-class 40 --epochs 2
uv run python scripts/mnist_vae_comparison/pull_sample_data.py --dataset mnist_digits
uv run dvc repro --downstream mnist_vae_comparison_engineer
uv run python scripts/mnist_vae_comparison/cache_reference.py mnist_digits
```

Repeat the VAE pull, downstream repro, and cache for `fashion_mnist`. `--downstream` starts at `engineer` and does not rerun `get_data`, so the pull stays in place. `cache_reference.py` copies that run to `reference/<name>/` in the layout the notebook already uses.

## Notebooks

Open these from the repository root after the files they name exist.

| Notebook | What it does |
| --- | --- |
| `notebooks/mnist_ae_compression/engineering.ipynb` | Design notes for the autoencoder: the stage order, the Optuna study, and which files each stage writes. It does not train. |
| `notebooks/mnist_ae_compression/task_analysis.ipynb` | Reads `reports/mnist_ae_compression/mnist_ae.db`, plots validation MSE against latent size, loads `models/mnist_ae_compression/mnist_ae.pt`, and draws original digits next to reconstructions. Needs the pixel table from `dvc repro`. |
| `notebooks/mnist_betta_vae/exploration.ipynb` | Reads `reports/mnist_betta_vae/beta_exploration.json` and the traversal grids. Plots test MSE, active latent units, and sample diversity against β. |
| `notebooks/mnist_gan_comparison/comparison.ipynb` | Reads `reports/mnist_gan_comparison/generations/digits/` and `.../fashion/`. Plots discriminator and generator loss, the logged pixel std, epoch grids, and the eval table (`sample_std`, mode collapse). Does not train. |
| `notebooks/mnist_gan_extensions/comparison.ipynb` | Reads `reports/mnist_gan_extensions/experiments/<cgan\|patchgan\|feature-matching-loss>/<digits\|fashion>/`. Compares diversity, sharpness, and class grids. Does not train. |
| `notebooks/mnist_pg_gan/growth.ipynb` | Reads `reports/mnist_pg_gan/generations/cifar/` after `scripts/mnist_pg_gan/train_progan.py`. One row of the timeline is one growth phase, from 4×4 up to 32×32. |
| `notebooks/mnist_vae_comparison/comparison.ipynb` | Reads `reference/mnist_digits/` and `reference/fashion_mnist/`. Compares ELBO and MSE, shows reconstruction and sample grids, then encodes both checkpoints for a latent scatter. It does not train. Those two caches are already in `reference/`. |

`src/mnist_f_vae` and `src/mnist_realnvp` have no notebooks. Their charts are the PNG and HTML files under `reports/<experiment>/plots/`.

## Checkpoints already in the tree

Copied from the source checkouts:

- `models/mnist_ae_compression/mnist_ae.pt`
- `models/mnist_gan_comparison/mnist_gan.pt`
- `models/mnist_gan_extensions/mnist_cgan.pt`
- `models/mnist_vae_comparison/mnist_vae.pt`
- `reference/mnist_digits/models/mnist_vae.pt` and `reference/fashion_mnist/models/mnist_vae.pt`

`mnist_betta_vae`, `mnist_f_vae`, `mnist_pg_gan`, and `mnist_realnvp` had no checkpoint file in the source checkout. `dvc repro` writes those under `models/<experiment>/`.
