# Learned Sparse News Recommendations (LSNR)

Repository accompanying the SIGIR paper **“Reliable and Controllable Learned Sparse News Recommendations for Short-History Users.”**

LSNR learns sparse, bag-of-words-style user profiles that restore reliable retrieval quality for short-history users while retaining an explicit performance–efficiency trade-off. This codebase reproduces the paper’s experiments on MIND and internal ablations, and provides utilities to adapt the approach to new settings.

---

## Contents

- `experiments/` – YAML experiment definitions used in the paper
- `src/` – training, evaluation, datasets, and model implementations
- `reports/`, `figures/`, `inspect_lsnr_news.py` – analysis notebooks and scripts

---

## 1. Environment Setup

The project targets Python ≥3.10. We recommend using `uv` for dependency management:

```bash
uv sync
source .venv/bin/activate
```

Alternatively, you can use `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Weights & Biases is used for experiment logging. Either log in with `wandb login` or disable logging by setting `logging.wandb.enabled: false` in the experiment YAML.

---

## 2. Data Preparation

Create the canonical directory layout before unpacking datasets:

```bash
mkdir -p data/mind/{train,val,test}

```

### 2.1 MIND

```bash
wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip
wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d data/mind/train
unzip MINDsmall_dev.zip   -d data/mind/val
unzip MINDsmall_dev.zip   -d data/mind/test
```



---

## 3. Preprocessing Pipelines

Preprocessing scripts materialize the tokenised splits, mapping files, and cold-start variants used during training/evaluation. They read defaults from `config.yaml` but accept overrides via environment variables or CLI arguments.

```bash
# MIND
python src/data_preprocess_mind.py



# EBNeRD
python src/data_preprocess_ebnerd.py --dataset ebnerd

```

Outputs are written to `data/<dataset>/{train,val,test}/`, including:

- `behaviors_parsed.tsv` (training) and `behaviors_parsed_cold*.tsv` (evaluation cold-threshold slices)
- `news.tsv` per split limited to referenced articles
- `user2int.tsv`, `category2int.tsv` shared across splits

---

## 4. Configuration & Experiment Management

Experiments are defined in `experiments/*.yaml` and may inherit from `experiments/base.yaml` via the `_base_` key. The loader (`src/config.py`) resolves inheritance, performs type conversions, and exposes:

- `experiment` metadata (name, model, description)
- `datasets` block with per-dataset hyperparameters
- `models` block with model-specific settings (including sparsity regularisation for LSNR)
- `training` block controlling optimisation, evaluation cadence, and cold-threshold selection

To create a new experiment, copy an existing YAML file and adjust the relevant sections. Any YAML string using scientific notation (e.g. `1e-4`) is automatically cast to floats.

---

## 5. Training

Training is driven by `src/train.py` and consumes experiment names:

```bash


# MIND variant
python src/train.py lsnr_baseline mind
```

Key options:

- Positional `experiment`: name of the YAML file in `experiments/`
- Optional positional `dataset`: defaults to `adressa`
- Logging: checkpoints saved under `checkpoints/<experiment>_<model>_<dataset>_cold<k>_lr<lr>/`

LSNR models expose a quadratic lambda warm-up; sparsity settings live in the experiment config under `models.lsnr`.

---

## 6. Evaluation & Analysis

### 6.1 CLI entrypoint

`src/evaluate.py` offers a consolidated interface that loads checkpoints and runs retrieval metrics over multiple cold thresholds:

```bash


# Evaluate NRMS on MIND with a custom checkpoint directory
python src/evaluate.py nrms --dataset mind --checkpoint-dir checkpoints/my_nrms_run
```

Useful flags:

- `--cold-thresholds`: space-separated list (default `[0, 1, 2, 3, 4, 5]`)
- `--lr`: select checkpoints trained with a specific learning rate
- `--batch-size`: override evaluation batch size
- `--regularization`: select checkpoint by regularization level (e.g., `baseline`)


## 7. Extending to New Datasets

To adapt LSNR to a new corpus:

1. Implement a dataset preprocessor producing `news.tsv`, `behaviors.tsv`, and derived mapping files.
2. Add a dataset configuration entry in `experiments/base.yaml` specifying tokenisation settings (`pretrained_model_name`, word limits, history length).
3. Update the experiment YAML with the new dataset key and any model overrides.
4. Reuse `datasets.BaseDataset` (MIND-style).

---

## 8. Checkpoints & Logging

- Checkpoints retain model/optimizer state and the evaluation step. Only the most recent checkpoint is kept per directory.
- WandB logs include training loss, regularisation diagnostics, retrieval metrics, and lambda scheduler values when applicable.
- `utils.save_checkpoint` writes `early_stop_value = -val_auc`; early stopping monitors validation AUC.

---
