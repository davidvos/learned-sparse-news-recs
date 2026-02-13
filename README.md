# LSNR: Learned Sparse News Recommendations with Sequential User Modeling

Official repository for the research paper **"Reliable and Controllable Learned Sparse News Recommendations for Short-History Users."**

Learned sparse retrieval (LSR) offers efficiency and interpretability advantages over dense retrieval, yet its application to news recommendations has not yet been explored. Current LSR approaches for recommender systems, such as STAR, are ill-suited for news recommendations because they treat a user's historical interaction as a single concatenated query string. This creates two bottlenecks: (i) It discards the sequential nature of news reading behavior, thus failing to capture how user interests shift over time. (ii) It quickly exceeds language model context windows, requiring truncation of user histories, which degrades recommendation quality.

To address these bottlenecks, we introduce learned sparse news recommendations (LSNR). LSNR replaces naive concatenation with a sequential user encoder that processes news articles independently. This approach explicitly models the sequential order of historical interactions and eliminates the context window bottleneck. Experiments on the MIND and EBNeRD benchmarks demonstrate that LSNR significantly outperforms STAR. These findings also hold in short-history settings, where history truncation is not a factor. We benchmark LSNR against state-of-the-art dense news recommenders, showing it remains competitive in standard settings while outperforming dense models in short-history scenarios. 

---

## 📂 Repository Structure

```text
├── experiments/      # YAML experiment definitions and model hyperparameters
├── src/
│   ├── data/         # Dataset loaders and preprocessing pipelines
│   ├── model/        # LSNR, STAR, and baseline (NRMS, NAML, etc.) implementations
│   ├── train.py      # Main training entry point (procedural)
│   ├── evaluate.py   # Comprehensive evaluation and prediction dumping
│   ├── config.py     # Configuration management and argument parsing
│   └── utils.py      # Metrics, logging, and checkpointing utilities
├── config.yaml       # Global defaults and dataset paths
└── pyproject.toml    # Environment and dependency specifications
```

---

## 🚀 Getting Started

### 1. Environment Setup
The codebase requires Python 3.10+ and uses `uv` for high-performance dependency management.

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

Logging is integrated with [Weights & Biases](https://wandb.ai/). Ensure you are logged in:
```bash
uv run wandb login
```

### 2. Data Acquisition & Preprocessing
LSNR supports the **MIND** (Microsoft News Dataset) and **EBNeRD** datasets.

#### MIND Setup
```bash
# 1. Download and extract MIND (small/large as needed)
mkdir -p data/mind/{train,val,test}
# ... download scripts ...

# 2. Run preprocessing pipeline
uv run python src/data_preprocess_mind.py
```

#### EBNeRD Setup
```bash
uv run python src/data_preprocess_ebnerd.py --dataset ebnerd
```

Pre-tokenizing the news content is highly recommended to avoid bottlenecks during training:

```bash
uv run python src/data/tokenize.py --dataset mind
```

Preprocessing generates standardized `behaviors_parsed.tsv` and `news.tsv` files, while the tokenization script produces `news_tokenized.pt` for accelerated training.

---

## 🧪 Running Experiments

Experiments are managed via YAML files in `experiments/`, supporting inheritance and parameter overrides.

### Training
To train a model (e.g., LSNR on MIND):
```bash
uv run python src/train.py --config_file experiments/lsnr_baseline.yaml --model_type lsnr --dataset mind
```

**Key Parameters:**
- `--config_file`: Path to experiment YAML.
- `--model_type`: Model architecture (`lsnr`, `star`, `nrms`, `naml`, `lstur`, `npa`).
- `--debug`: Enable debug mode with minimal data for verification.

### Evaluation
The evaluation suite provides retrieval metrics (AUC, MRR, nDCG@k) and sparsity analysis.

```bash
uv run python src/evaluate.py --experiment_name lsnr_baseline --model_type lsnr --dataset mind
```

For large-scale ranking, you can dump predictions to JSON:
```bash
uv run python src/evaluate.py --eval_mode dump_predictions --output_path predictions.json
```

---

## 🛠 Configuration System

The `src/config.py` module handles complex configuration merging:
- **Inheritance**: Configs can inherit from a base file (e.g., `_base_: base.yaml`).
- **Dynamic Overrides**: All training hyperparameters can be overridden via CLI flags.
- **Scientific Notation**: Strings like `1e-4` in YAML are automatically converted to floats.

---

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{lsnr2026,
  title={Reliable and Controllable Learned Sparse News Recommendations for Short-History Users},
  author={...},
  booktitle={Proceedings of the SIGIR},
  year={2026}
}
```

---
*This repository is maintained for academic reproducibility and research extension.*
