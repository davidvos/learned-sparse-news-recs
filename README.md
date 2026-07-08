# LSNR: Learned Sparse News Recommendations with Attentive User Modeling

Current learned sparse approaches for recommender systems, such as STAR, represent a user by concatenating their interaction history into a single query string. For news, this creates two bottlenecks: concatenation collapses a user's distinct reading interests into one flat query, and, because news articles are long-form text, concatenating several news articles quickly overruns the context window of the language model used as backbone and forces quality-degrading truncation.

To address both bottlenecks, we introduce learned sparse news recommendations (LSNR). LSNR encodes each article independently and constructs a sparse user vector by combining learned, attention-based weighting of the per-article sparse representations with a dense-projected expansion at the user level. Experiments on the MIND and EBNeRD datasets show that LSNR significantly outperforms STAR, with these gains holding even in short-history settings where truncation is not a factor. Compared to state-of-the-art dense recommenders, LSNR is competitive in standard settings and it outperforms them when the user history is short. An ablation of the user encoder confirms that learned attention-based aggregation outperforms simpler pooling alternatives. Beyond effectiveness, an additional advantage of LSNR is that it serves from an index an order of magnitude smaller than an uncompressed dense baseline.

---

## 📂 Repository Structure

```text
├── src/
│   ├── data/         # Dataset loaders and preprocessing pipelines
│   ├── model/        # LSNR, STAR, and baseline (NRMS, NAML, etc.) implementations
│   ├── train.py      # Main training entry point
│   ├── evaluate.py   # Retrieval evaluation and prediction dumping
│   ├── benchmark.py  # Index size and retrieval-latency benchmarking
│   ├── config.py     # Configuration management and argument parsing
│   └── utils.py      # Metrics, logging, and checkpointing utilities
├── config.yaml       # Central configuration for all models and datasets
└── pyproject.toml    # Environment and dependency specifications
```

---

## 🚀 Getting Started

### 1. Environment Setup
The codebase requires Python 3.10+ and uses `uv` for dependency management.

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

### 2. Data preprocessing
LSNR supports the **MIND** and **EBNeRD** datasets.

#### MIND Setup
```bash
# 1. Create the base directory
mkdir -p data/mind_small

# 2. Download and extract MIND-small (train and dev)
# Unzip should create MINDsmall_train/ and MINDsmall_dev/ inside data/mind_small/
# You can get the files from: https://msnews.github.io/
unzip MINDsmall_train.zip -d data/mind_small
unzip MINDsmall_dev.zip   -d data/mind_small

# 3. Run preprocessing pipeline
uv run python src/data/preprocess_mind.py
```

#### EBNeRD Setup
```bash
# 1. Create the directory structure
mkdir -p data/ebnerd/{train,validation}

# 2. Place the raw Parquet files
# - data/ebnerd/articles.parquet
# - data/ebnerd/train/behaviors.parquet
# - data/ebnerd/train/history.parquet
# - data/ebnerd/validation/behaviors.parquet
# - data/ebnerd/validation/history.parquet

# 3. Run preprocessing pipeline
uv run python src/data/preprocess_ebnerd.py
```

Pre-tokenizing the news content is highly recommended to avoid bottlenecks during training:

```bash
uv run python src/data/tokenize.py --dataset mind
```

Preprocessing generates standardized `behaviors_parsed.tsv` and `news.tsv` files, while the tokenization script produces `news_tokenized.pt` for accelerated training.

---


### Training
To train a model (e.g., LSNR on MIND):
```bash
uv run python src/train.py --model_type lsnr --dataset mind
```

**Key Parameters:**
- `--model_type`: Model architecture (`lsnr`, `star`, `nrms`, `naml`, `lstur`, `npa`).
- `--dataset`: Dataset type (`mind` or `ebnerd`).
- `--config_file`: Path to a custom YAML configuration (defaults to `config.yaml`).
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

### Efficiency Benchmarking
`benchmark.py` measures index size and per-query retrieval latency, building a
[Seismic](https://github.com/TusKANNy/seismic) inverted index for sparse models
(LSNR, STAR) and a [FAISS](https://github.com/facebookresearch/faiss) HNSW index
for dense baselines (NRMS, NAML, LSTUR, NPA).

```bash
uv run python src/benchmark.py --model_type lsnr --dataset mind \
    --experiment_name lsnr_mind_1e-05 --output results.json
```