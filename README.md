# LSNR: Learned Sparse News Recommendations with Attentive User Modeling

Current learned sparse approaches for recommender systems, such as STAR, represent a user by concatenating their interaction history into a single query string. For news, this creates two bottlenecks: concatenation collapses a user's distinct reading interests into one flat query, and, because news articles are long-form text, concatenating several news articles quickly overruns the context window of the language model used as backbone and forces quality-degrading truncation.

To address both bottlenecks, we introduce **Learned Sparse News Recommendations (LSNR)**. LSNR encodes each article independently and constructs a sparse user vector by combining learned, attention-based weighting of the per-article sparse representations with a dense-projected expansion at the user level. Experiments on the MIND and EBNeRD datasets show that LSNR significantly outperforms STAR, with these gains holding even in short-history settings where truncation is not a factor. Compared to state-of-the-art dense recommenders, LSNR is competitive in standard settings and outperforms them when the user history is short. An ablation of the user encoder confirms that learned attention-based aggregation outperforms simpler pooling alternatives. Beyond effectiveness, LSNR serves from an index an order of magnitude smaller than an uncompressed dense baseline.

---

## Repository Structure

```text
├── src/
│   ├── data/
│   │   ├── dataset.py           # Dataset loaders for training and evaluation
│   │   ├── preprocess_mind.py   # MIND preprocessing pipeline
│   │   ├── preprocess_ebnerd.py # EBNeRD preprocessing pipeline
│   │   └── tokenize.py          # Pre-tokenization for faster training
│   ├── model/
│   │   ├── LSNR/                # LSNR news encoder, user encoder, and model
│   │   ├── STAR/                # STAR sparse bi-encoder baseline
│   │   ├── baselines.py         # Dense baselines: NRMS, NAML, LSTUR, NPA
│   │   ├── base.py              # Abstract base model
│   │   └── layers.py            # Shared attention and prediction layers
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Retrieval evaluation and prediction export
│   ├── benchmark.py             # Index size and retrieval latency benchmarking
│   ├── config.py                # Configuration loading and CLI parsing
│   └── utils.py                 # Metrics, checkpointing, and logging utilities
├── scripts/
│   ├── train_lsnr.sh            # Train LSNR (recommended hyperparameters)
│   └── train_star.sh            # Train STAR baseline
├── plots/                       # Generated result figures
├── config.yaml                  # Central configuration for all models and datasets
├── pyproject.toml               # Project metadata and dependencies
└── LICENSE                      # MIT License
```

---

## Requirements

The codebase requires **Python 3.10+**. Dependencies are managed via `uv`, but standard `pip` can also be used. Key dependencies include:
- `torch`
- `transformers`
- `pyseismic-lsr` (for fast sparse retrieval)
- `faiss-cpu` (for dense baselines)
- `wandb` (for experiment tracking)

---

## Getting Started

### 1. Environment Setup

Using `uv` for dependency management:

```bash
uv sync
source .venv/bin/activate
```

Logging is integrated with [Weights & Biases](https://wandb.ai/):

```bash
uv run wandb login
```

### 2. Data Preprocessing

LSNR supports the **MIND** and **EBNeRD** datasets.

#### MIND

```bash
mkdir -p data/mind_small

# Download MINDsmall_train.zip and MINDsmall_dev.zip from https://msnews.github.io/
unzip MINDsmall_train.zip -d data/mind_small
unzip MINDsmall_dev.zip   -d data/mind_small

uv run python src/data/preprocess_mind.py
```

#### EBNeRD

```bash
mkdir -p data/ebnerd/{train,validation}

# Place the raw Parquet files:
#   data/ebnerd/articles.parquet
#   data/ebnerd/train/behaviors.parquet
#   data/ebnerd/train/history.parquet
#   data/ebnerd/validation/behaviors.parquet
#   data/ebnerd/validation/history.parquet

uv run python src/data/preprocess_ebnerd.py
```

Pre-tokenizing the news corpus is strongly recommended to avoid I/O bottlenecks during training:

```bash
uv run python src/data/tokenize.py --dataset mind
```

Preprocessing produces standardized `behaviors_parsed.tsv` and `news.tsv` files; tokenization produces `news_tokenized.pt` for accelerated data loading.

---

### 3. Training

```bash
# LSNR (paper hyperparameters)
bash scripts/train_lsnr.sh mind      # or ebnerd

# STAR baseline
bash scripts/train_star.sh mind

# Any baseline (NRMS, NAML, LSTUR, NPA)
uv run python src/train.py --model_type nrms --dataset mind
```

Key CLI flags (see `src/train.py --help` for the full list):

| Flag | Description |
|---|---|
| `--model_type` | `lsnr`, `star`, `nrms`, `naml`, `lstur`, `npa` |
| `--dataset` | `mind` or `ebnerd` |
| `--lambda_n` | FLOPS regularizer weight for candidate news |
| `--lambda_history` | FLOPS regularizer weight for history news |
| `--lambda_u` | Regularizer weight for user vectors |
| `--topk_sparsity_queries` | Hard top-k constraint on user query vectors |
| `--user_aggregation` | History aggregation: `attention`, `average`, `max`, `decay` |
| `--debug` | Run with minimal data for a quick sanity check |

---

### 4. Evaluation

```bash
uv run python src/evaluate.py \
    --experiment_name lsnr_mind \
    --model_type lsnr \
    --dataset mind \
    --split test
```

To export ranked predictions for submission:

```bash
uv run python src/evaluate.py \
    --eval_mode dump_predictions \
    --output_path predictions.json \
    --model_type lsnr \
    --dataset mind
```

---

### 5. Efficiency Benchmarking

Measures index size and retrieval latency (Seismic for sparse models, FAISS HNSW for dense models):

```bash
uv run python src/benchmark.py \
    --model_type lsnr \
    --dataset mind \
    --experiment_name lsnr_mind \
    --output results.json
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
