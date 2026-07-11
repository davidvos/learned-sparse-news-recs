"""Scan all val users and rank by sparse-profile semantic quality.

Useful for finding a compelling User C to replace in the qualitative table.
Prints the top candidates per dominant category, ranked by the fraction of
top-20 tokens that are either lexical matches or thematically coherent with
the user's click history (based on a light keyword heuristic).

Note:
    Requires an existing LSNR checkpoint at the hardcoded path.

Usage (from project root):
    python src/find_good_profile.py [--category foodanddrink] [--top 15]
"""
import argparse
import sys
import os
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent))
from config import load_config
from evaluate import build_news_vectors
from utils import set_seeds, get_device, load_model_and_checkpoint

CHECKPOINT = (
    "lsnr_attention_pe_notopk_low_reg_lsnr_mind_cold0_lr1e-05_ln0.005"
    "_lh0.0005_lu0.0_hls1_attention_pe1"
)
DATASET = "mind"
MODEL_TYPE = "lsnr"
TOP_K = 20
MIN_HISTORY = 8   # ≥8 val-set articles so category estimate is reliable
MAX_HISTORY = 100
MIN_DOM_PCT = 0.40

# Tokens that are almost always noise regardless of topic
ALWAYS_NOISE = {
    "librarian", "demolished", "pupils", "sulfur", "removed",
    "intervention", "cemetery", "[sep]", "[mask]", "[pad]", "[cls]",
    "asteroid", "earthquake", "hurricane", "tornado",
    "##trust", "##gled", "##nted", "##stein",
}

# Per-category keyword sets for semantic scoring
CATEGORY_KEYWORDS = {
    "foodanddrink": {
        "food", "eat", "eating", "recipe", "cook", "cooking", "kitchen",
        "restaurant", "menu", "meal", "dinner", "lunch", "breakfast",
        "chicken", "beef", "pork", "fish", "seafood", "pizza", "burger",
        "sandwich", "pasta", "bread", "cake", "pie", "cookie", "dessert",
        "chocolate", "cheese", "milk", "cream", "egg", "fruit", "vegetable",
        "salad", "soup", "sauce", "flavor", "spice", "coffee", "tea",
        "wine", "beer", "drink", "beverage", "fast", "chef", "bake",
        "grill", "fry", "roast", "snack", "taste", "tasty", "delicious",
        "pumpkin", "tomato", "onion", "garlic", "pepper", "apple", "bacon",
        "sausage", "turkey", "avocado", "calorie", "grocery", "oven",
        "ingredient", "flour", "rice", "bean", "herb", "broth", "gravy",
    },
    "sports": {
        "sport", "game", "team", "player", "coach", "season", "league",
        "win", "loss", "score", "goal", "point", "match", "tournament",
        "championship", "playoff", "draft", "trade", "contract", "nfl",
        "nba", "mlb", "nhl", "soccer", "football", "basketball", "baseball",
        "tennis", "golf", "racing", "athlete", "stadium", "arena", "field",
        "court", "pitch", "ball", "run", "pass", "throw", "kick", "shot",
        "quarter", "inning", "set", "round", "medal", "olympic", "record",
        "marathon", "sprint", "swimming", "gymnastics", "boxing", "wrestling",
        "quarterback", "pitcher", "forward", "defender", "midfielder",
        "rookie", "veteran", "injured", "injury", "roster", "lineup",
    },
    "finance": {
        "money", "bank", "stock", "market", "invest", "fund", "share",
        "price", "tax", "economy", "economic", "debt", "loan", "credit",
        "interest", "rate", "federal", "reserve", "inflation", "budget",
        "salary", "wage", "income", "profit", "revenue", "loss", "asset",
        "portfolio", "bond", "yield", "dividend", "merger", "acquisition",
        "ipo", "wall", "street", "hedge", "venture", "startup", "billion",
        "million", "trillion", "rich", "wealth", "wealthy", "finance",
        "financial", "banking", "insurance", "mortgage", "real", "estate",
        "retail", "consumer", "trade", "commerce", "business", "corporate",
    },
    "entertainment": {
        "movie", "film", "actor", "actress", "director", "star", "stars",
        "celebrity", "famous", "tv", "television", "show", "series",
        "music", "song", "singer", "band", "album", "concert", "award",
        "oscar", "emmy", "grammy", "drama", "comedy", "thriller", "horror",
        "romance", "box", "office", "netflix", "disney", "hollywood",
        "premiere", "trailer", "episode", "season", "character", "role",
        "cast", "talent", "artist", "performer", "stage", "theater",
        "fashion", "style", "beauty", "celebrity", "instagram", "viral",
    },
    "health": {
        "health", "medical", "doctor", "hospital", "patient", "disease",
        "cancer", "heart", "brain", "blood", "body", "weight", "diet",
        "exercise", "fitness", "mental", "anxiety", "depression", "drug",
        "medicine", "treatment", "therapy", "symptom", "risk", "study",
        "research", "vaccine", "virus", "bacteria", "infection", "immune",
        "sleep", "stress", "pain", "chronic", "care", "insurance",
        "surgery", "procedure", "diagnosis", "prevention", "wellness",
        "vitamin", "supplement", "nutrition", "calorie", "protein", "fat",
    },
    "lifestyle": {
        "home", "house", "design", "decor", "style", "fashion", "trend",
        "travel", "vacation", "trip", "hotel", "resort", "beach", "city",
        "family", "parent", "child", "relationship", "wedding", "marriage",
        "pet", "dog", "cat", "garden", "outdoor", "fitness", "yoga",
        "meditation", "mindful", "wellness", "beauty", "skin", "hair",
        "career", "work", "life", "balance", "inspiration", "motivation",
        "hobby", "craft", "diy", "art", "creative", "social",
    },
    "autos": {
        "car", "cars", "vehicle", "truck", "suv", "sedan", "electric",
        "engine", "model", "brand", "toyota", "ford", "gm", "honda",
        "tesla", "bmw", "mercedes", "audi", "drive", "driving", "speed",
        "fuel", "gas", "hybrid", "mpg", "horsepower", "torque", "wheel",
        "tire", "brake", "safety", "recall", "dealership", "dealer",
        "price", "sale", "lease", "insurance", "repair", "maintenance",
        "race", "racing", "track", "auto", "motor", "motorcycle",
    },
    "news": {
        "president", "government", "congress", "senate", "law", "policy",
        "election", "vote", "democrat", "republican", "white", "house",
        "pentagon", "federal", "state", "city", "mayor", "governor",
        "trial", "court", "judge", "verdict", "military", "war", "peace",
        "international", "foreign", "trade", "sanction", "treaty",
        "climate", "environment", "energy", "immigration", "border",
        "police", "crime", "shooting", "protest", "rally", "speech",
    },
    "tv": {
        "tv", "television", "show", "series", "episode", "season", "cast",
        "character", "plot", "drama", "comedy", "reality", "network",
        "abc", "nbc", "cbs", "fox", "hbo", "netflix", "streaming",
        "premiere", "finale", "ratings", "viewer", "audience", "host",
        "anchor", "broadcast", "cable", "channel",
    },
}


def compute_semantic_score(tokens, weights, dom_cat, click_text_lower):
    """Return (semantic_frac, noise_frac, lexical_matches, quality)."""
    cat_kws = CATEGORY_KEYWORDS.get(dom_cat.lower(), set())
    total_w = sum(weights) + 1e-9
    sem_score = noise_score = 0.0
    lexical = 0

    for tok, w in zip(tokens, weights):
        tok_bare = tok.lower().lstrip("#")
        frac = w / total_w
        if tok_bare in click_text_lower or tok.lower() in click_text_lower:
            lexical += 1
        # Check against category keyword set (substring match)
        if any(kw in tok_bare or tok_bare in kw for kw in cat_kws):
            sem_score += frac
        if tok_bare in ALWAYS_NOISE:
            noise_score += frac

    quality = sem_score + 0.15 * lexical / len(tokens) - 2.0 * noise_score
    return sem_score, noise_score, lexical, quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="Filter to this dominant category")
    parser.add_argument("--top", type=int, default=15, help="Show this many candidates")
    parser.add_argument("--exclude", nargs="*", default=["737", "1252"],
                        help="Val-row indices to exclude (already used in table)")
    args = parser.parse_args()

    config = load_config("config.yaml")
    config["global"]["dataset"] = DATASET
    config["global"]["model_type"] = MODEL_TYPE
    config["global"]["debug_mode"] = False

    # Force no top-k at inference so we see the full vector, then
    # pick top-20 ourselves after filtering generic tokens.
    config["models"]["lsnr"]["topk_sparsity_queries"] = None

    device = get_device("auto")
    set_seeds(config["global"]["seed"])
    ds_cfg = config["datasets"][DATASET]

    # ── Load news metadata ───────────────────────────────────────────────────
    print("Loading news metadata...")
    val_news_path = f"{ds_cfg['path']}/val/news.tsv"
    news_df = pd.read_table(
        val_news_path, header=None,
        names=["id", "category", "subcategory", "title", "abstract", "url", "te", "ae"],
        usecols=["id", "category", "title", "abstract"],
    )
    news_df["title"] = news_df["title"].fillna("")
    news_df["abstract"] = news_df["abstract"].fillna("")
    news_meta = {
        row.id: {"cat": row.category.lower(), "title": row.title, "abstract": row.abstract}
        for row in news_df.itertuples()
    }

    # ── Load behaviors ───────────────────────────────────────────────────────
    print("Loading behaviors...")
    beh_df = pd.read_table(
        f"{ds_cfg['path']}/val/behaviors_parsed.tsv",
        names=["impression_id", "user", "time", "clicked_news",
               "impressions", "candidate_news", "clicked"],
        usecols=range(7),
    )
    beh_df["clicked_news"] = beh_df["clicked_news"].fillna("").str.split()
    val_news_ids = set(news_meta.keys())
    # Filter using val-set article count (excludes train-era IDs not in val news)
    beh_df["val_hist_len"] = beh_df["clicked_news"].apply(
        lambda ids: sum(1 for nid in ids if nid in val_news_ids)
    )
    beh_df = beh_df[(beh_df["val_hist_len"] >= MIN_HISTORY) & (beh_df["val_hist_len"] <= MAX_HISTORY)]
    beh_df = beh_df.reset_index(drop=True)

    # For each val-row index, compute dominant category (val-set articles only)
    candidates = []
    for row_idx, row in beh_df.iterrows():
        if str(row_idx) in (args.exclude or []):
            continue
        hist = row["clicked_news"]
        # Count categories only for val-set articles (what will actually drive the user vector)
        val_hist = [nid for nid in hist if nid in val_news_ids]
        cats = [news_meta.get(nid, {}).get("cat", "unknown") for nid in val_hist]
        cat_ctr = Counter(cats)
        total = sum(cat_ctr.values())
        if total == 0:
            continue
        dom_cat, dom_cnt = cat_ctr.most_common(1)[0]
        dom_pct = dom_cnt / total
        if dom_pct < MIN_DOM_PCT:
            continue
        if args.category and dom_cat.lower() != args.category.lower():
            continue
        candidates.append({
            "row_idx": row_idx,
            "user": row["user"],
            "hist": hist,
            "dom_cat": dom_cat,
            "dom_pct": dom_pct,
            "hist_len": len(val_hist),  # usable val-set articles
            "cat_ctr": cat_ctr,
        })

    print(f"Candidates (dom_pct >= {MIN_DOM_PCT:.0%}): {len(candidates)}")
    cat_dist = Counter(c["dom_cat"] for c in candidates)
    print("Category distribution:", dict(sorted(cat_dist.items(), key=lambda x: -x[1])))

    # ── Load model and news vectors ──────────────────────────────────────────
    print("\nLoading model...")
    model, _, _ = load_model_and_checkpoint(
        MODEL_TYPE, DATASET, config, device,
        checkpoint_dir=f"./checkpoints/{CHECKPOINT}",
    )
    model.eval()

    print("Building news vectors...")
    news2vector, _ = build_news_vectors(model, config, device, "val")

    tokenizer = AutoTokenizer.from_pretrained(ds_cfg["pretrained_model_name"])
    id2token = {v: k for k, v in tokenizer.get_vocab().items()}

    # ── Identify generic tokens (high-frequency across all users) ────────────
    # Sample a random subset of users to find tokens that appear in most profiles.
    print("\nIdentifying generic tokens from a sample of users...")
    import random
    random.seed(42)
    sample_size = min(500, len(candidates))
    sample_cands = random.sample(candidates, sample_size)
    num_news = int(ds_cfg.get("num_clicked_news_a_user", 30))
    token_user_count: Counter = Counter()

    for cand in tqdm(sample_cands, desc="Sampling for generic tokens"):
        hist_valid = [nid for nid in cand["hist"] if nid in news2vector]
        if not hist_valid:
            continue
        hist_padded = hist_valid[-num_news:]
        pad_len = num_news - len(hist_padded)
        mask = [1] * len(hist_padded) + [0] * pad_len
        hist_padded = hist_padded + ["PADDED_NEWS"] * pad_len
        with torch.no_grad():
            user_history = (
                torch.stack([news2vector[nid][1] for nid in hist_padded]).unsqueeze(0).to(device),
                torch.stack([news2vector[nid][0] for nid in hist_padded]).unsqueeze(0).to(device),
            )
            mask_t = torch.tensor([mask], dtype=torch.float).to(device)
            uv = model.get_user_vector(user_history, mask=mask_t)
            uv = uv[0].detach().cpu().numpy()
        for idx in np.nonzero(uv)[0]:
            token_user_count[id2token.get(int(idx), f"unk_{idx}")] += 1

    # Tokens present in >30% of sampled users are generic
    generic_threshold = 0.30 * sample_size
    generic_tokens: set = {
        tok for tok, cnt in token_user_count.items() if cnt >= generic_threshold
    }
    print(f"Found {len(generic_tokens)} generic tokens (present in >{generic_threshold:.0f} of {sample_size} users)")
    print("Top 20 generic:", [t for t, _ in token_user_count.most_common(20)])

    # ── Score each candidate ─────────────────────────────────────────────────
    print(f"\nScoring {len(candidates)} candidates...")
    scored = []

    for cand in tqdm(candidates):
        hist_valid = [nid for nid in cand["hist"] if nid in news2vector]
        if len(hist_valid) < 3:
            continue
        hist_padded = hist_valid[-num_news:]
        pad_len = num_news - len(hist_padded)
        mask = [1] * len(hist_padded) + [0] * pad_len
        hist_padded = hist_padded + ["PADDED_NEWS"] * pad_len

        with torch.no_grad():
            user_history = (
                torch.stack([news2vector[nid][1] for nid in hist_padded]).unsqueeze(0).to(device),
                torch.stack([news2vector[nid][0] for nid in hist_padded]).unsqueeze(0).to(device),
            )
            mask_t = torch.tensor([mask], dtype=torch.float).to(device)
            uv = model.get_user_vector(user_history, mask=mask_t)
            uv = uv[0].detach().cpu().numpy()

        nz = np.nonzero(uv)[0]
        vals = uv[nz]
        order = np.argsort(-vals)
        # Filter generic tokens before selecting top-20
        filtered = [
            (nz[i], vals[i]) for i in order
            if id2token.get(int(nz[i]), "") not in generic_tokens
        ]
        top_idx = np.array([i for i, _ in filtered[:TOP_K]])
        top_vals = np.array([v for _, v in filtered[:TOP_K]])
        top_tokens = [id2token.get(int(i), f"unk_{i}") for i in top_idx]

        click_texts = [
            news_meta.get(nid, {}).get("title", "") + " " + news_meta.get(nid, {}).get("abstract", "")
            for nid in hist_valid
        ]
        click_text_lower = " ".join(click_texts).lower()

        sem, noise, lex, quality = compute_semantic_score(
            top_tokens, top_vals.tolist(), cand["dom_cat"], click_text_lower
        )

        # Extra metric: fraction of post-filter tokens that are either lexical or
        # thematic (as opposed to appearing to be generic or off-topic noise).
        tok_words = [t.lower().lstrip("#") for t in top_tokens]
        cat_kws = CATEGORY_KEYWORDS.get(cand["dom_cat"].lower(), set())
        coherent = sum(
            1 for w in tok_words
            if w in click_text_lower or any(kw in w or w in kw for kw in cat_kws)
        )
        coherence_frac = coherent / max(len(top_tokens), 1)

        scored.append({
            **cand,
            "tokens": top_tokens,
            "weights": top_vals.tolist(),
            "click_texts": click_texts,
            "click_text_lower": click_text_lower,
            "sem_score": sem,
            "noise_score": noise,
            "lexical": lex,
            "quality": quality,
            "coherence_frac": coherence_frac,
        })

    scored.sort(key=lambda x: x["quality"], reverse=True)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    target_cat = args.category or "ALL CATEGORIES"
    print(f"TOP {args.top} CANDIDATES — {target_cat.upper()} (ranked by quality)")
    print("=" * 72)

    for rank, r in enumerate(scored[:args.top], 1):
        profile_parts = []
        for tok, w in zip(r["tokens"], r["weights"]):
            is_lex = tok.lower().lstrip("#") in r["click_text_lower"] or tok.lower() in r["click_text_lower"]
            flag = "*" if is_lex else ""
            profile_parts.append(f"{flag}{tok}({w:.2f})")
        profile_str = ", ".join(profile_parts)

        print(f"\n#{rank}  row_idx={r['row_idx']}  cat={r['dom_cat']}({r['dom_pct']:.0%})"
              f"  hist={r['hist_len']}  sem={r['sem_score']:.3f}"
              f"  noise={r['noise_score']:.3f}  lex={r['lexical']}"
              f"  coherence={r['coherence_frac']:.2f}  Q={r['quality']:.3f}")
        print(f"   cats: {dict(sorted(r['cat_ctr'].items(), key=lambda x: -x[1]))}")
        print(f"   4 clicks: {[t[:80] for t in r['click_texts'][:4]]}")
        # Print full profile on separate line to avoid truncation
        print(f"   profile ({len(r['tokens'])} tokens):")
        print(f"     {profile_str}")

    # ── LaTeX snippet for best candidate ────────────────────────────────────
    if scored:
        best = scored[0]
        print("\n\n" + "=" * 72)
        print("LaTeX snippet for best candidate:")
        print("=" * 72)
        click_text_lower = best["click_text_lower"]
        tok_parts = []
        for tok, w in zip(best["tokens"], best["weights"]):
            is_lex = tok.lower().lstrip("#") in click_text_lower or tok.lower() in click_text_lower
            entry = f"\\textbf{{{tok}}}\\,({w:.2f})" if is_lex else f"{tok}\\,({w:.2f})"
            tok_parts.append(entry)
        print(f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{User C}} \\hfill"
              f" idx {best['row_idx']} \\quad $\\bullet$ \\quad"
              f" {best['hist_len']} articles \\quad $\\bullet$ \\quad"
              f" {best['dom_pct']:.0%} {best['dom_cat'].title()}}} \\\\")
        print("\\midrule")
        print("Clicks &")
        for t in best["click_texts"][:4]:
            print(f"\\textit{{{t[:120].strip()}}}.")
        print("\\\\")
        print("\\addlinespace[2pt]")
        print("Profile &")
        print(", ".join(tok_parts) + " \\\\")


if __name__ == "__main__":
    main()
