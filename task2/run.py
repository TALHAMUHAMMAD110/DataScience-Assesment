"""
run.py

Main pipeline script for RFQ similarity analysis and supplier data cleaning assessment.

This script:
1. Loads and cleans RFQ + reference datasets.
2. Joins and enriches RFQs with grade properties.
3. Handles missing values and parses ranges.
4. Computes similarity between RFQs and outputs top-3 matches.
5. Performs ablation study (dimensions vs grade vs categorical).
6. Evaluates alternative similarity metrics (cosine + Jaccard + IoU).
7. Groups RFQs into clusters using KMeans.

Outputs are stored in the `outputs/` folder.
"""

import pandas as pd
import numpy as np
from utils import (
    normalize_grade, parse_range, compute_similarity,
    alt_similarity, cluster_rfq
)

# -----------------------
# Load Data into DataFrames for Processing
# -----------------------
rfq = pd.read_csv("data/rfq.csv")
ref = pd.read_csv("data/reference_properties.tsv", sep="\t")


# -----------------------
# Task B.1 — Reference join & missing values
# -----------------------

# Normalize grade keys
rfq["grade_norm"] = rfq["grade"].apply(normalize_grade)
ref["grade_norm"] = ref["Grade/Material"].apply(normalize_grade)

# Join RFQ with reference
rfq_ref = rfq.merge(ref, on="grade_norm", how="left")


# -----------------------
# Task B.2 — Feature engineering
# -----------------------

# Parse numeric ranges into min/max/mid
for col in ["Tensile strength (Rm)", "Yield strength (Re or Rp0.2)"]:
    rfq_ref[[col+"_min", col+"_max"]] = rfq_ref[col].apply(lambda x: pd.Series(parse_range(x)))
    rfq_ref[col+"_mid"] = rfq_ref[[col+"_min", col+"_max"]].mean(axis=1)

# Impute missing midpoints with category median
for col in ["Tensile strength (Rm)_mid", "Yield strength (Re or Rp0.2)_mid"]:
    rfq_ref[col] = rfq_ref.groupby("Category")[col].transform(lambda x: x.fillna(x.median()))

# -----------------------
# Task B.3: Top-3 Similarity (baseline)
# -----------------------
results = []
for i, row in rfq_ref.iterrows():
    sims = []
    for j, row2 in rfq_ref.iterrows():
        if i == j:
            continue
        sim = compute_similarity(row, row2)
        sims.append((row["id"], row2["id"], sim))
    top3 = sorted(sims, key=lambda x: x[2], reverse=True)[:3]
    results.extend(top3)

top3_df = pd.DataFrame(results, columns=["rfq_id", "match_id", "similarity_score"])
top3_df.to_csv("outputs/top3.csv", index=False)

# -----------------------------------------------------------------
# Bonus / Stretch Goals 
# ---------------------------------------------------------------


# Bonus 1: Ablation Study

ablation_results = []
configs = {
    "dims_only": dict(use_dims=True, use_cats=False, use_grade=False),
    "grade_only": dict(use_dims=False, use_cats=False, use_grade=True),
    "cats_only": dict(use_dims=False, use_cats=True, use_grade=False),
    "all": dict(use_dims=True, use_cats=True, use_grade=True),
}

for name, cfg in configs.items():
    for i, row in rfq_ref.iterrows():
        sims = []
        for j, row2 in rfq_ref.iterrows():
            if i == j:
                continue
            sim = compute_similarity(row, row2, **cfg)
            sims.append((row["id"], row2["id"], sim, name))
        ablation_results.extend(sims)

pd.DataFrame(
    ablation_results,
    columns=["rfq_id", "match_id", "similarity", "config"]
).to_csv("outputs/ablation_results.csv", index=False)

# ----------------------------------------------------------------
# Bonus 2: Alternative Metrics
# ---------------------------------------------------------------
alt_results = []
for i, row in rfq_ref.iterrows():
    for j, row2 in rfq_ref.iterrows():
        if i == j:
            continue
        sim = alt_similarity(row, row2)
        alt_results.append((row["id"], row2["id"], sim))

pd.DataFrame(
    alt_results,
    columns=["rfq_id", "match_id", "alt_similarity"]
).to_csv("outputs/alt_similarity.csv", index=False)

# -----------------------
# Bonus 3: Clustering
# -----------------------
rfq_clustered, model = cluster_rfq(rfq_ref, k=5)
rfq_clustered[["id", "cluster"]].to_csv("outputs/rfq_clusters.csv", index=False)

print("✅ Pipeline complete. Results saved in outputs/")
