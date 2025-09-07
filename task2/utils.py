"""
utils.py

Helper functions for RFQ similarity pipeline:
- Data cleaning (normalize grade names, parse ranges).
- Similarity metrics (IoU, cosine, Jaccard, hybrid).
- Clustering (KMeans).
"""

import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

# -----------------------
# Normalization Functions
# -----------------------

def normalize_grade(grade):
    """
    Normalize steel grade strings for joining RFQs with reference data.

    Parameters
    ----------
    g : str or None
        Grade string (e.g., "S235 JR", " s235jr ")

    Returns
    -------
    str or None
        Normalized grade (e.g., "S235JR"), or None if input invalid.
    """
    if grade is None or str(grade).lower() == "nan":
        return None
    grade = str(grade).upper().strip()
    grade = grade.replace("-", "").replace(" ", "")
    return grade


def parse_range(val):
    """
    Parse numeric range strings into min/max tuple.

    Examples
    --------
    "200-300 MPa" -> (200.0, 300.0)
    "≤250 MPa"    -> (0.0, 250.0)
    "≥500 MPa"    -> (500.0, nan)
    "950 MPa"     -> (950.0, 950.0)

    Parameters
    ----------
    val : str or None
        Raw string with range or numeric value.

    Returns
    -------
    tuple(float, float)
        (min, max) values. np.nan if parsing fails.
    """
    if val is None or str(val).lower() == "nan":
        return (np.nan, np.nan)

    s = str(val).strip()
    # Remove units (MPa, %, HB, HV, etc.)
    s = re.sub(r"[A-Za-z%]", "", s).strip()

    if "-" in s:
        try:
            a, b = s.split("-", 1)
            return (float(a.strip()), float(b.strip()))
        except:
            return (np.nan, np.nan)
    if s.startswith("≤"):
        return (0.0, float(s[1:].strip()))
    if s.startswith("≥"):
        return (float(s[1:].strip()), np.nan)
    try:
        return (float(s), float(s))
    except:
        return (np.nan, np.nan)

# -----------------------
# Similarity Metrics
# -----------------------

def interval_iou(a_min, a_max, b_min, b_max):
    """
    Compute interval overlap using Intersection-over-Union (IoU).

    Parameters
    ----------
    a_min, a_max, b_min, b_max : float
        Interval boundaries.

    Returns
    -------
    float
        IoU score in [0, 1], or NaN if invalid.
    """
    if np.isnan(a_min) or np.isnan(b_min):
        return np.nan
    inter = max(0, min(a_max, b_max) - max(a_min, b_min))
    union = max(a_max, b_max) - min(a_min, b_min)
    return inter / union if union > 0 else 0


def compute_similarity(r1, r2, use_dims=True, use_cats=True, use_grade=True, weights=None):
    """
    Compute similarity score between two RFQs.

    Combines:
    - Interval overlap (IoU) for dimensions.
    - Exact match for categorical features.
    - Distance-based similarity for grade midpoints.

    Parameters
    ----------
    r1, r2 : pd.Series
        RFQ rows.
    use_dims, use_cats, use_grade : bool
        Toggle feature groups.
    weights : dict
        Feature group weights, e.g. {"dims": 1, "cats": 1, "grade": 1}

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    score_parts = []
    weights = weights or {"dims": 1, "cats": 1, "grade": 1}

    # Dimensions
    if use_dims:
        t_iou = interval_iou(r1["thickness_min"], r1["thickness_max"],
                             r2["thickness_min"], r2["thickness_max"])
        if not np.isnan(t_iou):
            score_parts.append((t_iou, weights["dims"]))

    # Categorical
    if use_cats:
        for col in ["coating", "finish", "form", "surface_type"]:
            score_parts.append((1.0 if r1[col] == r2[col] else 0.0, weights["cats"]))

    # Grade properties
    if use_grade:
        for col in ["Tensile strength (Rm)_mid", "Yield strength (Re or Rp0.2)_mid"]:
            if not (np.isnan(r1[col]) or np.isnan(r2[col])):
                diff = abs(r1[col] - r2[col])
                sim = 1 / (1 + diff / 100)  # scale similarity
                score_parts.append((sim, weights["grade"]))

    if not score_parts:
        return 0
    weighted_sum = sum(v * w for v, w in score_parts)
    total_w = sum(w for _, w in score_parts)
    return weighted_sum / total_w


def alt_similarity(r1, r2):
    """
    Alternative similarity metric combining:
    - Cosine similarity for numeric grade properties.
    - Jaccard similarity for categorical features.
    - IoU for dimensions.

    Parameters
    ----------
    r1, r2 : pd.Series
        RFQ rows.

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    vec1 = [r1["Tensile strength (Rm)_mid"], r1["Yield strength (Re or Rp0.2)_mid"]]
    vec2 = [r2["Tensile strength (Rm)_mid"], r2["Yield strength (Re or Rp0.2)_mid"]]
    if not any(np.isnan(vec1)) and not any(np.isnan(vec2)):
        num_sim = cosine_similarity([vec1], [vec2])[0][0]
    else:
        num_sim = 0

    set1 = {r1["coating"], r1["finish"], r1["form"], r1["surface_type"]}
    set2 = {r2["coating"], r2["finish"], r2["form"], r2["surface_type"]}
    cat_sim = len(set1 & set2) / len(set1 | set2)

    dim_sim = interval_iou(r1["thickness_min"], r1["thickness_max"],
                           r2["thickness_min"], r2["thickness_max"])

    return 0.4 * num_sim + 0.3 * cat_sim + 0.3 * (dim_sim if not np.isnan(dim_sim) else 0)

# -----------------------
# Clustering
# -----------------------

def cluster_rfq(rfq_ref, k=5):
    """
    Cluster RFQs into families using KMeans.

    Parameters
    ----------
    rfq_ref : pd.DataFrame
        Enriched RFQ dataset with numeric and categorical features.
    k : int
        Number of clusters.

    Returns
    -------
    (pd.DataFrame, KMeans)
        DataFrame with 'cluster' column, fitted KMeans model.
    """
    num_feats = rfq_ref[["Tensile strength (Rm)_mid", "Yield strength (Re or Rp0.2)_mid"]].fillna(0)
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    cat_feats = ohe.fit_transform(rfq_ref[["coating", "finish", "form", "surface_type"]].fillna("Unknown"))

    X = np.hstack([num_feats.values, cat_feats])
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42)
    rfq_ref["cluster"] = kmeans.fit_predict(X_scaled)
    return rfq_ref, kmeans
