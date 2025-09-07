# 🏗️ RFQ Similarity & Supplier Data Cleaning

This repository contains my solution to the **Vanilla Steel Data Challenge**.  
It covers both scenarios from the assessment:

- **Scenario A**: Supplier data cleaning & joining.
- **Scenario B**: RFQ similarity enrichment, feature engineering, similarity scoring, and clustering.

---

## 📂 Project Structure

```
rfq_similarity/
│── run.py                  # Main pipeline script
│── utils.py                # Helper functions (parsing, similarity, clustering)
│── README.md               # Documentation
│── requirements.txt        # Dependencies
│── data/
│    ├── rfq.csv
│    ├── reference_properties.tsv
│    ├── supplier_data1.xlsx    # (Task A)
│    ├── supplier_data2.xlsx    # (Task A)
│── notebooks/
│    ├── supplier_cleaning.ipynb  # Data cleaning for Task A
│── outputs/
│    ├── inventory_dataset.csv    # Cleaned supplier data (Task A)
│    ├── top3.csv                 # Top-3 RFQ similarities (Task B.3)
│    ├── ablation_results.csv     # Ablation study (Bonus)
│    ├── alt_similarity.csv       # Alternative similarity metric (Bonus)
│    ├── rfq_clusters.csv         # Clustering results (Bonus)
```

---

## ⚡ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies:

- pandas
- numpy
- scikit-learn
- (optional) jupyter, openpyxl

### 2. Run pipeline

```bash
python run.py
```

All results will be stored in the `outputs/` folder.

---

## 📝 Scenario A — Supplier Data Cleaning

- **Objective:** Clean & join `supplier_data1.xlsx` and `supplier_data2.xlsx`.
- **Steps:**
  - Standardized column names (thickness, width, weight).
  - Normalized formats (numbers, units).
  - Filled missing values using median/mean by material category.
  - Joined into **inventory_dataset.csv**.

📌 Deliverable: `outputs/inventory_dataset.csv`

---

## 📝 Scenario B — RFQ Similarity

### Task B.1 — Reference Join & Missing Values

- Normalized steel grade keys (case, spaces, suffixes).
- Parsed ranges (`200–300 MPa`, `≤250`, `≥500`) into numeric min/max.
- Computed midpoints for tensile/yield strength.
- Missing midpoints imputed with **median within Category**.

### Task B.2 — Feature Engineering

- **Dimensions** → Represented as intervals with IoU (Intersection-over-Union).
- **Categorical** → Exact match (1/0) for coating, finish, form, surface_type.
- **Grade properties** → Used numeric midpoints.

### Task B.3 — Similarity Calculation

- Aggregate similarity = weighted average of:
  - IoU overlap (dimensions).
  - Categorical matches.
  - Grade property similarity.
- Produced **top-3 most similar RFQs per line**.

📌 Deliverable: `outputs/top3.csv`

### Task B.4 — Pipeline

- Implemented in `run.py` (single reproducible script).
- Modularized with `utils.py`.

---

## 🎯 Bonus Goals

### 🔬 Ablation Analysis

- Compared similarity using subsets of features:
  - Dimensions only
  - Grade only
  - Categorical only
  - All features
- 📌 Deliverable: `outputs/ablation_results.csv`

### 📐 Alternative Metrics

- Hybrid similarity:
  - **Cosine similarity** for numeric grade properties.
  - **Jaccard similarity** for categorical features.
  - **IoU** for dimensions.
- 📌 Deliverable: `outputs/alt_similarity.csv`

### 🧩 Clustering

- Built feature matrix (grade midpoints + one-hot categorical).
- Scaled features, applied **KMeans clustering**.
- Interpreted RFQ families (structural steels, coated sheets, high-strength grades).
- 📌 Deliverable: `outputs/rfq_clusters.csv`

---

## 📊 Example Outputs

**Top-3 Similar RFQs (top3.csv):**
| rfq_id | match_id | similarity_score |
|--------|----------|------------------|
| 1 | 42 | 0.87 |
| 1 | 57 | 0.82 |
| 1 | 9 | 0.80 |

**Clusters (rfq_clusters.csv):**
| id | cluster |
|-----|---------|
| 1 | 0 |
| 2 | 0 |
| 3 | 2 |

---

## ✅ Evaluation Checklist

- [x] Supplier data cleaned & joined (Task A).
- [x] Reference join + missing values (Task B.1).
- [x] Feature engineering (Task B.2).
- [x] Similarity definition + top-3 output (Task B.3).
- [x] Pipeline reproducibility (Task B.4).
- [x] Bonus: Ablation, Alternative Metrics, Clustering.

---

## 📌 Notes

- This solution is designed for clarity and reproducibility.
- Missing values are handled by **median imputation per Category** or flagged as unknown.
- Similarity can be easily extended with additional weights or metrics.
