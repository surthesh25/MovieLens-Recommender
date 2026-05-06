# 🎬 MovieLens Movie Recommender System

A movie recommendation system built on the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) dataset, implementing and comparing two collaborative filtering approaches:

| Model | File | Algorithm |
|---|---|---|
| **KNN** | `src/knn_recommender.py` | K-Nearest Neighbours (user-based, Euclidean distance) |
| **CFF** | `src/cff_recommender.py` | Collaborative Filtering via Ullman method (Pearson correlation) |

---

## 📁 Repository Structure

```
MovieLens-Recommender/
│
├── Dataset/                         # Raw data files (not included — see below)
│   ├── Movies.csv
│   ├── Ratings.csv
│   ├── Tags.csv
│   └── Links.csv
│
├── src/
│   ├── knn_recommender.py           # KNN model: prediction, evaluation, recommendations
│   └── cff_recommender.py           # CFF model: EDA, Ullman CF, evaluation, recommendations
│
├── Results/                         # Auto-generated on first run
│   │
│   ├── ── KNN outputs ──
│   ├── ratings_predictions_10_knn_uw_k3.csv
│   ├── ratings_predictions_10_knn_wt_k3.csv
│   ├── ratings_predictions_10_knn_uw_k5.csv
│   ├── ratings_predictions_10_knn_wt_k5.csv
│   ├── ratings_predictions_10_knn_uw_k10.csv
│   ├── ratings_predictions_10_knn_wt_k10.csv
│   ├── rmse_vs_k.png
│   │
│   ├── ── CFF outputs ──
│   ├── ratings_predictions_40_cf.csv
│   ├── ratings_predictions_30_cf.csv
│   ├── ratings_predictions_20_cf.csv
│   ├── ratings_predictions_10_cf.csv
│   ├── eda_rating_distribution.png
│   ├── eda_user_activity.png
│   ├── eda_genre_distribution.png
│   ├── cff_rmse_vs_split.png
│   └── cff_runtime_vs_split.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

This project uses the **MovieLens ml-latest-small** dataset provided by [GroupLens](https://grouplens.org/datasets/movielens/).

| File | Contents |
|---|---|
| `Movies.csv` | 9,742 movies with titles and genres |
| `Ratings.csv` | 100,836 ratings from 610 users (scale: 0.5 – 5.0 ⭐) |
| `Tags.csv` | User-generated tags for movies |
| `Links.csv` | Cross-reference IDs to IMDb and TMDb |

> **Note:** The dataset files are **not included** in this repository due to their size and licensing terms.  
> Download them from: [https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)  
> Place the CSV files inside the `Dataset/` folder before running either script.

### Citation

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*  
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.  
> https://doi.org/10.1145/2827872

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MovieLens-Recommender.git
cd MovieLens-Recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download `ml-latest-small.zip` from [GroupLens](https://grouplens.org/datasets/movielens/latest/), extract it, and place the CSV files in the `Dataset/` folder:

```
Dataset/
├── Movies.csv
├── Ratings.csv
├── Tags.csv
└── Links.csv
```

### 4. Run either model

```bash
# KNN-based recommender
python src/knn_recommender.py

# CFF (Ullman) recommender
python src/cff_recommender.py
```

The `Results/` folder is created automatically with all prediction CSVs and plots.

---

## 🤖 Model 1 — KNN Recommender (`knn_recommender.py`)

### How It Works

1. **Preprocessing** — Merge ratings with movie titles; filter movies with fewer than 50 ratings; split 90% train / 10% test
2. **User-Item Matrix** — Pivot table: rows = users, columns = movies, values = ratings
3. **Neighbour Search** — For each test sample, find k users who rated the target movie with the smallest Euclidean distance over commonly rated movies
4. **Prediction** — Two variants:
   - **Unweighted**: simple average of k neighbours' ratings
   - **Weighted**: distance-weighted average (closer neighbours contribute more)
5. **Evaluation** — RMSE computed for all 6 configurations (k ∈ {3, 5, 10} × 2 methods)
6. **Recommendations** — Filter already-seen movies; return top-N by predicted rating

### Results

| Configuration | RMSE |
|---|---|
| Unweighted KNN, k=3 | ~1.01 |
| Weighted KNN, k=3 | ~1.03 |
| Unweighted KNN, k=5 | ~0.96 |
| Weighted KNN, k=5 | ~0.98 |
| **Unweighted KNN, k=10** | **~0.93 ✅ Best** |
| Weighted KNN, k=10 | ~0.95 |

**Key findings:**
- RMSE decreases as k increases for both methods
- Unweighted KNN consistently outperforms Weighted KNN
- Best model: **Unweighted KNN with k=10**

---

## 🤖 Model 2 — CFF Recommender (`cff_recommender.py`)

### How It Works

1. **EDA** — Rating distribution, user activity, and genre breakdown plots
2. **Mean-Centering** — Movie ratings are centred around their per-movie average to remove popularity bias
3. **Similarity Matrix** — Pearson correlation computed between all movies (Ullman method)
4. **Prediction** — Similarity-weighted sum of centred ratings, shifted back by the user's mean rating
5. **Evaluation** — RMSE and runtime measured across 4 train/test splits: 60/40, 70/30, 80/20, 90/10
6. **Recommendations** — Exclude already-rated movies; return top-N from the predicted rating matrix

### Results

| Split | RMSE |
|---|---|
| 60% train / 40% test | ~1.01 |
| 70% train / 30% test | ~0.99 |
| 80% train / 20% test | ~0.97 |
| **90% train / 10% test** | **~0.95 ✅ Best** |

**Key findings:**
- RMSE decreases slightly as training set size grows
- The model is stable across all tested split ratios
- A larger training set gives more context for computing accurate similarities

---

## 🔁 Model Comparison

| | KNN | CFF (Ullman) |
|---|---|---|
| **Similarity metric** | Euclidean distance (user-based) | Pearson correlation (item-based) |
| **Best RMSE** | ~0.93 (unweighted, k=10) | ~0.95 (90/10 split) |
| **Tunable parameter** | k (number of neighbours) | Train/test split ratio |
| **Scalability** | Slower (per-sample neighbour search) | Faster (pre-computed matrix) |
| **Recommendation basis** | User similarity | Item similarity |

---

## 🧰 Tech Stack

- **Python 3.x**
- [pandas](https://pandas.pydata.org/) — data loading and manipulation
- [NumPy](https://numpy.org/) — numerical computation
- [scikit-learn](https://scikit-learn.org/) — train/test split, RMSE, pairwise distances
- [SciPy](https://scipy.org/) — Euclidean distance (KNN)
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — visualisations

---

## 📄 License

The **MovieLens dataset** is made available by GroupLens under their [usage license](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html). It may not be used for commercial purposes without prior permission.

The code in this repository is available under the [MIT License](LICENSE).
