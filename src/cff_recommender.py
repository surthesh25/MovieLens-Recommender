# =============================================================================
# MovieLens Collaborative Filtering Recommender (Ullman / Pearson Method)
# =============================================================================
# This script builds a model-based collaborative filtering recommender using
# the Ullman method (Pearson correlation similarity) on the MovieLens dataset.
#
# Workflow:
#   1.  Load and explore ratings & movies data (EDA)
#   2.  Define similarity (Pearson correlation) and prediction functions
#   3.  Run the model across 4 train/test splits: 60/40, 70/30, 80/20, 90/10
#   4.  Save predictions to CSV for each split
#   5.  Evaluate using RMSE and measure runtime
#   6.  Visualise Runtime vs Split Ratio and RMSE vs Split Ratio
#   7.  Generate top-N movie recommendations for a given user
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────

import os
import time
import warnings
from collections import defaultdict
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# ── 1. Load Data ─────────────────────────────────────────────────────────────

ratings = pd.read_csv("Dataset/Ratings.csv")
movies  = pd.read_csv("Dataset/Movies.csv")

print("Ratings sample:")
print(ratings.head(), "\n")

print("Movies sample:")
print(movies.head(), "\n")


# ── 2. Exploratory Data Analysis ─────────────────────────────────────────────

# --- Summary statistics ---
print("=== Ratings Info ===")
ratings.info()
print(ratings.describe(), "\n")

print("=== Movies Info ===")
movies.info()

# --- Unique counts ---
print(f"\nUnique users  : {ratings['userId'].nunique()}")
print(f"Unique movies : {ratings['movieId'].nunique()}")
print(f"Unique genres : {movies['genres'].nunique()}")

# --- Missing values ---
print("\nMissing values in ratings:")
print(ratings.isnull().sum())
print("\nMissing values in movies:")
print(movies.isnull().sum())

# --- Rating distribution ---
sns.histplot(ratings['rating'], bins=5, kde=False, color='lightgreen')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("Results/eda_rating_distribution.png", dpi=150)
plt.show()

# --- Top movies by rating count ---
movie_ratings_count = ratings.groupby('movieId')['rating'].count().sort_values(ascending=False)
print("\nTop 5 most-rated movies (by count):")
print(movie_ratings_count.head())

# --- Top movies by average rating ---
movie_ratings_avg = ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False)
print("\nTop 5 highest-rated movies (by average):")
print(movie_ratings_avg.head())

# --- Merge with movie titles for readability ---
popular_movies = movies.merge(movie_ratings_count, on='movieId', how='left')
popular_movies.columns = ['movieId', 'title', 'genres', 'rating_count']
print("\nPopular movies with rating counts:")
display(popular_movies.head())

# --- User activity distribution ---
user_ratings_count = ratings.groupby('userId')['rating'].count()
sns.histplot(user_ratings_count, bins=30, kde=False, color='green')
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("Results/eda_user_activity.png", dpi=150)
plt.show()

# --- Genre distribution ---
movies['genres_list'] = movies['genres'].str.split('|')
all_genres   = movies.explode('genres_list')['genres_list']
genre_count  = all_genres.value_counts()

genre_count.plot(kind='bar', color='purple', figsize=(10, 5))
plt.title('Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("Results/eda_genre_distribution.png", dpi=150)
plt.show()


# ── 3. Core Model Functions (Ullman / Pearson Method) ────────────────────────

def calculation_ullman(movie_centered):
    """
    Compute the movie-movie similarity matrix using Pearson correlation.

    NaN values (from empty columns) are replaced with 0 to ensure stable
    matrix operations.

    Parameters
    ----------
    movie_centered : np.ndarray – Mean-centred user-item matrix (users × movies)

    Returns
    -------
    np.ndarray – Symmetric similarity matrix (movies × movies)
    """
    similarity_matrix = np.corrcoef(movie_centered.T)
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
    return similarity_matrix


def prediction_ullman(similarity_matrix, movie_centered, user_means):
    """
    Predict ratings using the Ullman weighted-sum method.

    Each prediction is a similarity-weighted sum of the mean-centred ratings,
    normalised by the total absolute weight, then shifted back by the user mean.

    Parameters
    ----------
    similarity_matrix : np.ndarray – Movie-movie similarity matrix
    movie_centered    : np.ndarray – Mean-centred user-item matrix
    user_means        : pd.Series  – Per-user mean ratings

    Returns
    -------
    np.ndarray – Predicted rating matrix (users × movies)
    """
    weighted_sum   = similarity_matrix.dot(movie_centered.T)
    sum_of_weights = np.abs(similarity_matrix).sum(axis=1, keepdims=True)
    sum_of_weights[sum_of_weights == 0] = 1e-10   # avoid division by zero

    predictions        = weighted_sum / sum_of_weights
    user_means_aligned = user_means.values[:, np.newaxis]

    return predictions.T + user_means_aligned


def calculate_rmse(y_true, y_pred):
    """
    Compute RMSE between true and predicted ratings, ignoring NaN values.

    Parameters
    ----------
    y_true : np.ndarray – Ground-truth ratings (may contain NaN)
    y_pred : np.ndarray – Predicted ratings    (may contain NaN)

    Returns
    -------
    float – RMSE value
    """
    mask             = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered  = y_true[mask]
    y_pred_filtered  = y_pred[mask]
    return sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))


# ── 4. Run Model Across Multiple Train/Test Splits ───────────────────────────
# Splits tested: 60/40, 70/30, 80/20, 90/10
# For each split: train → similarity matrix → predict → save CSV → compute RMSE

os.makedirs("Results", exist_ok=True)

SPLITS   = [0.4, 0.3, 0.2, 0.1]   # test_size values
runtimes = []
rmses    = []

for test_size in SPLITS:
    train_pct = int((1 - test_size) * 100)
    test_pct  = int(test_size * 100)
    print(f"\n{'='*60}")
    print(f"  Split: {train_pct}% train / {test_pct}% test")
    print(f"{'='*60}")

    start_time = time.time()

    # --- Train/test split ---
    train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=5)

    # --- Build mean-centred user-item matrix from training data ---
    movie_averages = train_df.groupby('movieId')['rating'].mean()
    train_pivot    = train_df.pivot(index='userId', columns='movieId', values='rating')
    movie_centered = train_pivot.sub(movie_averages, axis=1)
    user_means     = train_pivot.mean(axis=1).fillna(0)

    # --- Compute similarity and predict ---
    similarity_matrix  = calculation_ullman(movie_centered.fillna(0))
    test_pivot         = test_df.pivot(index='userId', columns='movieId', values='rating')

    # For 90/10 split, align test pivot to training index before predicting
    if test_size == 0.1:
        test_pivot_aligned      = test_pivot.reindex_like(train_pivot)
        test_predictions        = prediction_ullman(similarity_matrix, movie_centered.fillna(0), user_means)
        test_predictions_df     = pd.DataFrame(test_predictions, index=train_pivot.index, columns=train_pivot.columns)
        test_predictions_lookup = test_predictions_df.reindex_like(test_pivot_aligned)
    else:
        test_predictions    = prediction_ullman(similarity_matrix, movie_centered.fillna(0), user_means)
        test_predictions_df = pd.DataFrame(test_predictions, index=test_pivot.index, columns=train_pivot.columns)
        test_predictions_lookup = test_predictions_df

    # --- Map predictions back to the test dataframe rows ---
    test_df['predicted_rating'] = test_df.apply(
        lambda x: test_predictions_lookup.loc[x['userId'], x['movieId']]
        if x['movieId'] in test_predictions_lookup.columns
        and x['userId']  in test_predictions_lookup.index
        else np.nan,
        axis=1
    )
    test_df['predicted_rating'] = test_df['predicted_rating'].round(1)

    # --- Save predictions ---
    filename = f"Results/ratings_predictions_{test_pct}_cf.csv"
    test_df.to_csv(filename, index=False)
    print(f"Saved predictions → {filename}")

    # --- Compute RMSE ---
    test_pivot_masked        = test_pivot.where(~test_pivot.isna(), np.nan)
    test_predictions_masked  = test_predictions_df.reindex_like(test_pivot_masked)
    rmse = calculate_rmse(
        test_pivot_masked.values.flatten(),
        test_predictions_masked.values.flatten()
    )
    rmses.append(rmse)
    print(f"RMSE: {rmse:.4f}")

    # --- Runtime ---
    runtime = time.time() - start_time
    runtimes.append(runtime)
    print(f"Runtime: {runtime:.2f} seconds")


# ── 5. Summary Table ─────────────────────────────────────────────────────────

split_labels = ["60/40", "70/30", "80/20", "90/10"]
print("\n=== Results Summary ===")
print(f"{'Split':<10} {'RMSE':>8} {'Runtime (s)':>12}")
print("-" * 32)
for label, rmse, rt in zip(split_labels, rmses, runtimes):
    print(f"{label:<10} {rmse:>8.4f} {rt:>12.2f}")


# ── 6. Visualisations ────────────────────────────────────────────────────────

split_ratios = [40, 30, 20, 10]   # test % (x-axis label)

# --- Runtime vs Split Ratio ---
plt.figure(figsize=(8, 5))
plt.scatter(split_ratios, runtimes)
plt.plot(split_ratios, runtimes)
plt.grid(True)
plt.xlabel('Test Split Ratio (%)')
plt.ylabel('Runtime (seconds)')
plt.title('Test Split Ratio vs Runtime')
plt.tight_layout()
plt.savefig("Results/cff_runtime_vs_split.png", dpi=150)
plt.show()

# Key finding:
#   Runtime varies across split sizes. Larger test sets (40%) require the model
#   to generate predictions for more users, which can increase computation time.

# --- RMSE vs Split Ratio ---
plt.figure(figsize=(8, 5))
plt.scatter(split_ratios, rmses)
plt.plot(split_ratios, rmses)
plt.grid(True)
plt.xlabel('Test Split Ratio (%)')
plt.ylabel('RMSE')
plt.title('Test Split Ratio vs RMSE')
plt.tight_layout()
plt.savefig("Results/cff_rmse_vs_split.png", dpi=150)
plt.show()

# Key finding:
#   RMSE decreases slightly as the training set grows (smaller test split).
#   A larger training set gives the model more data to learn similarities,
#   leading to marginally better predictions. The model is fairly stable
#   across all tested split ratios.


# ── 7. Movie Recommendations ─────────────────────────────────────────────────

def recommend_movies(user_id, train_pivot, predictions_df, movies_df, n=10):
    """
    Generate top-N movie recommendations for a user using CF predictions.

    Excludes movies the user already rated in the training set, then ranks
    remaining movies by predicted rating (descending).

    Parameters
    ----------
    user_id        : int       – Target user ID
    train_pivot    : DataFrame – User-item matrix from training data
    predictions_df : DataFrame – Full predicted rating matrix (users × movies)
    movies_df      : DataFrame – Movies metadata (movieId, title)
    n              : int       – Number of recommendations to return

    Returns
    -------
    DataFrame with columns: Movie ID, Predicted Rating, Movie Title
    """
    if user_id not in train_pivot.index:
        print(f"User {user_id} not found in training data.")
        return pd.DataFrame(columns=['Movie ID', 'Predicted Rating', 'Movie Title'])

    # Exclude already-rated movies
    already_rated  = train_pivot.loc[user_id].dropna().index.tolist()
    user_preds     = predictions_df.loc[user_id].drop(already_rated, errors='ignore')

    top_n = (
        user_preds
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={'movieId': 'movieId', user_id: 'predicted_rating'})
    )

    # The reset_index gives columns: movieId, <user_id>
    top_n.columns = ['movieId', 'Predicted Rating']

    recommendations = (
        top_n
        .merge(movies_df[['movieId', 'title']], on='movieId', how='left')
        .rename(columns={'movieId': 'Movie ID', 'title': 'Movie Title'})
    )

    return recommendations[['Movie ID', 'Predicted Rating', 'Movie Title']]


# Example: top-10 recommendations for User 5 using the 90/10 split predictions
TARGET_USER = 5
recs = recommend_movies(TARGET_USER, train_pivot, test_predictions_df, movies)
display(f"Top-10 recommendations for User {TARGET_USER}:", recs)
