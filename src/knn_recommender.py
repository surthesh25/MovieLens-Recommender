# =============================================================================
# MovieLens KNN-Based Movie Recommender System
# =============================================================================
# This script builds a collaborative filtering recommender system using the
# K-Nearest Neighbors (KNN) algorithm on the MovieLens ml-latest-small dataset.
#
# Workflow:
#   1. Load and merge ratings + movies data
#   2. Filter movies with fewer than 50 ratings
#   3. Split data into 90% train / 10% test
#   4. Build a user-item matrix from training data
#   5. Predict ratings using Unweighted and Weighted KNN
#   6. Save predictions to CSV for k = 3, 5, 10
#   7. Evaluate predictions using RMSE
#   8. Visualize RMSE vs K values
#   9. Generate top-N movie recommendations for a given user
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────

from IPython.display import display
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean


# ── 1. Load Data ─────────────────────────────────────────────────────────────

movies  = pd.read_csv("Dataset/Movies.csv",  usecols=['movieId', 'title'])
ratings = pd.read_csv("Dataset/Ratings.csv", usecols=['userId', 'movieId', 'rating'])

print("Movies sample:")
print(movies.head(), "\n")

print("Ratings sample:")
print(ratings.head(), "\n")


# ── 2. Merge & Filter ────────────────────────────────────────────────────────

# Merge ratings with movie titles
data = pd.merge(ratings, movies, on='movieId')

# Drop rows with missing titles
movies_ratings = data.dropna(axis=0, subset=['title'])

# Count total ratings per movie
ratings_count = (
    movies_ratings
    .groupby('title')['rating']
    .count()
    .reset_index()
    .rename(columns={'rating': 'totalRatingCount'})
)

# Attach rating counts back to the main dataframe
data = movies_ratings.merge(ratings_count, on='title', how='left')

# Keep only movies with at least 50 ratings (reduces noise)
RATING_THRESHOLD = 50
data = data.query('totalRatingCount >= @RATING_THRESHOLD')

print(f"Dataset shape after filtering (>= {RATING_THRESHOLD} ratings): {data.shape}\n")


# ── 3. Train / Test Split ────────────────────────────────────────────────────

# 90% training — 10% testing, fixed seed for reproducibility
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

print(f"Training set size : {train_data.shape}")
print(f"Test set size     : {test_data.shape}\n")


# ── 4. User-Item Matrix ──────────────────────────────────────────────────────
# Rows = users, Columns = movies, Values = ratings (NaN where not rated)

user_item_matrix = train_data.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)


# ── 5a. Unweighted KNN Prediction ────────────────────────────────────────────

def unweighted_prediction(user_id, movie_id, k, user_item_matrix):
    """
    Predict a user's rating for a movie using unweighted KNN.

    Finds the k nearest neighbors (by Euclidean distance over shared ratings)
    and returns the simple average of their ratings for the target movie.

    Parameters
    ----------
    user_id         : int   – Target user ID
    movie_id        : int   – Target movie ID
    k               : int   – Number of nearest neighbors
    user_item_matrix: DataFrame – User × Movie rating matrix

    Returns
    -------
    float – Predicted rating (0.0 if prediction is not possible)
    """
    if movie_id not in user_item_matrix.columns:
        return 0.0

    neighbors = []

    for neighbor_id in user_item_matrix.index:
        if neighbor_id == user_id:
            continue
        if pd.isna(user_item_matrix.loc[neighbor_id, movie_id]):
            continue  # neighbor hasn't rated the target movie

        # Only compare over movies both users have rated
        common = user_item_matrix.loc[[user_id, neighbor_id]].dropna(axis=1)
        if common.shape[1] == 0:
            continue

        dist = euclidean(common.loc[user_id], common.loc[neighbor_id])
        neighbors.append((neighbor_id, dist))

    # Select k closest neighbors
    neighbors = sorted(neighbors, key=lambda x: x[1])[:k]
    ratings   = [user_item_matrix.loc[nid, movie_id] for nid, _ in neighbors]

    if not ratings:
        return 0.0
    return round(sum(ratings) / len(ratings), 1)


# ── 5b. Weighted KNN Prediction ──────────────────────────────────────────────

def weighted_prediction(user_id, movie_id, k, user_item_matrix):
    """
    Predict a user's rating for a movie using distance-weighted KNN.

    Closer neighbors (smaller Euclidean distance) receive higher weights
    (weight = 1 / distance). The prediction is a weighted average.

    Parameters
    ----------
    user_id         : int   – Target user ID
    movie_id        : int   – Target movie ID
    k               : int   – Number of nearest neighbors
    user_item_matrix: DataFrame – User × Movie rating matrix

    Returns
    -------
    float – Predicted rating (0.0 if prediction is not possible)
    """
    if movie_id not in user_item_matrix.columns:
        return 0.0

    neighbors = []

    for neighbor_id in user_item_matrix.index:
        if neighbor_id == user_id:
            continue
        if pd.isna(user_item_matrix.loc[neighbor_id, movie_id]):
            continue

        common = user_item_matrix.loc[[user_id, neighbor_id]].dropna(axis=1)
        if common.shape[1] == 0:
            continue

        dist = euclidean(common.loc[user_id], common.loc[neighbor_id])
        neighbors.append((neighbor_id, dist))

    neighbors = sorted(neighbors, key=lambda x: x[1])[:k]

    weighted_ratings = []
    weights          = []

    for neighbor_id, dist in neighbors:
        weight = 1.0 if dist == 0 else 1.0 / dist
        weighted_ratings.append(user_item_matrix.loc[neighbor_id, movie_id] * weight)
        weights.append(weight)

    if not weighted_ratings:
        return 0.0
    return round(sum(weighted_ratings) / sum(weights), 1)


# ── 6. Generate & Save Predictions ───────────────────────────────────────────

def generate_predictions(test_data, k, method, user_item_matrix):
    """
    Run the KNN model over the test set and collect predictions.

    Parameters
    ----------
    test_data        : DataFrame – Test split with userId, movieId, rating columns
    k                : int       – Number of nearest neighbors
    method           : str       – 'unweighted' or 'weighted'
    user_item_matrix : DataFrame – User × Movie rating matrix (built from train data)

    Returns
    -------
    list of tuples – (userId, movieId, realRating, predictedRating)
    """
    predictions = []
    predict_fn  = unweighted_prediction if method == 'unweighted' else weighted_prediction

    for _, row in test_data.iterrows():
        user_id, movie_id, real_rating = row['userId'], row['movieId'], row['rating']
        predicted = predict_fn(user_id, movie_id, k, user_item_matrix)
        predictions.append((user_id, movie_id, real_rating, predicted))

    return predictions


def save_predictions(predictions, method, k):
    """
    Save prediction results to a CSV file under the Results/ directory.

    File naming convention:
        ratings_predictions_10_knn_<uw|wt>_k<k>.csv

    Parameters
    ----------
    predictions : list of tuples – Output of generate_predictions()
    method      : str            – 'unweighted' or 'weighted'
    k           : int            – Number of nearest neighbors used
    """
    abbreviation = "uw" if method == "unweighted" else "wt"
    save_dir     = "Results"
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"ratings_predictions_10_knn_{abbreviation}_k{k}.csv")

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        f.write("userId,movieId,realRating,predictedRating\n")
        for row in predictions:
            f.write(",".join(map(str, row)) + "\n")

    print(f"Saved → {filename}")


# Run for all combinations of k and method
K_VALUES = [3, 5, 10]
METHODS  = ['unweighted', 'weighted']

for k in K_VALUES:
    for method in METHODS:
        preds = generate_predictions(test_data, k, method, user_item_matrix)
        save_predictions(preds, method, k)


# ── 7. RMSE Evaluation ───────────────────────────────────────────────────────

def calculate_rmse(file_path):
    """
    Calculate Root Mean Squared Error (RMSE) from a predictions CSV file.

    Parameters
    ----------
    file_path : str – Path to the predictions CSV

    Returns
    -------
    float – RMSE value
    """
    df = pd.read_csv(file_path)
    return np.sqrt(mean_squared_error(df['realRating'], df['predictedRating']))


# Evaluate all saved prediction files
prediction_files_flat = [
    "Results/ratings_predictions_10_knn_uw_k3.csv",
    "Results/ratings_predictions_10_knn_wt_k3.csv",
    "Results/ratings_predictions_10_knn_uw_k5.csv",
    "Results/ratings_predictions_10_knn_wt_k5.csv",
    "Results/ratings_predictions_10_knn_uw_k10.csv",
    "Results/ratings_predictions_10_knn_wt_k10.csv",
]

print("\nRMSE Results:")
print("-" * 55)
for file in prediction_files_flat:
    rmse = calculate_rmse(file)
    print(f"  {os.path.basename(file):<45} {rmse:.4f}")


# ── 8. RMSE Visualisation ────────────────────────────────────────────────────

prediction_files_grouped = {
    "unweighted": [
        "Results/ratings_predictions_10_knn_uw_k3.csv",
        "Results/ratings_predictions_10_knn_uw_k5.csv",
        "Results/ratings_predictions_10_knn_uw_k10.csv",
    ],
    "weighted": [
        "Results/ratings_predictions_10_knn_wt_k3.csv",
        "Results/ratings_predictions_10_knn_wt_k5.csv",
        "Results/ratings_predictions_10_knn_wt_k10.csv",
    ],
}

rmse_results = {method: [calculate_rmse(f) for f in files]
                for method, files in prediction_files_grouped.items()}

plt.figure(figsize=(8, 6))
plt.plot(K_VALUES, rmse_results['unweighted'], marker='o', label='Unweighted KNN')
plt.plot(K_VALUES, rmse_results['weighted'],   marker='o', label='Weighted KNN')
plt.title('RMSE vs K Values for KNN Predictions')
plt.xlabel('K (Number of Neighbours)')
plt.ylabel('RMSE')
plt.xticks(K_VALUES)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Results/rmse_vs_k.png", dpi=150)
plt.show()

# Key finding:
#   Both methods improve (lower RMSE) as k increases.
#   Unweighted KNN slightly outperforms weighted KNN across all k values.
#   Best configuration: Unweighted KNN with k=10 (RMSE ≈ 0.9328).


# ── 9. Movie Recommendations ─────────────────────────────────────────────────

def recommend_movies(user_id, predictions_file, user_item_matrix, movies_df, n=10):
    """
    Generate top-N movie recommendations for a user.

    Filters out movies the user has already rated, then ranks the remaining
    predictions by predicted rating (descending).

    Parameters
    ----------
    user_id          : int       – Target user ID
    predictions_file : str       – Path to the predictions CSV
    user_item_matrix : DataFrame – Training user-item matrix
    movies_df        : DataFrame – Movies metadata (movieId, title)
    n                : int       – Number of recommendations to return

    Returns
    -------
    DataFrame with columns: Movie ID, Real Rating, Predicted Rating, Movie Title
    """
    preds_df = pd.read_csv(predictions_file)
    user_preds = preds_df[preds_df['userId'] == user_id]

    if user_preds.empty:
        print(f"No predictions found for user {user_id}.")
        return pd.DataFrame(columns=['Movie ID', 'Predicted Rating', 'Movie Title'])

    # Exclude movies the user has already rated
    already_rated = user_item_matrix.loc[user_id].dropna().index.tolist()
    user_preds = user_preds[~user_preds['movieId'].isin(already_rated)]

    if user_preds.empty:
        print(f"User {user_id} has already rated all predicted movies.")
        return pd.DataFrame(columns=['Movie ID', 'Predicted Rating', 'Movie Title'])

    top_n = user_preds.sort_values('predictedRating', ascending=False).head(n)

    recommendations = (
        top_n
        .merge(movies_df[['movieId', 'title']], on='movieId', how='left')
        .rename(columns={
            'movieId':          'Movie ID',
            'realRating':       'Real Rating',
            'predictedRating':  'Predicted Rating',
            'title':            'Movie Title',
        })
    )

    return recommendations[['Movie ID', 'Real Rating', 'Predicted Rating', 'Movie Title']]


# Example: top-10 recommendations for User 1 using the best model (unweighted, k=10)
BEST_PREDICTIONS_FILE = "Results/ratings_predictions_10_knn_uw_k10.csv"
TARGET_USER           = 1

recs = recommend_movies(TARGET_USER, BEST_PREDICTIONS_FILE, user_item_matrix, movies)
display(f"Top-10 recommendations for User {TARGET_USER}:", recs)
