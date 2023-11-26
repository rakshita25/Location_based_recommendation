# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:21:49 2023

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('modified_merged_data.csv')

# Preprocessing: fill or drop missing values as per requirement
df.fillna(0, inplace=True)  # Example: fill missing values with 0

# Creating user-item matrix
user_movie_matrix = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Apply matrix factorization
svd = TruncatedSVD(n_components=10)
latent_matrix = svd.fit_transform(user_movie_matrix)

print("SVD Done")

# Convert to DataFrame for easier handling
user_features = pd.DataFrame(latent_matrix, index=user_movie_matrix.index)

# Combine genre and location data into a single string column
#df['content_features'] = df['genres'] + ' ' + df['address']

# Creating a TF-IDF Vectorizer
#tfidf = TfidfVectorizer(stop_words='english', min_df=0.01)

#tfidf_matrix = tfidf.fit_transform(df['content_features'])

#print(tfidf_matrix.shape)
# Compute cosine similarity matrix
#cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#print("Cosine similarity Done") 

# Create a mapping of movie ID to index in the DataFrame
#movie_idx = pd.Series(df.index, index=df['movieId']).drop_duplicates()

def get_recommendations(user_id, num_recommendations=5):
    # Get user features
    user_vector = user_features.loc[user_id].values.reshape(1, -1)
    
    # Calculate similarity with other users
    user_similarity = cosine_similarity(user_vector, latent_matrix)
    
    # Sort similar users and extract top N similar users
    similar_users = user_similarity.argsort().flatten()[-num_recommendations:]

    # Get movie preferences of similar users
    similar_users_preferences = user_movie_matrix.iloc[similar_users].mean(axis=0)
    similar_users_preferences.sort_values(ascending=False, inplace=True)

    # Get top N movies from similar users preferences
    top_movies_from_similar_users = similar_users_preferences.head(num_recommendations).index

    # Get content-based recommendations
    content_recommendations = []
    while 1==0:
        for movie_id in top_movies_from_similar_users:
            idx = movie_idx[movie_id]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:num_recommendations]
            movie_indices = [i[0] for i in sim_scores]
            content_recommendations.extend(movie_indices)

    # Combine and deduplicate recommendations
    combined_recommendations = set(top_movies_from_similar_users).union(set(content_recommendations))
    
    # Return recommended movie IDs
    return list(combined_recommendations)[:num_recommendations]

# Example Usage
recommended_movies = get_recommendations(user_id=1)
print(recommended_movies)

from sklearn.metrics import precision_score, recall_score, f1_score

# Assume you have a test set with user-item interactions for evaluation
# You may split your existing dataset into training and testing sets, or have a separate evaluation dataset

# Example: Create a test set by randomly selecting some ratings from the original dataset
test_set = user_movie_matrix.sample(frac=0.2, random_state=42)

from sklearn.metrics import accuracy_score

# Modify the evaluate_recommendations function to include accuracy
def evaluate_recommendations(test_set, user_features, latent_matrix, num_recommendations=5):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    rmse_scores = []

    for user_id in test_set.index:
        # Get actual movies liked by the user in the test set
        actual_ratings = test_set.loc[user_id].values.reshape(1, -1)  # Reshape to match the shape of predicted_ratings

        # Get predicted ratings (or recommendations)
        user_vector = user_features.loc[user_id].values.reshape(1, -1)
        predicted_ratings = np.dot(user_vector, latent_matrix.T)

        # Find indices of top N recommendations
        top_recommendations = np.argsort(predicted_ratings)[0][::-1][:num_recommendations]

        # Evaluate precision, recall, and F1-score
        true_positives = len(set(top_recommendations) & set(np.where(actual_ratings > 0)[1]))
        false_positives = len(set(top_recommendations) - set(np.where(actual_ratings > 0)[1]))
        false_negatives = len(set(np.where(actual_ratings > 0)[1]) - set(top_recommendations))

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(actual_ratings[:len(predicted_ratings)], predicted_ratings))

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        rmse_scores.append(rmse)

    # Calculate average metrics over all users
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_rmse = sum(rmse_scores) / len(rmse_scores)

    return avg_precision, avg_recall, avg_f1, avg_rmse

# Example usage
avg_precision, avg_recall, avg_f1, avg_rmse = evaluate_recommendations(test_set, user_features, latent_matrix)
print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall: {avg_recall:.2f}")
print(f"Average F1-score: {avg_f1:.2f}")
print(f"Average RMSE: {avg_rmse:.2f}")
