import pandas as pd

# Load the datasets
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
# Merge the datasets
merged_df = pd.merge(ratings, movies, on='movieId')
# Save to a new CSV file
merged_df.to_csv('ml-latest-small/merged_data.csv', index=False)

# Or use the merged DataFrame for further analysis
# For example, displaying the first few rows
print(merged_df.head())
