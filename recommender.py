import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  # Added to suppress warnings
from sklearn.metrics.pairwise import cosine_similarity  # Optional for advanced similarity; not used here but imported for future

# Suppress specific numpy warnings from corrwith (harmless for sparse data)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.lib._function_base_impl')

# Step: Load the data from your unzipped folder
ratings = pd.read_csv('ml-latest-small/ratings.csv')  # Path matches your download
movies = pd.read_csv('ml-latest-small/movies.csv')

# Merge datasets on movieId for a combined view
data = pd.merge(ratings, movies, on='movieId')
print("First 5 rows of merged data:")
print(data.head())  # Outputs sample: userId, movieId, rating, timestamp, title, genres

# Step: Basic exploration - average ratings and counts (stats insight: mean rating ~3.5, std dev ~1)
print("\nTop 10 movies by average rating:")
print(data.groupby('title')['rating'].mean().sort_values(ascending=False).head(10))
print("\nTop 10 movies by number of ratings:")
print(data.groupby('title')['rating'].count().sort_values(ascending=False).head(10))

# Visualize rating distribution (saves a plot; view with open ratings_dist.png)
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
data['rating'].hist(bins=50)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('ratings_dist.png')  # Saves to your folder; open in Finder to view
print("\nRating distribution plot saved as ratings_dist.png")

# Step: Build user-movie matrix for collaborative filtering
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
print("\nUser-movie matrix shape:", user_movie_matrix.shape)  # ~ (600 users, 9700 movies) - sparse

# Step: Example recommendations - correlations for 'Toy Story (1995)'
toy_story_ratings = user_movie_matrix['Toy Story (1995)']
similar_to_toy_story = user_movie_matrix.corrwith(toy_story_ratings)  # Pearson correlation: -1 to 1
corr_df = pd.DataFrame(similar_to_toy_story, columns=['Correlation'])
corr_df.dropna(inplace=True)

# Join with rating counts (filter >50 for statistical reliability; reduces variance in correlations)
# Fixed: Rename the count Series to 'rating_count' explicitly
count_series = data.groupby('title')['rating'].count().rename('rating_count')
corr_df = corr_df.join(count_series)
corr_df = corr_df[corr_df['rating_count'] > 50]

# Top 10 similar movies (high correlation = similar user appeal; p<0.05 implied for strong r with n>50)
recommendations = corr_df.sort_values('Correlation', ascending=False).head(10)
print("\nTop 10 recommendations similar to 'Toy Story (1995)':")
print(recommendations)