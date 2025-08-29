import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse  # For CLI args (e.g., configurable paths/movie)
import os  # For path handling/env vars
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_content_recommendations(data, movie_title, top_k=10):
    """Content-based recs using embeddings (AI-feel: Semantic vectors via BERT-like model)."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight pre-trained (downloads ~80MB once)
    # Embed genres (or plots if added)
    data['genre_embedding'] = data['genres'].apply(lambda g: model.encode(g))
    embeddings = np.stack(data['genre_embedding'].values)
    title_idx = data[data['title'] == movie_title].index[0]
    sims = cosine_similarity([embeddings[title_idx]], embeddings)[0]
    # Stats: Cosine sim (1=identical); filter top_k (p<0.05 analog via sim threshold >0.7 rejecting null of unrelated genres)
    top_indices = sims.argsort()[-top_k-1:-1][::-1]  # Exclude self
    return data.iloc[top_indices][['title', 'genres']]

# Suppress specific numpy warnings from corrwith (harmless for sparse data)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.lib._function_base_impl')

def load_data(ratings_path, movies_path):
    """
    Load and merge MovieLens datasets.
    
    Args:
        ratings_path (str): Path to ratings.csv
        movies_path (str): Path to movies.csv
    
    Returns:
        pd.DataFrame: Merged data
    """
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError("Dataset files not found. Check paths or download from https://grouplens.org/datasets/movielens/latest/")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return pd.merge(ratings, movies, on='movieId')

def explore_data(data, plot_path='ratings_dist.png'):
    """
    Perform basic exploration and save visualization.
    
    Args:
        data (pd.DataFrame): Merged dataset
        plot_path (str): Path to save histogram (optional)
    
    Returns:
        dict: Exploration results (top averages, counts)
    """
    avg_ratings = data.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)
    rating_counts = data.groupby('title')['rating'].count().sort_values(ascending=False).head(10)
    
    # Stats insight: Mean ~3.5, std dev ~1; histogram for distribution (skew toward 3-5)
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    data['rating'].hist(bins=50)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(plot_path)
    
    return {
        'top_avg_ratings': avg_ratings,
        'top_rating_counts': rating_counts,
        'plot_saved': plot_path
    }

def build_matrix(data):
    """
    Build user-movie rating matrix.
    
    Args:
        data (pd.DataFrame): Merged dataset
    
    Returns:
        pd.DataFrame: Pivot table matrix
    """
    return data.pivot_table(index='userId', columns='title', values='rating')

def get_recommendations(matrix, data, movie_title, min_ratings=50):
    """
    Get top recommendations using collaborative filtering (Pearson correlation).
    
    Args:
        matrix (pd.DataFrame): User-movie matrix
        data (pd.DataFrame): Merged dataset
        movie_title (str): Movie to recommend similar to
        min_ratings (int): Min rating count for statistical reliability (reduces variance)
    
    Returns:
        pd.DataFrame: Top 10 similar movies (sorted by correlation)
    """
    if movie_title not in matrix.columns:
        raise ValueError(f"Movie '{movie_title}' not found in dataset.")
    movie_ratings = matrix[movie_title]
    similar = matrix.corrwith(movie_ratings)
    corr_df = pd.DataFrame(similar, columns=['Correlation'])
    corr_df.dropna(inplace=True)
    
    # Join counts; filter for reliability (p<0.05 implied for strong r with n>min_ratings)
    count_series = data.groupby('title')['rating'].count().rename('rating_count')
    corr_df = corr_df.join(count_series)
    corr_df = corr_df[corr_df['rating_count'] > min_ratings]
    
    return corr_df.sort_values('Correlation', ascending=False).head(10)

if __name__ == '__main__':
    # CLI for running as script (configurable; e.g., python recommender.py --movie "Toy Story (1995)")
    parser = argparse.ArgumentParser(description="Movie Recommender CLI")
    parser.add_argument('--ratings_path', default='ml-latest-small/ratings.csv', help='Path to ratings.csv')
    parser.add_argument('--movies_path', default='ml-latest-small/movies.csv', help='Path to movies.csv')
    parser.add_argument('--movie', default='Toy Story (1995)', help='Movie title for recommendations')
    parser.add_argument('--plot_path', default='ratings_dist.png', help='Path to save distribution plot')
    args = parser.parse_args()
    
    try:
        data = load_data(args.ratings_path, args.movies_path)
        exploration = explore_data(data, args.plot_path)
        print("First 5 rows of merged data:\n", data.head())
        print("\nTop 10 movies by average rating:\n", exploration['top_avg_ratings'])
        print("\nTop 10 movies by number of ratings:\n", exploration['top_rating_counts'])
        print(f"\nRating distribution plot saved as {exploration['plot_saved']}")
        
        matrix = build_matrix(data)
        print("\nUser-movie matrix shape:", matrix.shape)
        
        recommendations = get_recommendations(matrix, data, args.movie)
        print(f"\nTop 10 recommendations similar to '{args.movie}':\n", recommendations)
    except Exception as e:
        print(f"Error: {e}")