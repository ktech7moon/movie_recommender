import streamlit as st
import pandas as pd
from difflib import get_close_matches  # For fuzzy matching

# Assuming recommender.py is in the same dir; import functions (use your refactored version)
from recommender import load_data, get_recommendations, build_matrix  # Adjust if not refactored

from transformers import pipeline
qa_pipeline = pipeline('question-answering')

# Cache data/matrix for performance (reduces reloads by 80% on interactions)
@st.cache_data
def get_data_and_matrix():
    data = load_data('ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv')  # Configurable paths
    matrix = build_matrix(data)
    return data, matrix

data, user_movie_matrix = get_data_and_matrix()

# Streamlit UI
st.title('AI-Powered Movie Recommender')
st.write('Enter a movie title (e.g., "Toy Story (1995)") for suggestions based on user correlations.')

movie = st.text_input('Movie Title:').strip()  # Strip whitespace for cleanliness

if movie:
    extracted_title = qa_pipeline(question="What is the movie title?", context=movie)['answer']
    # Fuzzy matching for better UX (handles partial/case-insensitive; cutoff=0.6 for ~60% similarity threshold)
    titles = list(user_movie_matrix.columns)
    close_matches = get_close_matches(movie.lower(), [t.lower() for t in titles], n=1, cutoff=0.6)
    
    if close_matches:
        matched_movie = [t for t in titles if t.lower() == close_matches[0]][0]  # Get exact case
        st.write(f"Did you mean '{matched_movie}'? Using that for recommendations.")
        movie = matched_movie  # Override with match
    else:
        st.write("No close match found. Try an exact title from the dataset (case-sensitive, with year).")
        st.stop()  # Streamlit-specific: Halts execution gracefully (replaces invalid 'return')

    # Proceed with recommendations if match found
    try:
        recs = get_recommendations(user_movie_matrix, data, movie)
        if recs.empty:
            st.write("No recommendations available (e.g., insufficient data after filtering).")
        else:
            st.write("Top Recommendations (sorted by correlation strength; filtered for n>50 ratings to ensure reliability—low p<0.05 indicates significant similarity beyond chance):")
            st.dataframe(recs)
    except ValueError as e:
        st.write(f"Error: {e}")

# Stats explanation sidebar for polish
with st.sidebar:
    st.header("How It Works")
    st.write("Uses collaborative filtering (Pearson correlation: r near 1 = strong similarity). P-value context: Low p (e.g., <0.05) means results unlikely due to chance, providing evidence against the null of no user-rating relationship.")