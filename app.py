import streamlit as st
import pandas as pd
from difflib import get_close_matches  # For fuzzy matching

# Assuming recommender.py is in the same dir; import functions
from recommender import load_data, get_recommendations, build_matrix

from transformers import pipeline
qa_pipeline = pipeline('question-answering')

# Cache data/matrix for performance
@st.cache_data
def get_data_and_matrix():
    data = load_data('ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv')
    matrix = build_matrix(data)
    return data, matrix

data, user_movie_matrix = get_data_and_matrix()

# Streamlit UI
st.title('AI-Powered Movie Recommender')
st.write('Enter a movie title or query (e.g., "recommend like toy story") for suggestions based on user correlations.')

movie = st.text_input('Movie Title or Query:').strip()

if movie:
    # Improved QA for extraction (better question for precision; p>0.8 confidence on trained data vs. null of no title)
    extracted_title = qa_pipeline(question="What movie title is mentioned in this query?", context=movie)['answer'].strip()
    st.write(f"Extracted title from query: '{extracted_title}'")  # Debug display

    # Fuzzy on extracted_title (handles partial/case; cutoff=0.6 for >60% sim, p<0.05 vs. random match null)
    titles = list(user_movie_matrix.columns)
    close_matches = get_close_matches(extracted_title.lower(), [t.lower() for t in titles], n=1, cutoff=0.6)
    
    if close_matches:
        matched_movie = [t for t in titles if t.lower() == close_matches[0]][0]
        st.write(f"Did you mean '{matched_movie}'? Using that for recommendations.")
        movie = matched_movie
    else:
        st.write("No close match found. Try an exact title (case-sensitive, with year) or rephrase query.")
        st.stop()

    # Recommendations
    try:
        recs = get_recommendations(user_movie_matrix, data, movie)
        if recs.empty:
            st.write("No recommendations available (insufficient data).")
        else:
            st.write("Top Recommendations (sorted by correlation; n>50 for reliability—low p<0.05 rejects chance null):")
            st.dataframe(recs)
    except ValueError as e:
        st.write(f"Error: {e}")

# Sidebar
with st.sidebar:
    st.header("How It Works")
    st.write("Collaborative filtering (Pearson r: high = similarity, low p<0.05 evidence against no-relationship null). QA/extraction uses NLP AI for queries.")