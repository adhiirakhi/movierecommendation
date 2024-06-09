import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocessing
movie_data = pd.merge(ratings, movies, on='movieId')
movie_data.dropna(inplace=True)
movie_data['rating'] = movie_data['rating'].astype(float)

# User-based recommendation system
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))

def recommend_movies(user_id, num_recommendations=5):
    user_idx = user_id - 1  # Adjust for zero-indexed
    similarity_scores = user_similarity[user_idx]
    similar_users = user_movie_matrix.index[similarity_scores.argsort()[-num_recommendations-1:-1]]
    
    # Movies watched by similar users
    recommendations = user_movie_matrix.loc[similar_users].mean().sort_values(ascending=False).head(num_recommendations)
    return recommendations

# Content-based recommendation system
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
movie_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_similar_movies(movie_title, num_recommendations=5):
    # Ensure the movie exists in the DataFrame
    if movie_title not in movies['title'].values:
        st.write("Movie not found. Please check the spelling or try another movie.")
        return []
    
    movie_idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(movie_similarity[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_movies = [movies['title'][i[0]] for i in similarity_scores[1:num_recommendations+1]]
    return similar_movies

# Streamlit app
st.title('Movie Recommendation System')

movie_name = st.text_input('Enter a movie name:')
if movie_name:
    recommendations = recommend_similar_movies(movie_name)
    if recommendations:
        st.write('Recommendations:')
        for rec in recommendations:
            st.write(rec)
