import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

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
def compute_genre_similarity(genres1, genres2):
    set1 = set(genres1.split('|'))
    set2 = set(genres2.split('|'))
    return len(set1 & set2) / len(set1 | set2)

def recommend_similar_movies(movie_title, num_recommendations=5):
    # Ensure the movie exists in the DataFrame
    movie_title = movie_title.strip().lower()
    movie_titles = movies['title'].str.lower()
    if movie_title not in movie_titles.values:
        st.write("Movie not found. Please check the spelling or try another movie.")
        return []
    
    movie_idx = movie_titles[movie_titles == movie_title].index[0]
    target_genres = movies.at[movie_idx, 'genres']
    
    # Compute similarity scores based on genres
    movies['similarity'] = movies['genres'].apply(lambda x: compute_genre_similarity(target_genres, x))
    similar_movies = movies.sort_values(by=['similarity', 'movieId'], ascending=[False, True])['title'].head(num_recommendations + 1)
    
    return similar_movies.iloc[1:].tolist()  # Exclude the input movie itself

# Streamlit app
st.title('Movie Recommendation System')

movie_name = st.text_input('Enter a movie name:')
if movie_name:
    recommendations = recommend_similar_movies(movie_name)
    if recommendations:
        st.write('Recommendations:')
        for rec in recommendations:
            st.write(rec)

genre_filter = st.selectbox('Select a genre:', sorted(set('|'.join(movies['genres']).split('|'))))
if genre_filter:
    filtered_movies = movies[movies['genres'].str.contains(genre_filter)]
    st.write(f'Movies with genre {genre_filter}:')
    for movie in filtered_movies['title'].head(10):  # Display top 10 movies for the selected genre
        st.write(movie)
